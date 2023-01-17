!pip install transformers
import os
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tokenizers import BertWordPieceTokenizer
from tqdm.notebook import tqdm
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import transformers
from transformers import TFAutoModel, AutoTokenizer
import matplotlib.pyplot as plt
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()
test=pd.read_csv(r'../input/nlp-getting-started/test.csv')
train=pd.read_csv(r'../input/nlp-getting-started/train.csv')
sample_submission=pd.read_csv(r'../input/nlp-getting-started/sample_submission.csv')
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
# dropping id, location column due to large no of Nan.

train.drop(['id','location'],axis=1,inplace=True)
test.drop(['id','location'],axis=1,inplace=True)
x=0  # counter for rows containing 'ablaze' keyword and target=1
y=0  # counter for total rows having keyword 'ablaze'
for i in range(len(train)):
    if (train['keyword'].iloc[i]=='ablaze'):
        x+=train['target'].iloc[i]
        y+=1
x,y
train.drop(['keyword'],axis=1,inplace=True)
test.drop(['keyword'],axis=1,inplace=True)
train['target'].value_counts()
!pip install clean-text[gpl]
from cleantext import clean
def text_cleaning(text):
    text=clean(text,
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=True,               # replace all numbers with a special token
    no_digits=True,                # replace all digits with a special token
    no_currency_symbols=True,      # replace all currency symbols with a special token
    no_punct=True,                 # fully remove punctuation
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"                       # set to 'de' for German special handling
    )
    return text
for i in range(len(train)):
    train['text'].iloc[i]=text_cleaning(train['text'].iloc[i])

for i in range(len(test)):
    test['text'].iloc[i]=text_cleaning(test['text'].iloc[i])  
train['text']
def build_model(transformer, max_len=512): 
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    x = tf.keras.layers.Dropout(0.35)(cls_token)
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=3e-5), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    
    return model
with strategy.scope():
    transformer_layer = transformers.TFBertModel.from_pretrained('bert-base-uncased')
    model = build_model(transformer_layer, max_len=512)
model.summary()
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
save_path = 'distilbert_base_uncased/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer.save_pretrained(save_path)
fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', lowercase=True)
fast_tokenizer
def fast_encode(texts, tokenizer, size=256, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    ids_full = []
    
    for i in tqdm(range(0, len(texts), size)):
        text = texts[i:i+size].tolist()
        encs = tokenizer.encode_batch(text)
        ids_full.extend([enc.ids for enc in encs])
    
    return np.array(ids_full)
x = fast_encode(train.text.astype(str), fast_tokenizer, maxlen=512)
x_test = fast_encode(test.text.astype(str), fast_tokenizer, maxlen=512)
BATCH_SIZE=64

test_data = (
    tf.data.Dataset# create dataset
    .from_tensor_slices(x_test) 
    .batch(BATCH_SIZE)
)
y=train['target'].values
train_dataset = (
    tf.data.Dataset 
      .from_tensor_slices((x, y))
      .repeat()
      .shuffle(2048)
      .batch(BATCH_SIZE)
    .prefetch(tf.data.experimental.AUTOTUNE) 
)
with strategy.scope():
    train_history = model.fit(
      train_dataset,

      steps_per_epoch=150,

      epochs=3
    )
final=sample_submission[['id']]
final['target'] = model.predict(test_data, verbose=1)
def replace(col_val):
    if col_val >=0.5:
        col_val=1
    else:
        col_val=0
    return col_val
final['target']=final['target'].apply(lambda x : replace(x))
final['target'].value_counts()
final
final.to_csv('sub_1.csv', index=False)
