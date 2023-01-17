import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import re
import gc

from nltk.corpus import stopwords
from tqdm import tqdm
sns.set()

stopwords = set(stopwords.words('english'))

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
# # detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
tweet = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
df = pd.concat([tweet,test])
df.fillna('', inplace=True)
df['text'] = df['keyword']+' '+df['text']

df['text'] = df['text'].str.replace('\%20',' ')
df['text'] = df['text'].str.replace('\x89ûò','')
maxlen = df.text.str.len().max()
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
df['text'] = df['text'].apply(lambda x : remove_URL(x))
df['text'] = df['text'].apply(lambda x : remove_html(x))
df['text'] = df['text'].apply(lambda x : remove_emoji(x))
df['text'] = df['text'].apply(lambda x : remove_punct(x))
df['text'] = df['text'].apply(lambda x: x.lower())
train = df[:tweet.shape[0]]
test = df[tweet.shape[0]:]

del df
gc.collect()
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/tokenization/tokenization.py", dst = "../working/tokenization.py")

import tokenization
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):
#     with tpu_strategy.scope():

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    return model
%%time
module_url = "/kaggle/input/bert-uncased-l24-h1024-a16/"
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.text.values, tokenizer, max_len=maxlen)

train_labels = train.target.values
train_labels = train_labels.astype(np.int8)

test_input = bert_encode(test.text.values, tokenizer, max_len=maxlen)

del train, test
gc.collect()
model = build_model(bert_layer, max_len=maxlen)
gc.collect()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=20,
    callbacks = [callback],
    batch_size=32,
    verbose=1,
)

model.save('model.h5')
test_pred = model.predict(test_input)
gc.collect()
submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission.csv', index=False)
submission.head()