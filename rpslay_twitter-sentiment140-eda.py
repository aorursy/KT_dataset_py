import warnings



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.options.display.max_colwidth = 255



from pandas_profiling import ProfileReport

from tqdm.notebook import tqdm



import transformers

from tokenizers import BertWordPieceTokenizer



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam



warnings.simplefilter("ignore")

##

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

DATASET_ENCODING = "ISO-8859-1"



BATCH_SIZE = 64

AUTO = tf.data.experimental.AUTOTUNE

TEST_SIZE = 0.1

RANDOM_STATE = 42

MAX_SEQ_LEN = 150

PRETRAINED_MODEL = 'bert-base-uncased'

N_EPOCHS = 10
data = pd.read_csv("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv", 

                   header=None,

                   encoding=DATASET_ENCODING)

data.columns = DATASET_COLUMNS

data = data[['text', 'target']]

data['target'] = data['target'].map(lambda x: 1.0 if x == 4 else (0. if x == 0 else np.NaN))
data.head()
data["contains_mention"] = data.text.str.contains("@")

data["contains_hashtag"] = data.text.str.contains("#")

data["contains_link"] = data.text.str.contains("http")


# ProfileReport(data[['target', 'contains_mention', 'contains_hashtag', 'contains_link']])
data['n_words'] = data['text'].map(lambda x: len(x.split(' ')))

data['n_chars'] = data['text'].map(len)
# ProfileReport(data[['target', 'n_words', 'n_chars']])
import re



mention_regex = re.compile("\@([a-zA-Z1-9]+)", flags=re.IGNORECASE)

hashtag_regex = re.compile("\#([a-zA-Z1-9]+)", flags=re.IGNORECASE)

link_regex = re.compile("http(s?):\/\/[^\s]+", flags=re.IGNORECASE)





def remove_mentions(s: str) -> str:

    return mention_regex.sub(" ", s)



def remove_hashtags(s: str) -> str:

    return hashtag_regex.sub(" ", s)



def remove_links(s: str) -> str:

    return link_regex.sub(" ", s)
def clean_text(s: str) -> str:

    s = remove_mentions(s)

    s = remove_hashtags(s)

    s = remove_links(s)

    

    s = s.lower() # task specific

    

    return s
def load_tokenizer(model_name: str) -> BertWordPieceTokenizer:

    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    

    save_path = f'/kaggle/working/{model_name}/'

    if not os.path.exists(save_path):

        os.makedirs(save_path)

    tokenizer.save_pretrained(save_path)

    

    piece_tokenizer = BertWordPieceTokenizer(f'/kaggle/working/{model_name}/vocab.txt', lowercase=False)

    return piece_tokenizer
def prepare_texts(texts:pd.Series, tokenizer: BertWordPieceTokenizer, chunk_size: int=256, max_length:int=512):

    tokenizer.enable_truncation(max_length=max_length)

    tokenizer.enable_padding(max_length=max_length)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
tokenizer = load_tokenizer(PRETRAINED_MODEL)
clean_texts = data['text'].map(clean_text)
X = prepare_texts(clean_texts, tokenizer, max_length=MAX_SEQ_LEN)

y = data['target'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



val_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_val, y_val))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



def classifier_model(x):

    x = tf.keras.layers.Dropout(0.35)(x)

    x = Dense(1, activation='sigmoid')(x)

    return x
def load_pretrained_encoder_model(model_name):

    model = transformers.TFBertModel.from_pretrained(model_name)

    

    return model
def build_encoder_classifier_model(encoder, classifier, loss='binary_crossentropy', max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = encoder(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    

    out = classifier(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=3e-5), loss=loss, metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()])

    

    return model
encoder = load_pretrained_encoder_model(PRETRAINED_MODEL)

model = build_encoder_classifier_model(encoder, classifier_model, max_len=MAX_SEQ_LEN)

model.summary()
train_history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    steps_per_epoch=250,

    validation_steps=75,

    epochs=N_EPOCHS

)
model.save('./sent_classifier.model')
!ls
!tar -czvf sent_classifier_model.tar.gz sent_classifier.model/
!rm -r sent_classifier.model/