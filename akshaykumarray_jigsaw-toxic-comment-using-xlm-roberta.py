# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path="/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"
training_set=pd.read_csv(path+"jigsaw-toxic-comment-train.csv")

test_set=pd.read_csv(path+"test.csv")

validation_set=pd.read_csv(path+"validation.csv")

sub=pd.read_csv(path+"sample_submission.csv")
training_set.head()
test_set.head()
validation_set.head()
training_set.isnull().sum()
test_set.isnull().sum()
validation_set.isnull().sum()
training_set.drop(["severe_toxic","obscene","threat","insult","identity_hate","id"],axis=1,inplace=True)

training_set.head()
training_set.head()
#training_set inspection

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

sns.countplot(x=training_set["toxic"])

plt.show()

sns.countplot(x=validation_set["toxic"])

plt.show()
#test_set different language

print(test_set["lang"].unique())

sns.countplot(x=test_set["lang"])

plt.show()
#validation set

print(validation_set["lang"].unique())

sns.countplot(x=validation_set["lang"])

plt.show()
#dropping lang,id  from both dataset

test_set.drop(["lang","id"],axis=1,inplace=True)

validation_set.drop(["lang","id"],axis=1,inplace=True)
import re

def clean_text(text):

    text=text.lower()

    text=re.sub("\n"," ",text)

    text=re.sub("\[\[User.*"," ",text)

    text=re.sub("\(http://.*?\s\(http://.*\)"," ",text)

    text=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"," ",text)

    text=re.sub("[.,]"," ",text)

    text = re.sub(r'[0-9"]', '', text)

    text = re.sub(r'#[\S]+\b', '', text)

    text = re.sub(r'@[\S]+\b', '', text)

    text = re.sub(r'https?\S+', '', text)

    text = re.sub(r'\s+', ' ', text)

    text=text.split()

    text=" ".join(text)

    return text
training_set["comment_text"]=training_set["comment_text"].apply(str).apply(lambda x:clean_text(x))

test_set["content"]=test_set["content"].apply(str).apply(lambda x:clean_text(x))

validation_set["comment_text"]=validation_set["comment_text"].apply(str).apply(lambda x:clean_text(x))
training_set.head()
test_set.head()
validation_set.head()
from wordcloud import WordCloud,STOPWORDS

stopwords=set(STOPWORDS)

wc=WordCloud(width=800,height=800,stopwords=stopwords,background_color="white",max_words=100,min_font_size=10).generate(str(training_set["comment_text"]))

plt.figure(figsize=(10,10))

plt.imshow(wc,interpolation="bilinear")

plt.axis("off")

plt.tight_layout(pad = 0) 

plt.show()
from wordcloud import WordCloud,STOPWORDS

stopwords=set(STOPWORDS)

wc=WordCloud(width=800,height=800,stopwords=stopwords,background_color="white",max_words=100,min_font_size=10).generate(str(validation_set["comment_text"]))

plt.figure(figsize=(10,10))

plt.imshow(wc,interpolation="bilinear")

plt.axis("off")

plt.tight_layout(pad = 0) 

plt.show()
from keras.models import Sequential

from keras.layers import Embedding

from keras.layers import Dense

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Flatten

from keras.layers import SimpleRNN

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])
def build_model(transformer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
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
AUTO = tf.data.experimental.AUTOTUNE

GCS_DS_PATH = KaggleDatasets().get_gcs_path()

EPOCHS = 3

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192

MODEL = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
x_train = regular_encode(training_set.comment_text.values, tokenizer, maxlen=MAX_LEN)

x_valid = regular_encode(validation_set.comment_text.values, tokenizer, maxlen=MAX_LEN)

x_test = regular_encode(test_set.content.values, tokenizer, maxlen=MAX_LEN)



y_train = training_set.toxic.values

y_valid = validation_set.toxic.values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(1024)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)
test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
with strategy.scope():

    transformer_layer = TFAutoModel.from_pretrained(MODEL)

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)
n_steps = x_valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=EPOCHS

)
sub['toxic'] = model.predict(test_dataset)

sub.head()

sub.to_csv('submission.csv', index=False)