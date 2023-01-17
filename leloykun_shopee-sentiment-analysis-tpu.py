!pip install nlpaug -q
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('use_inf_as_na', True)

import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')

import emoji

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

plt.rcParams['figure.figsize'] = [14, 8]

pd.set_option('display.max_colwidth', -1)

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

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

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

print('Using Tensorflow version:', tf.__version__)
df_test  = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv')
df_train_orig = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/train.csv', usecols=['review', 'rating'])

df_train_scraped = pd.read_csv('/kaggle/input/shopee-reviews/shopee_reviews.csv')[['text', 'label']]
df_train_scraped.columns = ['review', 'rating']

df_train_aug_sub = pd.read_csv('/kaggle/input/text-augmented/review_aug_sub.csv')
df_train_aug_insert = pd.read_csv('/kaggle/input/text-augmented/review_aug_insert.csv')
df_train = pd.concat([df_train_orig, df_train_scraped,
                df_train_aug_sub, df_train_aug_insert], axis=0, ignore_index=True)
df_train['review'].astype(str)
df_train = df_train.dropna()
df_train.tail()
df_train = df_train.reset_index().drop('index', axis=1)
df_test = df_test.reset_index().drop('index', axis=1)
%%time

have_emoji_train_idx = []
have_emoji_test_idx = []

for idx, review in enumerate(df_train['review']):
    review = str(review)
    if any(char in emoji.UNICODE_EMOJI for char in review):
        have_emoji_train_idx.append(idx)

for idx, review in enumerate(df_test['review']):
    if any(char in emoji.UNICODE_EMOJI for char in review):
        have_emoji_test_idx.append(idx)
def emoji_cleaning(text):
    # Change emoji to text
    text = emoji.demojize(text).replace(":", " ")
    # Delete repeated emoji
    tokenizer = text.split()
    repeated_list = []
    for word in tokenizer:
        if word not in repeated_list:
            repeated_list.append(word)
    text = ' '.join(text for text in repeated_list)
    text = text.replace("_", " ").replace("-", " ")
    return text
%%time
df_train.loc[have_emoji_train_idx, 'review'] = df_train.loc[have_emoji_train_idx, 'review'].apply(emoji_cleaning)
df_test.loc[have_emoji_test_idx, 'review'] = df_test.loc[have_emoji_test_idx, 'review'].apply(emoji_cleaning)
def review_cleaning(text):
    # delete lowercase and newline
    text = str(text)
    text = text.lower()
    text = re.sub(r'\n', '', text)
    # change emoticon to text
    text = re.sub(r':\(', 'dislike', text)
    text = re.sub(r': \(\(', 'dislike', text)
    text = re.sub(r':, \(', 'dislike', text)
    text = re.sub(r':\)', 'smile', text)
    text = re.sub(r';\)', 'smile', text)
    text = re.sub(r':\)\)\)', 'smile', text)
    text = re.sub(r':\)\)\)\)\)\)', 'smile', text)
    text = re.sub(r'=\)\)\)\)', 'smile', text)
    # delete punctuation
    text = re.sub('[^a-z0-9 ]', ' ', text)
    
    tokenizer = text.split()
    return ' '.join([text for text in tokenizer])
%%time
df_train['review'] = df_train['review'].apply(review_cleaning)
df_test['review']  = df_test['review'].apply(review_cleaning)
%%time
repeated_rows_train = []
repeated_rows_test = []

for idx, review in enumerate(df_train['review']):
    if re.match(r'\w*(\w)\1+', review):
        repeated_rows_train.append(idx)
        
for idx, review in enumerate(df_test['review']):
    if re.match(r'\w*(\w)\1+', review):
        repeated_rows_test.append(idx)
def delete_repeated_char(text):
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    return text
df_train.loc[repeated_rows_train, 'review'] = df_train.loc[repeated_rows_train, 'review'].apply(delete_repeated_char)
df_test.loc[repeated_rows_test, 'review'] = df_test.loc[repeated_rows_test, 'review'].apply(delete_repeated_char)
df_train.to_csv("train_clean.csv")
df_test.to_csv("test_clean.csv")
!gsutil cp train_clean.csv gs://shopee-sentiment-analysis
!gsutil cp test_clean.csv gs://shopee-sentiment-analysis
!gsutil cp gs://shopee-sentiment-analysis/train_clean.csv train_clean.csv
!gsutil cp gs://shopee-sentiment-analysis/test_clean.csv test_clean.csv
df_train = pd.read_csv('train_clean.csv')
df_test  = pd.read_csv('test_clean.csv')
def regular_encode(texts, tokenizer, maxlen=256):
    enc_di = tokenizer.batch_encode_plus(
             texts, 
             return_attention_masks=False, 
             return_token_type_ids=False,
             pad_to_max_length=True,
             max_length=maxlen)
    return np.array(enc_di['input_ids'])
def build_model(transformer, max_len=512):
    
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(5, activation='softmax')(cls_token) # 5 ratings to predict
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Configuration
EPOCHS = 5
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
MODEL = 'jplu/tf-xlm-roberta-large' # bert-base-multilingual-uncased
# since tf.keras reads your data take 0 as the reference, our category should start from 0 not 1
rating_mapper_encode = {1: 0,
                        2: 1,
                        3: 2,
                        4: 3,
                        5: 4}

# convert back to original rating after prediction later(dont forget!!)
rating_mapper_decode = {0: 1,
                        1: 2,
                        2: 3,
                        3: 4,
                        4: 5}

df_train['rating'] = df_train['rating'].map(rating_mapper_encode)
df_train = df_train.dropna()
from tensorflow.keras.utils import to_categorical

# convert to one-hot-encoding-labels
train_labels = to_categorical(df_train['rating'].astype('int'), num_classes=5)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df_train['review'],
                                                  train_labels,
                                                  stratify=train_labels,
                                                  test_size=0.1,
                                                  random_state=2020)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
MAX_LEN = 64

X_train = regular_encode(X_train.values.astype(str), tokenizer, maxlen=MAX_LEN)
X_val = regular_encode(X_val.values, tokenizer, maxlen=MAX_LEN)
X_test = regular_encode(df_test['review'].values, tokenizer, maxlen=MAX_LEN)
#class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(y_train.argmax(axis=1)),
#                                                  y_train.argmax(axis=1))
#class_weights = {i : class_weights[i] for i in range(len(class_weights))}
class_weights = {0 : 0.11388,
                 1 : 0.02350,
                 2 : 0.06051,
                 3 : 0.39692,
                 4 : 0.40519}
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .repeat()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test)
    .batch(BATCH_SIZE)
)
%%time

with strategy.scope():
    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
n_steps = X_train.shape[0] // BATCH_SIZE

train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=4,
    class_weight=class_weights,
)
X_test = regular_encode(df_test['review'].astype(str).values, tokenizer, maxlen=MAX_LEN)
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test)
    .batch(BATCH_SIZE)
)
pred = model.predict(test_dataset, verbose=1)
pred_sentiment = np.argmax(pred, axis=1)

print(pred_sentiment)
pred_sentiment.shape
submission = pd.DataFrame({'review_id': df_test['review_id'],
                           'rating': pred_sentiment})
submission['rating'] = submission['rating'].map(rating_mapper_decode)

submission.to_csv('submission.csv', index=False)
!gsutil cp submission.csv gs://shopee-sentiment-analysis
