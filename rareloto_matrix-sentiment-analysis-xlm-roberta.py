import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('../input/shopee-sentiment-analysis/train.csv')

test_df = pd.read_csv('../input/shopee-sentiment-analysis/test.csv')

print(train_df.shape, test_df.shape)
train_df.head()
test_df.head()
dup_df = train_df[train_df['review'].duplicated()]

print(f'No. of duplicate reviews on train data: {dup_df.shape[0]}')
dup_df['check'] = dup_df.apply(lambda x: str(x.review) + str(x.rating), axis = 1)

print(dup_df['check'].duplicated().sum(),'of duplicate reviews have the same rating')
train_df.drop_duplicates(subset = 'review', inplace = True)
train_df['rating'].value_counts()
count_df = train_df.groupby(['rating']).count()

count_df.drop(['review_id'], axis = 1, inplace = True)

count_df['percentage'] = 100 * count_df['review']  / count_df['review'].sum()

count_df
train_df['rating'].hist()
from nltk import word_tokenize



def count_len(text):

    return len(word_tokenize(text))
train_df['len'] = train_df['review'].apply(count_len)

test_df['len'] = test_df['review'].apply(count_len)
train_df['len'].hist()
test_df['len'].hist()
# kaggle.com/liuhh02/test-labelled

# old test leak labelled

test_labelled = pd.read_csv('../input/test-labelled/test_labelled.csv')

test_labelled
dup_testlab = test_labelled[test_labelled['review'].duplicated()]

print(f'No. of duplicate reviews: {dup_testlab.shape[0]}')

dup_testlab['check'] = dup_testlab.apply(lambda x: str(x.review) + str(x.rating), axis = 1)

print(dup_testlab['check'].duplicated().sum(),'of duplicate reviews have the same rating')
test_labelled.drop_duplicates(subset = 'review', inplace = True)
# kaggle.com/shymammoth/shopee-reviews

# scraped shopee reviews

scraped_reviews = pd.read_csv('../input/shopee-reviews/shopee_reviews.csv')

scraped_reviews
scraped_reviews.rename(columns = {'text': 'review', 'label': 'rating'}, inplace = True)

scraped_reviews.info()
scraped_reviews['rating'].value_counts()
scraped_reviews = scraped_reviews[scraped_reviews['rating'] != 'label']

scraped_reviews['rating'] = scraped_reviews['rating'].astype(int)

scraped_reviews['rating'].value_counts()
dup_scraped = scraped_reviews[scraped_reviews['review'].duplicated()]

print(f'No. of duplicate reviews: {dup_scraped.shape[0]}')
train_df = train_df.append(test_labelled, ignore_index = True)

train_df = train_df.append(scraped_reviews, ignore_index = True)

train_df = train_df.sample(frac = 1).reset_index(drop = True)
dup_train = train_df[train_df['review'].duplicated()]

print(f'No. of duplicate reviews on train data: {dup_train.shape[0]}')

dup_train['check'] = dup_train.apply(lambda x: str(x.review) + str(x.rating), axis = 1)

print(dup_train['check'].duplicated().sum(),'of duplicate reviews have the same rating')
train_df
train_df['rating'].value_counts()
train_df = train_df.drop(train_df[train_df['rating'] == 5].sample(frac = .6).index)

print(train_df['rating'].value_counts())

print(train_df.shape)
# adding column 'rating' to test dataset

test_df['rating'] = -1 # flag to separate train and test



# joining train and test datasets

reviews = pd.concat([train_df, test_df], ignore_index = True)

reviews
from nltk.tokenize import word_tokenize

from nltk import FreqDist

from nltk.stem import SnowballStemmer, WordNetLemmatizer



stemmer = SnowballStemmer('english')

lemma = WordNetLemmatizer()



from string import punctuation
def clean_review(review_col):

    review_corpus=[]

    

    for i in range(0, len(review_col)):

        review = str(review_col[i])

        review = re.sub('[^a-zA-Z]', ' ', review)

        review = [lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]

        review = ' '.join(review)

        

        review_corpus.append(review)

        

    return review_corpus
import emoji  



have_emoji_train_idx = []



for idx, review in enumerate(reviews['review']):

    if any(char in emoji.UNICODE_EMOJI for char in review):

        have_emoji_train_idx.append(idx)
def emoji_cleaning(text):

    

    # change emoji to text

    text = emoji.demojize(text).replace(":", " ")

    

    # delete repeated emoji

    tokenizer = text.split()

    repeated_list = []

    

    for word in tokenizer:

        if word not in repeated_list:

            repeated_list.append(word)

    

    text = ' '.join(text for text in repeated_list)

    text = text.replace("_", " ").replace("-", " ")

    return text
# emoji_cleaning

reviews.loc[have_emoji_train_idx, 'review'] = reviews.loc[have_emoji_train_idx, 'review'].apply(emoji_cleaning)
def review_cleaning(text):

    

    text = text.lower()

    text = re.sub(r'\n', '', text)

    

#     text = text.replace("n't", ' not')

    

    # change emoticon to text

    text = re.sub(r':\(', 'dislike', text)

    text = re.sub(r': \(\(', 'dislike', text)

    text = re.sub(r':, \(', 'dislike', text)

    text = re.sub(r':\)', 'smile', text)

    text = re.sub(r';\)', 'smile', text)

    text = re.sub(r':\)\)\)', 'smile', text)

    text = re.sub(r':\)\)\)\)\)\)', 'smile', text)

    text = re.sub(r'=\)\)\)\)', 'smile', text)

    

#     # delete punctuation

#     text = re.sub('[^a-z0-9 ]', ' ', text)

    

    tokenizer = text.split()

    

    return ' '.join([text for text in tokenizer])
reviews['review'] = reviews['review'].apply(review_cleaning)
repeated_rows_train = []



for idx, review in enumerate(reviews['review']):

    if re.match(r'\w*(\w)\1+', review):

        repeated_rows_train.append(idx)
def delete_repeated_char(text):

    

    text = re.sub(r'(\w)\1{2,}', r'\1', text)

    

    return text
reviews.loc[repeated_rows_train, 'review'] = reviews.loc[repeated_rows_train, 'review'].apply(delete_repeated_char)
def recover_shortened_words(text):

    

    text = re.sub(r'\bapaa\b', 'apa', text)

    

    text = re.sub(r'\bbsk\b', 'besok', text)

    text = re.sub(r'\bbrngnya\b', 'barangnya', text)

    text = re.sub(r'\bbrp\b', 'berapa', text)

    text = re.sub(r'\bbgt\b', 'banget', text)

    text = re.sub(r'\bbngt\b', 'banget', text)

    text = re.sub(r'\bgini\b', 'begini', text)

    text = re.sub(r'\bbrg\b', 'barang', text)

    

    text = re.sub(r'\bdtg\b', 'datang', text)

    text = re.sub(r'\bd\b', 'di', text)

    text = re.sub(r'\bsdh\b', 'sudah', text)

    text = re.sub(r'\bdri\b', 'dari', text)

    text = re.sub(r'\bdsni\b', 'disini', text)

    

    text = re.sub(r'\bgk\b', 'gak', text)

    

    text = re.sub(r'\bhrs\b', 'harus', text)

    

    text = re.sub(r'\bjd\b', 'jadi', text)

    text = re.sub(r'\bjg\b', 'juga', text)

    text = re.sub(r'\bjgn\b', 'jangan', text)

    

    text = re.sub(r'\blg\b', 'lagi', text)

    text = re.sub(r'\blgi\b', 'lagi', text)

    text = re.sub(r'\blbh\b', 'lebih', text)

    text = re.sub(r'\blbih\b', 'lebih', text)

    

    text = re.sub(r'\bmksh\b', 'makasih', text)

    text = re.sub(r'\bmna\b', 'mana', text)

    

    text = re.sub(r'\borg\b', 'orang', text)

    

    text = re.sub(r'\bpjg\b', 'panjang', text)

    

    text = re.sub(r'\bka\b', 'kakak', text)

    text = re.sub(r'\bkk\b', 'kakak', text)

    text = re.sub(r'\bklo\b', 'kalau', text)

    text = re.sub(r'\bkmrn\b', 'kemarin', text)

    text = re.sub(r'\bkmrin\b', 'kemarin', text)

    text = re.sub(r'\bknp\b', 'kenapa', text)

    text = re.sub(r'\bkcil\b', 'kecil', text)

    

    text = re.sub(r'\bgmn\b', 'gimana', text)

    text = re.sub(r'\bgmna\b', 'gimana', text)

    

    text = re.sub(r'\btp\b', 'tapi', text)

    text = re.sub(r'\btq\b', 'thanks', text)

    text = re.sub(r'\btks\b', 'thanks', text)

    text = re.sub(r'\btlg\b', 'tolong', text)

    text = re.sub(r'\bgk\b', 'tidak', text)

    text = re.sub(r'\bgak\b', 'tidak', text)

    text = re.sub(r'\bgpp\b', 'tidak apa apa', text)

    text = re.sub(r'\bgapapa\b', 'tidak apa apa', text)

    text = re.sub(r'\bga\b', 'tidak', text)

    text = re.sub(r'\btgl\b', 'tanggal', text)

    text = re.sub(r'\btggl\b', 'tanggal', text)

    text = re.sub(r'\bgamau\b', 'tidak mau', text)

    

    text = re.sub(r'\bsy\b', 'saya', text)

    text = re.sub(r'\bsis\b', 'sister', text)

    text = re.sub(r'\bsdgkan\b', 'sedangkan', text)

    text = re.sub(r'\bmdh2n\b', 'semoga', text)

    text = re.sub(r'\bsmoga\b', 'semoga', text)

    text = re.sub(r'\bsmpai\b', 'sampai', text)

    text = re.sub(r'\bnympe\b', 'sampai', text)

    text = re.sub(r'\bdah\b', 'sudah', text)

    

    text = re.sub(r'\bberkali2\b', 'repeated', text)

    

    text = re.sub(r'\byg\b', 'yang', text)

    

    return text
%%time

reviews['review'] = reviews['review'].apply(recover_shortened_words)
# cleaning round 2, lemmatization

reviews['review'] = clean_review(reviews['review'].values)

reviews
train_df = reviews[reviews.rating != -1]

train_df.drop(['review_id', 'len'], axis = 1, inplace = True)

train_df.head()
test_df = reviews[reviews.rating == -1]

test_df.drop(['rating', 'len'], axis = 1, inplace = True)

test_df['review_id'] = test_df['review_id'].astype(int)

test_df.head()
train_df[['review', 'rating']].to_csv('clean_extended_train.csv', index = False)

test_df[['review_id', 'review']].to_csv('clean_test_up.csv', index = False)
# train_df = pd.read_csv('../input/shopee-code-league-2020-sentiment-analysis/clean_extended_train.csv').fillna('')

# test_df = pd.read_csv('../input/shopee-code-league-2020-sentiment-analysis/clean_test_up.csv').fillna('')

# print(train_df.shape, test_df.shape)
train_df.head()
test_df.head()
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



print('Using Tensorflow version:', tf.__version__)
def regular_encode(texts, tokenizer, maxlen=512):

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



print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



# Configuration

EPOCHS = 4

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MODEL = 'jplu/tf-xlm-roberta-large' # bert-base-multilingual-uncased
# since keras takes 0 as the reference, our category should start from 0 not 1

rating_mapper_encode = {1: 0,

                        2: 1,

                        3: 2,

                        4: 3,

                        5: 4}



# convert back to original rating after prediction later

rating_mapper_decode = {0: 1,

                        1: 2,

                        2: 3,

                        3: 4,

                        4: 5}



train_df['rating'] = train_df['rating'].map(rating_mapper_encode)
from tensorflow.keras.utils import to_categorical



# convert to one-hot-encoding-labels

train_labels = to_categorical(train_df['rating'], num_classes=5)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(train_df['review'],

                                                  train_labels,

                                                  stratify=train_labels,

                                                  test_size=0.1,

                                                  random_state=1111)



X_train.shape, X_val.shape, y_train.shape, y_val.shape
# load tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL)
MAX_LEN = 104 # chosen from EDA



X_train = regular_encode(X_train.values, tokenizer, maxlen=MAX_LEN)

X_val = regular_encode(X_val.values, tokenizer, maxlen=MAX_LEN)

X_test = regular_encode(test_df['review'].values, tokenizer, maxlen=MAX_LEN)
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

    epochs=EPOCHS

)
plt.style.use('fivethirtyeight')



# Get training and test loss histories

training_loss = train_history.history['loss']

test_loss = train_history.history['val_loss']



# Create count of the number of epochs

epoch_count = range(1, len(training_loss) + 1)



# Visualize loss history

plt.plot(epoch_count, training_loss, 'r--')

plt.plot(epoch_count, test_loss, 'b-')

plt.legend(['Training Loss', 'Test Loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()
pred = model.predict(test_dataset, verbose=1)
# for ensemble

np.save('xlm-roberta', pred)
pred_sentiment = np.argmax(pred, axis=1)



print(pred_sentiment)
submission = pd.DataFrame({'review_id': test_df['review_id'],

                           'rating': pred_sentiment})
submission['rating'] = submission['rating'].map(rating_mapper_decode)

submission.to_csv('submission.csv', index=False)

submission['rating'].value_counts()