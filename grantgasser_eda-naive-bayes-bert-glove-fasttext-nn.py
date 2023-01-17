import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re

from tqdm import tqdm

import tensorflow as tf



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
DATA_PATH = '/kaggle/input/nlp-getting-started/'

SEED = 42

DROPOUT = 0.5

EPOCHS = 3

LEARN_RATE = 1e-4

SPLIT = 0.2

BATCH_SIZE = 32
train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

sample_submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

ds = ['train', 'test', 'sample submission']

print("Training set has {} rows and {} columns.".format(train.shape[0], train.shape[1]))

print("Test set has {} rows and {} columns.".format(test.shape[0], test.shape[1]))



print()

print(train.columns)

print(test.columns)
train[train.target == 0].head()
train.head()
print('Train Keyword Distribution:\n\n')

print(train.keyword.value_counts())

print('\n', '-' * 50, '\n')

print('Test Keyword Distribution:\n\n')

print(test.keyword.value_counts())
temp = train['target'].value_counts(dropna = False).reset_index()

temp.columns = ['target', 'counts']



countplt = sns.countplot(x = 'target', data = train, hue = train['target'])

countplt.set_xticklabels(['0: Not Disaster (4342)', '1: Disaster (3271)'])
train['target_mean'] = train.groupby('keyword')['target'].transform('mean')



fig = plt.figure(figsize=(8, 72), dpi=100)



sns.countplot(y=train.sort_values(by='target_mean', ascending=False)['keyword'],

              hue=train.sort_values(by='target_mean', ascending=False)['target'])



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc=1)

plt.title('Target Distribution in Keywords')



plt.show()



train.drop(columns=['target_mean'], inplace=True)
print('Count NaN:')

print(train.isnull().sum(), '\n')

print('Percentage NaN:')

print(train.isnull().sum()/ len(train))
print('Count NaN:')

print(test.isnull().sum(), '\n')

print('Percentage NaN:')

print(test.isnull().sum()/ len(test))
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

train_len = train[train['target'] == 0]['text'].str.len()

ax1.hist(train_len,color='blue')

ax1.set_title('Not A Disaster')

train_len = train[train['target'] == 1]['text'].str.len()

ax2.hist(train_len,color='red')

ax2.set_title('Disaster')

fig.suptitle('Characters in Train Set\'s Text')

plt.show()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

train_len = train[train['target'] == 0]['text'].str.split().map(lambda x: len(x))

ax1.hist(train_len,color='blue')

ax1.set_title('Not A Disaster')

train_len = train[train['target'] == 1]['text'].str.split().map(lambda x: len(x))

ax2.hist(train_len,color='red')

ax2.set_title('Disaster')

fig.suptitle('Words in Train Set\'s Text')

plt.show()
ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

train.loc[train['id'].isin(ids_with_target_error),'target'] = 0
train.loc[train['keyword'].notnull(), 'text'] = train['keyword'] + ' ' + train['text']

test.loc[test['keyword'].notnull(), 'text'] = test['keyword'] + ' ' + test['text']



# view

train[train['keyword'].notnull()].head()
train = train.drop(['id', 'keyword', 'location'], axis=1)

test = test.drop(['keyword', 'location'], axis=1) # keep id



train.head()
import string

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



# NLTK Tweet Tokenizer for now

from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(strip_handles=True)



corpus = []



# clean up text

def clean_text(text):

    """

    Copied from other notebooks

    """

    # expand acronyms

    

    # special characters

    text = re.sub(r"\x89Û_", "", text)

    text = re.sub(r"\x89ÛÒ", "", text)

    text = re.sub(r"\x89ÛÓ", "", text)

    text = re.sub(r"\x89ÛÏWhen", "When", text)

    text = re.sub(r"\x89ÛÏ", "", text)

    text = re.sub(r"China\x89Ûªs", "China's", text)

    text = re.sub(r"let\x89Ûªs", "let's", text)

    text = re.sub(r"\x89Û÷", "", text)

    text = re.sub(r"\x89Ûª", "", text)

    text = re.sub(r"\x89Û\x9d", "", text)

    text = re.sub(r"å_", "", text)

    text = re.sub(r"\x89Û¢", "", text)

    text = re.sub(r"\x89Û¢åÊ", "", text)

    text = re.sub(r"fromåÊwounds", "from wounds", text)

    text = re.sub(r"åÊ", "", text)

    text = re.sub(r"åÈ", "", text)

    text = re.sub(r"JapÌ_n", "Japan", text)    

    text = re.sub(r"Ì©", "e", text)

    text = re.sub(r"å¨", "", text)

    text = re.sub(r"SuruÌ¤", "Suruc", text)

    text = re.sub(r"åÇ", "", text)

    text = re.sub(r"å£3million", "3 million", text)

    text = re.sub(r"åÀ", "", text)

    

    # emojis

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    

    

    """

    Our Stuff

    """

    # remove numbers

    text = re.sub(r'[0-9]', '', text)

    

    # remove punctuation and special chars (keep '!')

    for p in string.punctuation.replace('!', ''):

        text = text.replace(p, '')

        

    # remove urls

    text = re.sub(r'http\S+', '', text)

    

    # tokenize

    text = tknzr.tokenize(text)

    

    # remove stopwords

    text = [w.lower() for w in text if not w in stop_words]

    corpus.append(text)

    

    # join back

    text = ' '.join(text)

    

    return text
%%time

train['text'] = train['text'].apply(lambda s: clean_text(s))

test['text'] = test['text'].apply(lambda s: clean_text(s))



# see some cleaned data

train.sample(10)
texts = train['text'].to_numpy()

word_freq = {}



for text in texts:

    for word in text.split():

        word_freq[word] = word_freq.get(word, 0) + 1
# # remove words occuring < 3 times

# freq_threshold = 3

# for i, text in enumerate(texts):

#     for word in text.split():

#         if word_freq[word] < freq_threshold:

#             print(word)

#             texts[i].replace(word, '')
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



MAX_SEQUENCE_LENGTH = 40



tokenizer = Tokenizer()



tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)



word_index = tokenizer.word_index

num_words = len(word_index) + 1

print('Found %s unique tokens.' % (num_words - 1))



# pad 

data = pad_sequences(

    sequences, 

    maxlen=MAX_SEQUENCE_LENGTH,

    padding='post', 

    truncating='post'

)



labels = train['target'].to_numpy()

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)
x_train = data

y_train = labels
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from sklearn.metrics import roc_auc_score



vectorizer = CountVectorizer()

x_train_vectorized = vectorizer.fit_transform(train['text'])



# print vocabulary

print(vectorizer.get_feature_names()[2500:2600])
y_train_NB = np.array(train['target'])
# alpha is smoothing param

model_NB = BernoulliNB(alpha=1.0)

model_NB.fit(x_train_vectorized, y_train_NB)



# prepare test

x_test_NB = vectorizer.transform(test['text'])
# read fasttext twitter embeddings

embeddings_df = pd.read_pickle('/kaggle/input/fasttext-twitter-derived-embeddings/twitter_derived_embeddings')

embeddings_df.head()
EMBEDDING_DIM = 400



fasttext_embedding_idx = {}

for idx, row in embeddings_df.iterrows():

    word = row[0]

    embeddings = np.asarray(row[1], 'float32')

    fasttext_embedding_idx[word] = embeddings



# print only 20

fasttext_embedding_idx['earthquake'][:20]
# NOTE: comment out if using fasttext

# glove_embedding_idx = {}

# EMBEDDING_DIM = 100

# with open('/kaggle/input/glove-global-vectors-for-word-representation/glove.twitter.27B.100d.txt','r') as f:

#     for line in f:

#         values=line.split()

#         word=values[0]

#         vectors=np.asarray(values[1:],'float32')

#         glove_embedding_idx[word] = vectors

# f.close()
iv = 0

oov = 0



embedding_idx = fasttext_embedding_idx # swap between embeddings



embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embedding_idx.get(word)

    if embedding_vector is not None:

        iv += 1

        # words not found in the embedding space are all zeros

        embedding_matrix[i] = embedding_vector

    else: oov += 1

        

print('%i tokens in vocab, %i tokens out of vocab' % (iv, oov)) # TODO: must reduce out of vocab
# create the embedding layer, this will not be trainable!

from tensorflow.keras.layers import Embedding



model_NN = tf.keras.Sequential()



embedding_layer = Embedding(

    num_words, 

    EMBEDDING_DIM,

    weights = [embedding_matrix],

    input_length=MAX_SEQUENCE_LENGTH,

    trainable=False

)
model_NN = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'),

    embedding_layer,

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dropout(DROPOUT),

    tf.keras.layers.Dense(1, activation='sigmoid') # add sigmoid to get [0,1]

])



loss = tf.keras.losses.BinaryCrossentropy(

    from_logits=False, 

    name='binary_crossentropy'

)



optimizer = tf.keras.optimizers.Adam(LEARN_RATE)



model_NN.compile(

    loss=loss,

    optimizer=optimizer,

    metrics=['accuracy']

)
model_NN.fit(x_train, y_train, epochs=EPOCHS, validation_split=SPLIT, batch_size=BATCH_SIZE)
test_tokenizer = Tokenizer()

test_texts = test['text'].to_numpy()

test_tokenizer.fit_on_texts(texts)

test_sequences = test_tokenizer.texts_to_sequences(test_texts)



test_word_index = test_tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)



test_embedding_matrix = np.zeros((len(test_word_index) + 1, EMBEDDING_DIM))

for word, i in test_word_index.items():

    embedding_vector = embedding_idx.get(word)

    if embedding_vector is not None:

        # words not found in the embedding space are all zeros

        test_embedding_matrix[i] = embedding_vector
raw_lstm_preds = model_NN.predict(test_data)

lstm_preds = raw_lstm_preds.round().astype(int)
train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
import tensorflow_hub as hub
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

    

import tokenization
%%time 

max_seq_length = 180

bert_url = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/1"



input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32)

input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32)

segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32)

bert_inputs = [input_word_ids, input_mask, segment_ids]



bert_layer = hub.KerasLayer(bert_url, trainable=True)

pooled_output, sequence_output = bert_layer(bert_inputs)
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
# vocab_file is a url, do_lower_case is a tf.Variable bool, and tokenizer is an object

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
x_train_bert = bert_encode(train['text'].values, tokenizer, max_len=max_seq_length)

x_test_bert = bert_encode(test['text'].values, tokenizer, max_len=max_seq_length)

y_train_bert = train['target'].values
#dropout = tf.keras.layers.Dropout(DROPOUT)(sequence_output[:, 0, :])

dense = tf.keras.layers.Dense(64, activation='relu')(sequence_output[:, 0, :])

pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)



# callbacks, adaptive learning rate

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)



model_NN_bert = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)



optimizer = tf.keras.optimizers.Adam(LEARN_RATE)



model_NN_bert.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model_NN_bert.summary()
model_NN_bert.fit(

    x_train_bert,

    y_train_bert,

    validation_split=SPLIT,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE

)
bert_preds = model_NN_bert.predict(x_test_bert)
# probabilities [0,1]

# print(model_NB.classes_)

nb_preds_prob = model_NB.predict_proba(x_test_NB)[:, 1]
nb_preds = model_NB.predict(x_test_NB)

sample_submission['target'] = nb_preds

sample_submission.head()
np.allclose(nb_preds, nb_preds_prob.round().astype(int))
sample_submission.to_csv('submission_NB.csv', index=False)
sample_submission['target'] = lstm_preds

sample_submission.head()
sample_submission.to_csv('submission_NN.csv', index=False)
sample_submission['target'] = bert_preds.round().astype(int)

sample_submission.head()
sample_submission.to_csv('submission_NN_bert.csv', index=False)
nb_preds_prob[:5]
raw_lstm_preds = raw_lstm_preds[:, 0]

raw_lstm_preds[:5]
bert_preds = bert_preds[:, 0]

bert_preds[:5]
ensemble_preds = .5*nb_preds_prob + .5*bert_preds
ensemble_preds[:5]
sample_submission['target'] = ensemble_preds.round().astype(int)

sample_submission.head()
sample_submission.to_csv('submission_ensemble.csv', index=False)