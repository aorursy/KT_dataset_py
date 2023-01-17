import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import re
import string

# Reset the output dimensions
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset

from scipy import linalg

from collections import Counter

import pickle

import nltk
nltk.download('wordnet')
from nltk import stem
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer

from gensim import matutils, models
from gensim.models import Word2Vec

import scipy.sparse

from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

plt.rcParams['figure.figsize'] = [24, 12]
plt.style.use('seaborn-darkgrid')
# Detect hardware, return appropriate distribution strategy

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

print("REPLICAS available: ", strategy.num_replicas_in_sync)
train = pd.read_csv('/kaggle/input/twitter-sentiment-analysis/train.csv')
test = pd.read_csv('/kaggle/input/twitter-sentiment-analysis/test.csv')
submission = pd.read_csv('/kaggle/input/twitter-sentiment-analysis/sample_submission.csv')
train.columns = train.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
test.columns = test.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

print('Train Data shape: ', train.shape, 'Test Data shape: ', test.shape)

train.head(10)
# word_count
train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split()))
test['word_count'] = test['tweet'].apply(lambda x: len(str(x).split()))

# unique_word_count
train['unique_word_count'] = train['tweet'].apply(lambda x: len(set(str(x).split())))
test['unique_word_count'] = test['tweet'].apply(lambda x: len(set(str(x).split())))

# stop_word_count
train['stop_word_count'] = train['tweet'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
test['stop_word_count'] = test['tweet'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

# url_count
train['url_count'] = train['tweet'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
test['url_count'] = test['tweet'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

# mean_word_length
train['mean_word_length'] = train['tweet'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test['mean_word_length'] = test['tweet'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# char_count
train['char_count'] = train['tweet'].apply(lambda x: len(str(x)))
test['char_count'] = test['tweet'].apply(lambda x: len(str(x)))

# punctuation_count
train['punctuation_count'] = train['tweet'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
test['punctuation_count'] = test['tweet'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# hashtag_count
train['hashtag_count'] = train['tweet'].apply(lambda x: len([c for c in str(x) if c == '#']))
test['hashtag_count'] = test['tweet'].apply(lambda x: len([c for c in str(x) if c == '#']))

# mention_count
train['mention_count'] = train['tweet'].apply(lambda x: len([c for c in str(x) if c == '@']))
test['mention_count'] = test['tweet'].apply(lambda x: len([c for c in str(x) if c == '@']))
METAFEATURES = ['word_count', 'unique_word_count', 'stop_word_count', 'url_count', 'mean_word_length',
        'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']

TWEETS = train['label'] == 1

fig, axes = plt.subplots(ncols = 2, nrows = len(METAFEATURES), figsize = (20, 50), dpi = 100)

for i, feature in enumerate(METAFEATURES):
    sns.distplot(train.loc[~TWEETS][feature], label = 'Positive', ax = axes[i][0], color = 'green')
    sns.distplot(train.loc[TWEETS][feature], label = 'Negative', ax = axes[i][0], color = 'red')

    sns.distplot(train[feature], label = 'Training', ax = axes[i][1])
    sns.distplot(test[feature], label = 'Test', ax = axes[i][1])
  
    for j in range(2):
        axes[i][j].set_xlabel('')
        axes[i][j].tick_params(axis = 'x', labelsize = 12)
        axes[i][j].tick_params(axis = 'y', labelsize = 12)
        axes[i][j].legend()
  
    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize = 13)
    axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize = 13)

plt.show()
fig, axes = plt.subplots(ncols = 2, figsize = (17, 4), dpi = 100)
plt.tight_layout()

train.groupby('label').count()['id'].plot(kind = 'pie', ax = axes[0], labels = ['Negative (92.9%)', 'Positive (7.1%)'])
sns.countplot(x = train['label'], hue = train['label'], ax = axes[1])

axes[0].set_ylabel('')
axes[1].set_ylabel('')
axes[1].set_xticklabels(['Negative (29720)', 'Positive (2242)'])
axes[0].tick_params(axis = 'x', labelsize = 15)
axes[0].tick_params(axis = 'y', labelsize = 15)
axes[1].tick_params(axis = 'x', labelsize = 15)
axes[1].tick_params(axis = 'y', labelsize = 15)

axes[0].set_title('Label Distribution in Training Set', fontsize = 13)
axes[1].set_title('Label Count in Training Set', fontsize = 13)

plt.show()
def remove_stopwords(string):
    word_list = [word.lower() for word in string.split()]
    stopwords_list = list(stopwords.words("english"))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    return ' '.join(word_list)
train['tweet_length'] = train['tweet'].apply(len)
train['tweet'] = train['tweet'].map(lambda x: re.sub('\\n',' ',str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r'\W',' ',str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r'https\s+|www.\s+',r'', str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r'http\s+|www.\s+',r'', str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r'\s+[a-zA-Z]\s+',' ',str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r'\^[a-zA-Z]\s+',' ',str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r'\s+',' ',str(x)))
train['tweet'] = train['tweet'].str.lower()

train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\’", "\'", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"won\'t", "will not", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"can\'t", "can not", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"don\'t", "do not", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"dont", "do not", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"n\’t", " not", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"n\'t", " not", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\'re", " are", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\'s", " is", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\’d", " would", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\d", " would", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\'ll", " will", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\'t", " not", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\'ve", " have", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\'m", " am", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\n", "", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\r", "", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"[0-9]", "digit", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\'", "", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r"\"", "", str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r'[?|!|\'|"|#]',r'', str(x)))
train['tweet'] = train['tweet'].map(lambda x: re.sub(r'[.|,|)|(|\|/]',r' ', str(x)))
train['tweet'] = train['tweet'].apply(lambda x: remove_stopwords(x))

test['tweet'] = test['tweet'].map(lambda x: re.sub('\\n',' ',str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r'\W',' ',str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r'\s+[a-zA-Z]\s+',' ',str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r'\^[a-zA-Z]\s+',' ',str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r'\s+',' ',str(x)))
test['tweet'] = test['tweet'].str.lower()

test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\’", "\'", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"won\'t", "will not", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"can\'t", "can not", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"don\'t", "do not", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"dont", "do not", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"n\’t", " not", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"n\'t", " not", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\'re", " are", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\'s", " is", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\’d", " would", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\d", " would", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\'ll", " will", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\'t", " not", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\'ve", " have", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\'m", " am", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\n", "", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\r", "", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"[0-9]", "digit", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\'", "", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r"\"", "", str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r'[?|!|\'|"|#]',r'', str(x)))
test['tweet'] = test['tweet'].map(lambda x: re.sub(r'[.|,|)|(|\|/]',r' ', str(x)))
test['tweet'] = test['tweet'].apply(lambda x: remove_stopwords(x))
def clean_text(text):
    '''make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    #text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('user ', '', text)
    text = re.sub('amp ', '', text)
    text = re.sub('like ', '', text)
    text = re.sub('new ', '', text)
    text = re.sub('people ', '', text)
    text = re.sub('bihday', 'birthday', text)
    text = re.sub('allahsoil', 'allah soil', text)
    return text
train['tweet'] = train['tweet'].apply(lambda x: clean_text(x))
test['tweet'] = test['tweet'].apply(lambda x: clean_text(x))
train_data = train.copy()

label = {0: 'A', 1: 'B'}
train['label'] = train['label'].map(label)
train = train.drop('id', axis = 1)

train = pd.get_dummies(train, columns = ['label'])
train.head()
categories = ['label_A', 'label_B']

train_dict = {}

for column in categories:
    a = train.loc[train[column] == 1, 'tweet'].tolist()
    train_dict[column] = ' '.join(a)
# We can either keep it in dictionary format or put it into a pandas dataframe

data_df = pd.DataFrame(train_dict.items())
data_df.columns = ['index', 'tweet']
data_df = data_df.set_index('index')
data_df = data_df.sort_index()
data_df.head()
data_df = pd.DataFrame(data_df['tweet'].apply(lambda x: clean_text(x)))
data_clean = data_df.copy()
data_df.head()
cv = CountVectorizer(stop_words = 'english')
data_cv = cv.fit_transform(data_df['tweet'])
data_dtm = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())
data_dtm.index = data_df.index
data_dtm = data_dtm.transpose()
data_dtm.head()
# Find the top 30 words on each category

top_dict = {}
for c in data_dtm.columns:
    top = data_dtm[c].sort_values(ascending = False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict
# Print the top 15 words said by each category

for category, top_words in top_dict.items():
    print(category + ":")
    print(', '.join([word for word, count in top_words[0:14]]))
    print('-----------------------------------------------------------------------------------------------------------------------')
# Let's first pull out the top words for each category

words = []
for category in data_dtm.columns:
    top = [word for (word, count) in top_dict[category]]
    for t in top:
        words.append(t)
        
words

# Let's aggregate this list and identify the most common words along with how many routines they occur in
Counter(words).most_common()
data_dtm
# Find the bottom 200 words in each category

bottom_dict = {}
for c in data_dtm.columns:
    bottom = data_dtm[c].sort_values(ascending = True).head(200)
    bottom_dict[c]= list(zip(bottom.index, bottom.values))

# Let's first pull out the bottom words for each category

bottom_words = []
for category in data_dtm.columns:
    bottom = [word for (word, count) in bottom_dict[category]]
    for b in bottom:
        bottom_words.append(b)

Counter(bottom_words).most_common()
# Let's make some word clouds!

stop_words = text.ENGLISH_STOP_WORDS

wc = WordCloud(stopwords = stop_words, background_color = "white", colormap = "Dark2", max_font_size = 150, random_state = 42)
# Create subplots for each category

for index, description in enumerate(data_dtm.columns):
    wc.generate(data_clean.tweet[description])
    
    plt.subplot(1, 2, index + 1)
    plt.imshow(wc, interpolation = "bilinear")
    plt.axis("off")
    plt.title(categories[index])
    
plt.show()
# Find the number of unique words that each category has

# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
unique_list = []
for category in data_dtm.columns:
    uniques = data_dtm[category].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(categories, unique_list)), columns = ['category', 'unique_words'])
data_unique_sort = data_words.sort_values(by = 'unique_words')
data_unique_sort
y_pos = np.arange(len(data_words))

plt.figure(figsize = (6, 8))
plt.barh(y_pos, data_unique_sort.unique_words, align = 'center')
plt.yticks(y_pos, data_unique_sort.category)
plt.title('Number of Unique Words', fontsize = 20)
plt.show()
Train, Test = train_test_split(train_data.drop('id', axis = 1), test_size = 0.25, random_state = 22) # Splits Dataset into Training and Testing set
print("Train Data size:", len(Train))
print("Test Data size", len(Test))
Train.head()
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(Train.tweet)

word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)
MAX_LENGTH = 30

from keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(tokenizer.texts_to_sequences(Train.tweet), maxlen = MAX_LENGTH)
X_test = pad_sequences(tokenizer.texts_to_sequences(Test.tweet),
                       maxlen = MAX_LENGTH)

print("Training X Shape:",X_train.shape)
print("Testing X Shape:",X_test.shape)
# Get all the train labels

labels = Train.tweet.unique().tolist()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(Train.label.to_list())

y_train = encoder.transform(Train.label.to_list())
y_test = encoder.transform(Test.label.to_list())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
GLOVE_EMB = '/kaggle/working/glove.6B.300d.txt'
BATCH_SIZE = 1024
EPOCHS = 15
MODEL_PATH = '/kaggle/working/best_model.hdf5'
embeddings_index = {}

f = open(GLOVE_EMB)
for line in f:
    values = line.split()
    word = value = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' %len(embeddings_index))
EMBEDDING_DIM = 300

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
import tensorflow as tf

embedding_layer = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, weights = [embedding_matrix], input_length = MAX_LENGTH, trainable = False)
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint
sequence_input = Input(shape = (MAX_LENGTH,), dtype = 'int32')
embedding_sequences = embedding_layer(sequence_input)
x = SpatialDropout1D(0.2)(embedding_sequences)
x = Conv1D(64, 5, activation = 'relu')(x)
x = Bidirectional(LSTM(64, dropout = 0.2, recurrent_dropout = 0.2))(x)
x = Dense(512, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation = 'relu')(x)
outputs = Dense(1, activation = 'sigmoid')(x)
model = tf.keras.Model(sequence_input, outputs)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

LR = 1e-3

model.compile(optimizer = Adam(learning_rate = LR), loss = 'binary_crossentropy', metrics = ['accuracy'])
ReduceLROnPlateau = ReduceLROnPlateau(factor = 0.1, min_lr = 0.01, monitor = 'val_loss', verbose = 1)
print("Training on GPU...") if tf.test.is_gpu_available() else print("Training on CPU...")
history = model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (X_test, y_test), callbacks = [ReduceLROnPlateau])
s, (at, al) = plt.subplots(2,1)
at.plot(history.history['accuracy'], c = 'b')
at.plot(history.history['val_accuracy'], c ='r')
at.set_title('model accuracy')
at.set_ylabel('accuracy')
at.set_xlabel('epoch')
at.legend(['LSTM_train', 'LSTM_val'], loc ='upper left')

al.plot(history.history['loss'], c ='m')
al.plot(history.history['val_loss'], c ='c')
al.set_title('model loss')
al.set_ylabel('loss')
al.set_xlabel('epoch')
al.legend(['train', 'val'], loc = 'upper left')
def decode_sentiment(score):

    return 1 if score > 0.5 else 0

scores = model.predict(X_test, verbose = 1, batch_size = 10000)
y_pred_1d = [decode_sentiment(score) for score in scores]
import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 13)
    plt.yticks(tick_marks, classes, fontsize = 13)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize = 17)
    plt.xlabel('Predicted label', fontsize = 17)
cnf_matrix = confusion_matrix(Test.label.to_list(), y_pred_1d)
plt.figure(figsize = (6,6))
plot_confusion_matrix(cnf_matrix, classes = Test.label.unique(), title = "Confusion matrix")
plt.show()
print(classification_report(list(Test.label), y_pred_1d))
results = pd.DataFrame()

Y_test = pad_sequences(tokenizer.texts_to_sequences(test.tweet), maxlen = MAX_LENGTH)


final_scores = model.predict(Y_test, verbose = 1, batch_size = 10000)
y_pred_1d = [decode_sentiment(score) for score in final_scores]

results['id'] = test['id'].tolist()
results['label'] = y_pred_1d
results.to_csv('tweets_v1.csv', index = False)
results.head()
import tensorflow as tf
import transformers
# Maximum sequence size for BERT is 512

def regular_encode(texts, tokenizer, maxlen = 512):
    enc_di = tokenizer.batch_encode_plus(texts, return_attention_masks = False, return_token_type_ids = False, pad_to_max_length = True, max_length = maxlen)
    return np.array(enc_di['input_ids'])
#bert large uncased pretrained tokenizer

tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased')
X_train, X_test, y_train, y_test = train_test_split(train_data['tweet'], train_data['label'], random_state = 22, test_size = 0.1)
#tokenizing the tweets' descriptions and converting the categories into one hot vectors using tf.keras.utils.to_categorical

Xtrain_encoded = regular_encode(X_train.astype('str'), tokenizer, maxlen = 128)
ytrain_encoded = tf.keras.utils.to_categorical(y_train, num_classes = 2, dtype = 'int32')
Xtest_encoded = regular_encode(X_test.astype('str'), tokenizer, maxlen = 128)
ytest_encoded = tf.keras.utils.to_categorical(y_test, num_classes = 2, dtype = 'int32')
def build_model(transformer, loss = 'categorical_crossentropy', max_len = 512):
    input_word_ids = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = "input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]

    #adding dropout layer
    x = tf.keras.layers.Dropout(0.40)(cls_token)

    #using a dense layer of 2 neurons as the number of unique categories is 2. 
    out = tf.keras.layers.Dense(2, activation = 'softmax')(x)

    model = tf.keras.Model(inputs = input_word_ids, outputs = out)
    model.compile(tf.keras.optimizers.Adam(lr = 3e-5), loss = loss, metrics = ['accuracy'])
    return model
#building the model on tpu

with strategy.scope():
    transformer_layer = transformers.TFAutoModel.from_pretrained('bert-large-uncased')
    model = build_model(transformer_layer, max_len = 128)
model.summary()
#creating the training and testing dataset.

BATCH_SIZE = 32*strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE 
train_dataset = (tf.data.Dataset.from_tensor_slices((Xtrain_encoded, ytrain_encoded)).repeat().shuffle(2048).batch(BATCH_SIZE).prefetch(AUTO))
test_dataset = (tf.data.Dataset.from_tensor_slices(Xtest_encoded).batch(BATCH_SIZE))
#training for 10 epochs

n_steps = Xtrain_encoded.shape[0] // BATCH_SIZE
train_history = model.fit(train_dataset, steps_per_epoch = n_steps, epochs = 20)
#making predictions 

preds = model.predict(test_dataset, verbose = 1)

#converting the one hot vector output to a linear numpy array.
pred_classes = np.argmax(preds, axis = 1)
Test = test['tweet']
TestEncoded = regular_encode(Test.astype('str'), tokenizer, maxlen = 128)
TestDataset = (tf.data.Dataset.from_tensor_slices(TestEncoded).batch(BATCH_SIZE))

#making predictions
Preds = model.predict(TestDataset, verbose = 1)

#converting the one hot vector output to a linear numpy array.
predClasses = np.argmax(Preds, axis = 1)
predClasses
results = pd.DataFrame()

results['id'] = test['id'].tolist()
results['label'] = predClasses
results.to_csv('tweets_BERT_v4.csv', index = False)
results.head()