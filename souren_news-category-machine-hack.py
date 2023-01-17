import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

from sklearn.model_selection import train_test_split

import time

import warnings

warnings.filterwarnings("ignore")

from nltk.corpus import stopwords

stop = stopwords.words('english')

from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import os

os.listdir('../input/')
train = pd.read_excel("../input/news-category-prediction/Data_Train.xlsx")

test = pd.read_excel("../input/news-category-prediction/Data_Test.xlsx")
crawl_vec = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
print("Train Data Shape {}\nTest Data Shape {}".format(train.shape, test.shape))
train.info()
train.head()
train.groupby(['SECTION']).size()
# Remove html tags from Text

def cleanhtml(raw_html):

  cleanr = re.compile('<.*?>')

  cleantext = re.sub(cleanr, '', raw_html)

  return cleantext
def decontracted(phrase):

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
def clean_data(train):

    text = cleanhtml(train)

    text = re.sub("’", "'",text)

    text = decontracted(text)

    text = re.sub(r"[,.;@#?!&$/\\-]+\ *", " ", text)

    text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', '', text)

    text = re.sub("[^a-zA-Z ]", "", text)

    text = [word.strip() for word in text.split() if word not in stop and len(word) > 1]

    text = " ".join(text)

    return text
train.STORY = train.STORY.apply(clean_data)

test.STORY = test.STORY.apply(clean_data)
train['len'] = [len(news) for news in train.STORY]

test['len'] = [len(news) for news in test.STORY]
print(train['len'].describe(), "\n\n", test['len'].describe())
max_len = 4742
from keras.preprocessing.text import Tokenizer, text_to_word_sequence



#Tokenize headline

tokenizer = Tokenizer()

tokenizer.fit_on_texts(train.STORY)

X = tokenizer.texts_to_sequences(train.STORY)

train['words'] = X

train.head()
word_index = tokenizer.word_index

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in X[0]])

print(decoded_review)
X = tokenizer.texts_to_sequences(test.STORY)

test['words'] = X
'''#Use GLOVE pretrained word-embeddings



EMBEDDING_DIMENSION = 300

embeddings_index = {}

f = open('crawl-300d_2M_vec/glove.6B.200d.txt', encoding="utf8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



EMBEDDING_FILE_PATH = "crawl-300d_2M_vec/crawl-300d_2M_vec.vec"

def get_coefs(word, *arr): 

    return word, np.asarray(arr, dtype='float32')



embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE_PATH, encoding='utf-8'))'''
'''#Create a weight matrix for words in training docs

word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index) + 1, 300))



for word, index in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector'''
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path,encoding = 'utf-8') as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words
embedding_matrix, unknown_words = build_matrix(word_index, crawl_vec)
print(unknown_words)
print("length of unknow words are ", len(unknown_words))
from keras.layers.embeddings import Embedding

from keras.initializers import Constant

EMBEDDING_DIMENSION = 300

#Create embedding layer from embedding matrix

embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIMENSION,

                            embeddings_initializer=Constant(embedding_matrix),

                            input_length=max_len, trainable=False)
from keras.utils import np_utils

from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split



#Prepare training and test data

X = np.array(list(sequence.pad_sequences(train.words, maxlen=max_len)))



category_dict = dict((i,k) for k,i in enumerate(list(train.groupby('SECTION').groups.keys())))

train['SECTION'] = train['SECTION'].apply(lambda x: category_dict[x])

Y = np_utils.to_categorical(list(train.SECTION))
from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Flatten



#RNN with LSTM

model = Sequential()



model.add(embedding_layer)

model.add(LSTM(30, dropout=0.25, recurrent_dropout=0.25))

#model.add(Dropout(0.25))

model.add(Dense(15, activation='softmax'))

#model.add(Dropout(0.25))

model.add(BatchNormalization())

model.add(Dense(len(category_dict), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
#model_history = model.fit(X, Y, batch_size=16, validation_split=0.20, epochs=6)
'''import matplotlib.pyplot as plt



acc = model_history.history['acc']

val_acc = model_history.history['val_acc']

loss = model_history.history['loss']

val_loss = model_history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.title('Training and validation accuracy')

plt.plot(epochs, acc, 'red', label='Training acc')

plt.plot(epochs, val_acc, 'blue', label='Validation acc')

plt.legend()



plt.figure()

plt.title('Training and validation loss')

plt.plot(epochs, loss, 'red', label='Training loss')

plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()



plt.show()'''
#test_X = np.array(list(sequence.pad_sequences(test.words, maxlen = max_len)))
#sub_data = model.predict_classes(test_X)
#pd.DataFrame(sub_data, columns=['SECTION']).to_excel("submisson.xlsx", index = False)