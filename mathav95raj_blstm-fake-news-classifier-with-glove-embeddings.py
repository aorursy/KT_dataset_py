import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import nltk

nltk.download('stopwords')

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer() 

from tensorflow.keras import regularizers, initializers, optimizers, callbacks

from tensorflow.keras.layers import *

from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
glove_path = "../input/glove6b100dtxt/glove.6B.100d.txt"

train_path = "../input/fake-news/train.csv"

test_path = "../input/fake-news/test.csv"
df_train = pd.read_csv(train_path)

df_test = pd.read_csv(test_path)

display(df_train)

display(df_test)

test_id = df_test['id']
print("missing values in train \n",df_train.isna().sum())

print("missing values in test \n",df_test.isna().sum())
def drop_missing(df):

  df.dropna(how = 'any', inplace=True)

  df.reset_index(drop = True, inplace=True)

  return df



def drop_cols(df):

    if('id' in df.columns):

      df.drop(['id', 'author', 'text'], axis = 1, inplace=True) 

    return df

df_train = drop_missing(df_train)

df_train = drop_cols(df_train)

df_test = drop_cols(df_test)

display(df_train.tail())

display(df_test.tail())
def basic_cleaning(df):

  df = df.apply(lambda x: x.astype(str).str.lower()) #apply to all rows, lambda is a temporary function

  df = df.apply(lambda x: x.astype(str).str.replace('\n', ""))#remove linebreaks

  return df

df_train = basic_cleaning(df_train)

df_test = basic_cleaning(df_test)
def extensive_cleaning(df):

  #cleaning done using regular expressions

  df = df.apply(lambda x: x.astype(str).str.replace(r'http[\w:/\.]+','URL'))

  #urls beginning wiht http followed by any word character set (\w) or (:) or (.) or (%20)

  df = df.apply(lambda x: x.astype(str).str.replace(r'(\S)+\.com((\S)+)?','URL')) 

  df = df.apply(lambda x: x.astype(str).str.replace(r'[^\.\w\s]','')) #remove everything but characters and punctuation

  df = df.apply(lambda x: x.astype(str).str.replace(r'\.','.')) #replace multiple periods with a single one

  df = df.apply(lambda x: x.astype(str).str.replace(r'\s\s+',' ')) #replace multiple white space with a single one

  return df

df_train = extensive_cleaning(df_train)

df_test = extensive_cleaning(df_test)
train_labels = df_train['label'].astype('int').to_numpy()

train_title = df_train['title']

test_title = df_test['title']

display(train_labels.shape)

display(train_title.shape)

display(test_title.shape)
def lemmatize(X):

  corpus = []

  for i in range(len(X)):

      news = X[i]

      news = news.split()

      news = [lemmatizer.lemmatize(word) for word in news if not word in stopwords.words('english')]

      news = ' '.join(news)

      corpus.append(news)

  return corpus
train_corpus = lemmatize(train_title)

test_corpus = lemmatize(test_title)
# Extract glove dictionary

word2glove = {}

glove_words = []

with open(glove_path, 'r') as f:

    for line in f:

        values = line.split()

        word = values[0]

        #values are in list of string format so convert to vector

        vector = np.asarray(values[1:], "float32")

        word2glove[word] = vector

        glove_words.append(word)

EMBEDDING_DIM = len(vector)
corpus = train_corpus

#PREPARE GLOVE DICTIONARY FOR ENCODING NEWS--->

all_text = ' '.join(corpus)

from collections import Counter

words = all_text.split()

corpus_u_words_counter = Counter(words).most_common() #tuple of unique words, frequency

corpus_u_words_frequent = [word[0] for word in corpus_u_words_counter if word[1]>5] #keep only words which have occured more than 5 times for deciding about words not in glove later

corpus_u_words = [word for word, count in corpus_u_words_counter]



glove_words_index = dict(zip(glove_words, range(len(glove_words))))



#filter corpus words --> words in glove / not in glove.. use a boolean array to reduce comparisons

#glove_word_index is a hash table --> quick search

is_corpus_word_in_glove = np.array([w in glove_words_index for w in corpus_u_words])

words_in_glove = [word for word, is_true in zip(corpus_u_words, is_corpus_word_in_glove)  if is_true]

words_not_in_glove = [word for word, is_true in zip(corpus_u_words, is_corpus_word_in_glove)  if not is_true]

freq_words_not_in_glove = [w for w in words_not_in_glove if w in corpus_u_words_frequent]



# create the dictionary of glove weights

word2index = dict(zip(words_in_glove, range(len(words_in_glove)))) #we have weights for this! --> fixed

freq_words_not_in_glove_index = dict(zip(freq_words_not_in_glove, range(len(word2index), len(word2index)+len(freq_words_not_in_glove)))) #we have to learn weights for this --> train embedding

word2index = dict(**word2index, **freq_words_not_in_glove_index) 

#creating dummy word for oov words

word2index['<Other>'] = len(word2index) #all others are accounted as other tag and one common embedding is learnt --> train embedding





print(len(corpus_u_words))

print(len(corpus_u_words_frequent))

print(len(words_in_glove))

print(len(freq_words_not_in_glove))

print(len(words_not_in_glove))

print(len(word2index))

print(100*len(words_in_glove)/len(corpus_u_words))
def encode_news(corpus):

  encoded_news = []

  for news in corpus:

    int_news = [word2index[word] if word in word2index else word2index['<Other>'] for word in news.split()] 

    encoded_news.append(int_news)

  return encoded_news



encoded_train_news = encode_news(train_corpus)

encoded_test_news = encode_news(test_corpus)
#histogram of title word length

plt.hist([len(t.split()) for t in corpus])

plt.show()

MAX_NEWS_LENGTH = 25



#creating a dummy word for padding

word2index['<PAD>'] = len(word2index)
def pad_news(encoded_news):

  for i, news in zip(range(len(encoded_news)), encoded_news):

    if (len(news) < MAX_NEWS_LENGTH):

      encoded_news[i] = [word2index['<PAD>']]*(MAX_NEWS_LENGTH - len(news)) + news

    elif (len(news) > MAX_NEWS_LENGTH):

      encoded_news[i] = news[:MAX_NEWS_LENGTH]

    else:

      continue

  return encoded_news
encoded_train_news = pad_news(encoded_train_news)

encoded_test_news = pad_news(encoded_test_news)
train_input = np.array(encoded_train_news)

train_label = train_labels.reshape(-1, 1)

test_input = np.array(encoded_test_news)



print(train_input.shape)

print(train_label.shape)
from keras.engine.topology import Layer

import keras.backend as K

from keras import initializers

import numpy as np



class Embedding2(Layer):



    def __init__(self, input_dim, output_dim, fixed_weights, embeddings_initializer='uniform', 

                 input_length=None, **kwargs):

        kwargs['dtype'] = 'int32'

        if 'input_shape' not in kwargs:

            if input_length:

                kwargs['input_shape'] = (input_length,)

            else:

                kwargs['input_shape'] = (None,)

        super(Embedding2, self).__init__(**kwargs)

    

        self.input_dim = input_dim

        self.output_dim = output_dim

        self.embeddings_initializer = embeddings_initializer

        self.fixed_weights = fixed_weights

        self.num_trainable = input_dim - len(fixed_weights)

        self.input_length = input_length

        

        w_mean = fixed_weights.mean(axis=0)

        w_std = fixed_weights.std(axis=0)

        self.variable_weights = w_mean + w_std*np.random.randn(self.num_trainable, output_dim)



    def build(self, input_shape, name='embeddings'):        

        fixed_weight = K.variable(self.fixed_weights, name=name+'_fixed')

        variable_weight = K.variable(self.variable_weights, name=name+'_var')

        

        self._trainable_weights.append(variable_weight)

        self._non_trainable_weights.append(fixed_weight)

        

        self.embeddings = K.concatenate([fixed_weight, variable_weight], axis=0)

        

        self.built = True



    def call(self, inputs):

        if K.dtype(inputs) != 'int32':

            inputs = K.cast(inputs, 'int32')

        out = K.gather(self.embeddings, inputs)

        return out



    def compute_output_shape(self, input_shape):

        if not self.input_length:

            input_length = input_shape[1]

        else:

            input_length = self.input_length

        return (input_shape[0], input_length, self.output_dim)
model = Sequential()

model.add(Embedding2(len(word2index) + 1, EMBEDDING_DIM,

                    fixed_weights=np.array([word2glove[w] for w in words_in_glove]))) 

model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(MAX_NEWS_LENGTH)))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

batch_size = 128

epochs = 5
x_train, x_val, y_train, y_val = train_test_split(train_input, train_label, test_size=0.2, random_state=42)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))
def plot_learningcurve(history):

  epoch_range = range(1,epochs+ 1)

  plt.plot(epoch_range, history.history['accuracy'])

  plt.plot(epoch_range, history.history['val_accuracy'])

  plt.title('Model accuracy')

  plt.ylabel('Accuracy')

  plt.xlabel('epoch')

  plt.legend(['train', 'val'], loc = 'upper right')

  plt.show()



  plt.plot(epoch_range, history.history['loss'])

  plt.plot(epoch_range, history.history['val_loss'])

  plt.title('Model loss')

  plt.ylabel('loss')

  plt.xlabel('epoch')

  plt.legend(['train', 'val'], loc = 'upper right')

  plt.show()
plot_learningcurve(history)
test_input.shape
predictions = model.predict_classes(test_input)

predictions
df_test['preds'] = predictions

df_test.iloc[[0]].to_numpy()

submission = pd.DataFrame({'id':test_id, 'label':predictions.flatten()})

submission.shape
submission.to_csv('submission.csv',index=False)