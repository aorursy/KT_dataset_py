import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer



from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

import nltk



from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer

from sklearn.svm import SVC, LinearSVC

from sklearn import tree

from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA



from matplotlib import pyplot as plt



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix



from keras.models import Sequential



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/stocknews/Combined_News_DJIA.csv')

data.head()
data['headlines'] = data[data.columns[2:]].apply(lambda x: '. '.join(x.dropna().astype(str)),axis=1)
data['comment_length'] = data['headlines'].apply(lambda x : len(x))

data['comment_length'].hist()
data['Label'].hist()
train = data[data['Date'] < '2015-01-01']

test = data[data['Date'] > '2014-12-31']
nltk.download('stopwords', quiet=True, raise_on_error=True)

stop_words_en = set(nltk.corpus.stopwords.words('english'))

stop_words_en.add("b")



class CustomTokenizer:

    

    def __init__(self):

        self.wnl = WordNetLemmatizer()

        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

        

    def _lem(self, token):

        if (token in stop_words_en):

            return token  # Solves error "UserWarning: Your stop_words may be inconsistent with your preprocessing."

        return self.wnl.lemmatize(token)

    

    def __call__(self, doc):

        return [self._lem(t) for t in self.tokenizer.tokenize(doc)]
vectorizer = CountVectorizer(tokenizer=CustomTokenizer(), stop_words=stop_words_en, lowercase=True, min_df=0.0075,  max_df=0.05, ngram_range=(2,2))



features_train = vectorizer.fit_transform(train['headlines'].tolist())

features_test = vectorizer.transform(test['headlines'].tolist())
feature_names = vectorizer.get_feature_names()

print(feature_names[50:100])



X_train = pd.DataFrame(features_train.todense(), columns = feature_names)

X_test = pd.DataFrame(features_test.todense(), columns = feature_names)



X_train.head()
from collections import defaultdict



up_unigrams = defaultdict(int)

down_unigrams = defaultdict(int)



for word in feature_names:

    up_unigrams[word] += np.sum(X_train[train['Label']==1][word])

    down_unigrams[word] += np.sum(X_train[train['Label']==0][word])

        

df_up_unigrams = pd.DataFrame(sorted(up_unigrams.items(), key=lambda x: x[1])[::-1])

df_down_unigrams = pd.DataFrame(sorted(down_unigrams.items(), key=lambda x: x[1])[::-1])

df_up_unigrams.head()
import seaborn as sns



N=25



fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)

plt.tight_layout()



sns.barplot(y=df_up_unigrams[0].values[:N], x=df_up_unigrams[1].values[:N], ax=axes[0], color='green')

sns.barplot(y=df_down_unigrams[0].values[:N], x=df_down_unigrams[1].values[:N], ax=axes[1], color='red')



for i in range(2):

    axes[i].spines['right'].set_visible(False)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    axes[i].tick_params(axis='x', labelsize=13)

    axes[i].tick_params(axis='y', labelsize=13)



axes[0].set_title(f'Top {N} most common unigrams in headlines resulting in stock up', fontsize=15)

axes[1].set_title(f'Top {N} most common unigrams in headlines resulting in stock down', fontsize=15)



plt.show()
clf = SVC()

clf = clf.fit(X_train, train["Label"].tolist())

print('Accuracy X_train: ' + str(clf.score(X_train, train["Label"].tolist())))



predictions = clf.predict(X_test)



pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
print (classification_report(test["Label"], predictions))

print ('Accuracy X_test: ' + str(accuracy_score(test["Label"], predictions)))
from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import Adadelta,Adam,RMSprop

from keras.utils import np_utils

from keras import Sequential, optimizers, regularizers



scale = np.max(X_train)

X_train /= scale

X_test /= scale



mean = np.mean(X_train)

X_train -= mean

X_test -= mean



num_features = X_train.shape[1]



model = Sequential()

model.add(Dense(8,input_shape=(num_features,), activation='relu', kernel_regularizer = regularizers.l2(0.1)))

model.add(Dropout(0.5))

#model.add(Dense(32, activation='relu', kernel_regularizer = regularizers.l2(0.001)))

#model.add(Dropout(0.5))

#model.add(Dense(32, activation='relu', kernel_regularizer = regularizers.l2(0.0001)))

#model.add(Dropout(0.5))

#model.add(Dense(32, activation='relu', kernel_regularizer = regularizers.l2(0.1)))

#model.add(Dropout(0.5))

#model.add(Dense(128, activation='relu', kernel_regularizer = regularizers.l2(0.001)))

#model.add(Dropout(0.5))

model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())



Y_train = np_utils.to_categorical(train["Label"], 2)

Y_test = np_utils.to_categorical(test["Label"], 2)



history = model.fit(X_train, Y_train, batch_size=32, validation_data=(X_test, Y_test), epochs=100, verbose=0)



# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

    

score = model.evaluate(X_test, Y_test)

print(score)
class MyTokenizer():

    def __init__(self):

        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

        self.lemmatizer = nltk.stem.WordNetLemmatizer()

        self.stop_words_en = set(nltk.corpus.stopwords.words('english'))

        self.stop_words_germ = set(nltk.corpus.stopwords.words('german'))

        self.stop_words = set()

        self.stop_words.add("b")

        

    def tokenize(self, headlines):

        # Tokenize

        tokens = [self.tokenizer.tokenize(article) for article in headlines]



        # Lemmatizer

        clean_tokens = []

        for words in tokens:

            clean_tokens.append([self.lemmatizer.lemmatize(word) for word in words])



        # Stop words

        final_tokens = []

        for words in clean_tokens:

            final_tokens.append([word.lower() for word in words if word.lower() not in self.stop_words_en and word.lower() not in self.stop_words_germ and word.lower() not in self.stop_words])

            

        return final_tokens
from gensim.models import Word2Vec



tokenizer = MyTokenizer()



headlines_train = train["headlines"]

headlines_test= test["headlines"]



tokens_train = tokenizer.tokenize(headlines_train)

tokens_test = tokenizer.tokenize(headlines_test)



model = Word2Vec(tokens_train, min_count=1,size= 50,workers=3, window =3, sg = 1)



word_vectors = model.wv

print("Number of word vectors: {}".format(len(word_vectors.vocab)))



print(model.wv.most_similar('husband'))
from tensorflow.python.keras.preprocessing.text import Tokenizer

from keras.layers import Flatten, Dense, LSTM, GRU, SpatialDropout1D, Bidirectional, concatenate

from keras.layers.embeddings import Embedding

from keras.preprocessing.sequence import pad_sequences

from keras.optimizers import SGD



my_tokenizer = MyTokenizer()



headlines_train = train["headlines"]

headlines_test= test["headlines"]

tokens_train = my_tokenizer.tokenize(headlines_train)

tokens_test = my_tokenizer.tokenize(headlines_test)



tokenizer = Tokenizer(num_words=20000)

tokenizer.fit_on_texts(train["headlines"])

        

vocab_size = 20000 #len(tokenizer.word_index) + 1

print('Number of words: ' + str(vocab_size))



X_train_tokens = tokenizer.texts_to_sequences(tokens_train)

X_test_tokens = tokenizer.texts_to_sequences(tokens_test)



max_length = 0

for words in X_train_tokens:

    if len(words)>max_length:

        max_length = len(words)

max_length = 200

print('max_length: ' + str(max_length))



X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')

X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')



y_train = train['Label']

y_test = test['Label']

Y_train = np_utils.to_categorical(y_train, 2)

Y_test = np_utils.to_categorical(y_test, 2)



model = Sequential()

model.add(Embedding(vocab_size, 16, input_length=max_length))

#model.add(SpatialDropout1D(0.5))

model.add(LSTM(8, dropout=0.2, recurrent_dropout=0.2)) 

#model.add(Bidirectional(LSTM(units=64, recurrent_dropout=0.5)))

#model.add(Flatten())

#model.add(Dense(32, activation='relu', kernel_regularizer = regularizers.l2(0.001)))

model.add(Dropout(0.5))

model.add(Dense(16, activation='relu', kernel_regularizer = regularizers.l2(0.01)))

#model.add(Dropout(0.5))

#model.add(Dense(64, activation='relu', kernel_regularizer = regularizers.l2(0.001)))

#model.add(Dropout(0.5))

#model.add(GRU(units=32, dropout=0.5, recurrent_dropout=0.5))

#model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



print(model.summary())



history = model.fit(X_train_pad, Y_train, batch_size=32, epochs=50, verbose=0, validation_data=(X_test_pad, Y_test))



# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



loss, accuracy = model.evaluate(X_test_pad,Y_test)

print('Testing Accuracy is {} '.format(accuracy*100))