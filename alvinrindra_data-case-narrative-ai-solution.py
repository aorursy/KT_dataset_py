# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns

# Data Collection

df = pd.read_csv('../input/Consumer_Complaints.csv', nrows=50000)

print(df.shape)

df.head()
_ = sns.countplot(df['Timely response?'])
# Exploratory Data Analysis

_ = sns.countplot(y = df['Product'])
# Data Processing: Drop the fatures that are not used for Neural Networks Training

sel_df = df.iloc[:, [1,5, 15, 16]]

sel_df = sel_df.dropna()

print(sel_df.info())

print(sel_df.head())
sel_df.Product.value_counts()
# Category selection

categories = ['Debt collection', 'Mortgage', 'Credit reporting', 'Credit card', 'Student loan', 'Consumer Loan', 'Bank account or service', 'Money transfers']

sel_cat = sel_df.Product.isin(categories)

sel_df_cat = sel_df[sel_cat]

sel_df_cat['Product'].value_counts()

_ = sns.countplot(y = sel_df_cat['Product'])
_ = sns.countplot(sel_df_cat['Consumer disputed?'])
_ = sns.countplot(df['Timely response?'])
from io import StringIO

col = ['Product', 'Consumer complaint narrative']

sel_df_cat = sel_df_cat[col]



sel_df_cat['category_id'] = sel_df_cat['Product'].factorize()[0]

category_id_df = sel_df_cat[['Product', 'category_id']].drop_duplicates().sort_values('category_id')

category_to_id = dict(category_id_df.values)

id_to_category = dict(category_id_df[['category_id', 'Product']].values)

sel_df_cat.head()
# NLTK

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer



import re

import string



word_cloud = []

def clean_str(text):

    

    ## Remove puncuation

    text = text.translate(string.punctuation)

    

    ## Convert words to lower case and split them

    text = text.lower().split()

    

    ## Remove stop words

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops and len(w) >= 3]

    

    text = " ".join(text)



    # Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    

    #removing xxx since it will be treated as importand words by tf-idf vectorization

    text = re.sub(r"x{2,}", "", text)

    

    # fixing XXX and xxx like as word

    #text = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx",text)

    # removing non ascii

    text = re.sub(r'[^\x00-\x7F]+', "", text) 

    

    text = re.sub(r"\\", "", text)    

    text = re.sub(r"\'", "", text)    

    text = re.sub(r"\"", "", text)   

    

    text = re.sub(r'[^\w]', ' ', text)

    

    # Stemming is important to reduce the number of features (variation from a single word), why stemming?

    # Lemmatization takes way longer time to process

    text = text.split()

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in text]

    text = " ".join(stemmed_words)

    

    word_cloud.extend(text)



    return text
X = sel_df_cat['Consumer complaint narrative']

y = sel_df_cat['Product']

print(X.shape)

print(y.shape)
from time import time

t0 = time()

X = X.map(lambda x: clean_str(x))

print ("\nCleaning time: ", round(time()-t0, 1), "s")
sel_df_cat['Consumer complaint narrative'][2]
X[2]
import collections as co



str_words = "".join(word_cloud)

co_words = co.Counter(str_words)
from wordcloud import WordCloud

# Need to make hash 'dictionaries' from nltk for fast processing



# wee need word cloud to visualize the majority of vocabs, to see whether we have expected vocabs or not like XXXX, XXX words.

text = str_words

# Generate a word cloud image

wordcloud = WordCloud(width=800, height=400).generate(text)



# Display the generated image:

# the matplotlib way:

import matplotlib.pyplot as plt



# take relative word frequencies into account, lower max_font_size

wordcloud = WordCloud(background_color="white",max_words=len(word_cloud),max_font_size=40, relative_scaling=.8).generate(text)

plt.figure(figsize=(14, 7))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(X).toarray()

labels = sel_df_cat.category_id

features.shape

print(features)

print(category_to_id)
from sklearn.feature_selection import chi2



N = 2

for Product, category_id in sorted(category_to_id.items()):

  features_chi2 = chi2(features, labels == category_id)

  indices = np.argsort(features_chi2[0])

  feature_names = np.array(tfidf.get_feature_names())[indices]

  #print(feature_names)

  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

  print("# '{}':".format(Product))

  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))

  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split



model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, sel_df_cat.index, test_size=0.33, random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print(accuracy_score(y_test, y_pred))



conf_mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(conf_mat, annot=True, fmt='d',

            xticklabels=category_id_df.Product.values, yticklabels=category_id_df.Product.values)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print(classification_report(y_test, y_pred, target_names=sel_df_cat['Product'].unique()))
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB



X_train, X_test, y_train, y_test = train_test_split(sel_df_cat['Consumer complaint narrative'], sel_df_cat['Product'], random_state = 0)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
selection = sel_df_cat.iloc[:, [0,1]]

debt_collection = selection[selection['Product'] == "Debt collection"].head(10)

print(debt_collection)
print(clf.predict(count_vect.transform(debt_collection['Consumer complaint narrative'])))
mortgage = selection[selection['Product'] == "Mortgage"].head(10)

print(mortgage)
print(clf.predict(count_vect.transform(mortgage['Consumer complaint narrative'])))
# Deep Learning libs import



from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.models import Sequential, Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint



%matplotlib inline
df = pd.read_csv('../input/Consumer_Complaints.csv')

print(df.shape)

#df.head()
sel_df = df.iloc[:, [1,5, 15, 16]]

sel_df = sel_df.dropna()

# Category selection

categories = ['Debt collection', 'Mortgage', 'Credit reporting', 'Credit card', 'Student loan', 'Consumer Loan', 'Bank account or service', 'Money transfers']

sel_cat = sel_df.Product.isin(categories)

#sel_df.loc[sel_df.Product.isin(categories)]

sel_df_cat = sel_df[sel_cat]

sel_df_cat['Product'].value_counts()
X = sel_df_cat['Consumer complaint narrative']

y = sel_df_cat['Product']

print(X.shape)

print(y.shape)
from time import time

t0 = time()

X = X.map(lambda x: clean_str(x))

print ("\nCleaning time: ", round(time()-t0, 1), "s")
from tensorflow.contrib import learn



# Preprocessing to encode the text to sequences

max_doc_len = max([len(x.split(" ")) for x in X])

vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_len)

vocab_processor.fit_transform(X)

vocab_size=len(vocab_processor.vocabulary_)
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.15)



#Label Distribution between training set and test set

_ = sns.countplot(y = y_train)
_ = sns.countplot(y = y_test)
# Feature Engineering to encode the text to sequences and to encode the label to categorical sequences

def sentences_to_sequences(X):

    token = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ')

    token.fit_on_texts(X)

    X_seq = token.texts_to_sequences(X)

    X_seq = sequence.pad_sequences(X_seq, maxlen=max_doc_len)

    return X_seq



def label_encoding(y, num_cls):

    le = LabelEncoder()

    y_en = le.fit_transform(y)

    y_en = to_categorical(y_en, num_classes= num_cls)

    return y_en
X_seq = sentences_to_sequences(X)

y_en = label_encoding(y, len(y.value_counts()))
#X_train = sentences_to_sequences(X_train)

#X_test = sentences_to_sequences(X_test)

#y_train = label_encoding(y_train)

#y_test = label_encoding(y_test)
X_train,X_test,y_train,y_test = train_test_split(X_seq, y_en,test_size=0.15)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
print(max_doc_len)

print(vocab_size)
max_features = vocab_size

#max_features = 20000

maxlen = max_doc_len

embedding_dims = 50

filters = 128

kernel_size = 3

hidden_dims = 128



def ProductClassifier():

    model = Sequential()

    # Add embedding layer

    model.add(Embedding(max_features,

                        embedding_dims,

                        input_length=maxlen))

    model.add(Dropout(0.25))

    

    # Conv1D for filtering layer

    model.add(Conv1D(filters,

                     kernel_size,

                     padding='valid',

                     activation='relu',

                     strides=1))

    # max pooling:

    model.add(GlobalMaxPooling1D())



    # add a hidden layer:

    model.add(Dense(hidden_dims))

    model.add(Dropout(0.25))

    #model.add(Flatten())

    

    model.add(Activation('relu'))



    # Using Softmax for multiclass classifications

    # model.add(Dense(18))

    model.add(Dense(8))

    model.add(Activation('softmax'))

    return model



def BinaryClassifier():

    model = Sequential()

    # Add embedding layer

    model.add(Embedding(max_features,

                        embedding_dims,

                        input_length=maxlen))

    model.add(Dropout(0.25))

    

    # Conv1D for filtering layer

    model.add(Conv1D(filters,

                     kernel_size,

                     padding='valid',

                     activation='relu',

                     strides=1))

    # max pooling:

    model.add(GlobalMaxPooling1D())



    # add a hidden layer:

    model.add(Dense(hidden_dims))

    model.add(Dropout(0.25))

    #model.add(Flatten())

    

    model.add(Activation('relu'))



    # Using Softmax for multiclass classifications

    # model.add(Dense(18))

    model.add(Dense(2))

    model.add(Activation('sigmoid'))

    return model



def ProductClassifierWithLSTM():

    model = Sequential()

    model.add(Embedding(vocab_size, embedding_dims, input_length=maxlen))

    model.add(Dropout(0.2))

    model.add(Conv1D(64, 5, activation='relu'))

    model.add(MaxPooling1D(pool_size=4))

    model.add(LSTM(100))

    model.add(Dense(8, activation='softmax'))

    return model
dl_clf = ProductClassifier()

dl_clf.summary()

dl_clf.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from keras.utils import plot_model

from IPython.display import SVG

from IPython.display import Image



plot_model(dl_clf, to_file='model.png', show_shapes=True, show_layer_names=True)

Image(data="model.png")



#from keras.utils import model_to_dot

#SVG(model_to_dot(model).create(prog='dot', format='svg'))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)



t0 = time()

fit = dl_clf.fit(X_train, y_train,

          batch_size=128,

          verbose=1,

          shuffle=True,

          callbacks=[es, mc],

          epochs=10,

          validation_data=(X_test, y_test))

time_c1 = round(time()-t0, 1)

print ("\Training time: ", time_c1, "s")
print(fit.history.keys())

# summarize history for accuracy

plt.plot(fit.history['acc'])

plt.plot(fit.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(fit.history['loss'])

plt.plot(fit.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
index = 60

x_test = np.array([sel_df_cat.iloc[index, 1]])

y_result = np.array([sel_df_cat.iloc[index, 0]])

X_test_indices = sentences_to_sequences(x_test)

le = LabelEncoder()

le.fit_transform(sel_df_cat['Product'])

print('Narrative: ' + x_test[0] + ', Expected Product: ' + y_result[0] + ', Prediction Product: '+  le.inverse_transform([np.argmax(dl_clf.predict(X_test_indices))]))
x_test = np.array(['I have a problem with my loan and I need the solution fast'])

X_test_indices = sentences_to_sequences(x_test)

le = LabelEncoder()

le.fit_transform(sel_df_cat['Product'])

print('Narrative: ' + x_test[0] + ', Prediction Product: '+  le.inverse_transform([np.argmax(dl_clf.predict(X_test_indices))]))
x_test = debt_collection['Consumer complaint narrative']

X_test_indices = sentences_to_sequences(x_test)

le = LabelEncoder()

le.fit_transform(sel_df_cat['Product'])

predicts = dl_clf.predict(X_test_indices)

for predict in predicts:

    print(le.inverse_transform([np.argmax(predict)]))
dl_clf = ProductClassifierWithLSTM()

dl_clf.summary()

dl_clf.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from keras.utils import plot_model

from IPython.display import SVG

from IPython.display import Image



plot_model(dl_clf, to_file='model_lstm.png', show_shapes=True, show_layer_names=True)

Image(data="model_lstm.png")
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)



t0 = time()

fit = dl_clf.fit(X_train, y_train,

          batch_size=128,

          verbose=1,

          shuffle=True,

          callbacks=[es, mc],

          epochs=1,

          validation_data=(X_test, y_test))

time_c2 = round(time()-t0, 1)

print ("\Training time: ", time_c2, "s")
print(fit.history.keys())

# summarize history for accuracy

plt.plot(fit.history['acc'])

plt.plot(fit.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(fit.history['loss'])

plt.plot(fit.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
x_test = debt_collection['Consumer complaint narrative']

X_test_indices = sentences_to_sequences(x_test)

le = LabelEncoder()

le.fit_transform(sel_df_cat['Product'])

predicts = dl_clf.predict(X_test_indices)

for predict in predicts:

    print(le.inverse_transform([np.argmax(predict)]))
sel_df_cat.shape
sel_df_concat = sel_df_cat

#sel_df_cat_1 = sel_df_cat[:120000]

#sel_df_cat_2 = sel_df_cat[120000:]

#sel_df_cat_1 = sel_df_cat_1[sel_df_cat_1['Consumer disputed?'] != 'No']

#sel_df_concat = pd.concat([sel_df_cat_1,sel_df_cat_2])
sel_df_concat.head()
X = sel_df_concat['Consumer complaint narrative']

y = sel_df_concat['Consumer disputed?']

print(X.shape)

print(y.shape)
_ = sns.countplot(y)
y.value_counts()
X_seq = sentences_to_sequences(X)

y_en = label_encoding(y, len(y.value_counts()))
X_train,X_test,y_train,y_test = train_test_split(X_seq, y_en,test_size=0.05)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
dl_clf_disputed = BinaryClassifier()

dl_clf_disputed.summary()

dl_clf_disputed.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)



t0 = time()

fit = dl_clf_disputed.fit(X_train, y_train,

          batch_size=128,

          verbose=1,

          shuffle=True,

          callbacks=[es, mc],

          epochs=10,

          validation_data=(X_test, y_test))

time_c3 = round(time()-t0, 1)

print ("Training time: ", time_c3, "s")
print(fit.history.keys())

# summarize history for accuracy

plt.plot(fit.history['acc'])

plt.plot(fit.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(fit.history['loss'])

plt.plot(fit.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
X = sel_df_concat['Consumer complaint narrative']

y = sel_df_concat['Timely response?']

print(X.shape)

print(y.shape)
_ = sns.countplot(y)
X_seq = sentences_to_sequences(X)

y_en = label_encoding(y, len(y.value_counts()))
X_train,X_test,y_train,y_test = train_test_split(X_seq, y_en,test_size=0.05)
dl_clf_time = BinaryClassifier()

dl_clf_time.summary()

dl_clf_time.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)



t0 = time()

fit = dl_clf_time.fit(X_train, y_train,

          batch_size=128,

          verbose=1,

          shuffle=True,

          callbacks=[es, mc],

          epochs=10,

          validation_data=(X_test, y_test))

time_c3 = round(time()-t0, 1)

print ("Training time: ", time_c3, "s")
print(fit.history.keys())

# summarize history for accuracy

plt.plot(fit.history['acc'])

plt.plot(fit.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(fit.history['loss'])

plt.plot(fit.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
index = 60

x_test = np.array([sel_df_cat.iloc[index, 1]])

y_product = np.array([sel_df_cat.iloc[index, 0]])

y_time = np.array([sel_df_cat.iloc[index, 2]])

y_disputed = np.array([sel_df_cat.iloc[index, 3]])

X_test_indices = sentences_to_sequences(x_test)

le_disputed = LabelEncoder()

le_disputed.fit_transform(sel_df_concat['Consumer disputed?'])

le_product = LabelEncoder()

le_product.fit_transform(sel_df_concat['Product'])

le_time = LabelEncoder()

le_time.fit_transform(sel_df_concat['Timely response?'])



print('Narrative: ' + x_test[0])

print('Expected Product: ' + y_product[0] + ', Prediction Product: '+  

      le_product.inverse_transform([np.argmax(dl_clf.predict(X_test_indices))]))

print('Expected Timely response: ' + y_time[0] + ', Prediction Timely response: '+  

      le_time.inverse_transform([np.argmax(dl_clf_time.predict(X_test_indices))]))

print('Expected Disputed: ' + y_disputed[0] + ', Prediction Disputed: '+  

      le_disputed.inverse_transform([np.argmax(dl_clf_disputed.predict(X_test_indices))]))
time_yes = sel_df_cat[sel_df_cat['Timely response?'] == "Yes"].head(10)

time_yes
x_test = time_yes['Consumer complaint narrative']

X_test_indices = sentences_to_sequences(x_test)

le = LabelEncoder()

le.fit_transform(sel_df_cat['Timely response?'])

predicts = dl_clf_time.predict(X_test_indices)

for predict in predicts:

    print(le.inverse_transform([np.argmax(predict)]))
time_dict = {'model': ['CNN', 'LSTM'], 'time': [time_c1 / 60, time_c2 / 60]}

time_df = pd.DataFrame.from_dict(time_dict)

time_df
index = np.arange(len(time_df.model))

_ = sns.barplot(x="model", y="time", data=time_df).set_title('Training time per minute')