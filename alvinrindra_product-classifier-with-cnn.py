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

df = pd.read_csv('../input/Consumer_Complaints.csv', nrows = 50000)

print(df.shape)

df.head()
_ = sns.countplot(y = df['Product'])
df['Product'].value_counts()
# Data Processing: Drop the fatures that are not used for Neural Networks Training

sel_df = df.iloc[:, [1,5]]

sel_df = sel_df.dropna()

print(sel_df.info())

print(sel_df.head())

_ = sns.countplot(y = sel_df['Product'])
sel_df.Product.value_counts()
# Category selection

# categories = ['Debt collection', 'Mortgage', 'Credit reporting', 'Credit card', 'Student loan', 'Consumer Loan']

exclude = [

    'Credit reporting, credit repair services, or other personal consumer reports', 

    'Credit card or prepaid card', 

    'Money transfer, virtual currency, or money service ',

    'Payday loan, title loan, or personal loan'

]

sel_cat = sel_df.Product.isin(exclude)

sel_df_cat = sel_df[~sel_cat] # select category not in the exclude categories 

sel_df_cat['Product'].value_counts()
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



# Text Normalization

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

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    

        

    text = re.sub(r"\$", " $ ", text) #isolate $

    text = re.sub(r"\%", " % ", text) #isolate %

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    #text = re.sub(r" e g ", " eg ", text)

    #text = re.sub(r" b g ", " bg ", text)

    #text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    #text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    #text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    

    #removing xxx since it will be treated as importand words by tf-idf vectorization

    text = re.sub(r"x{2,}", " ", text)

    

    # fixing XXX and xxx like as word

    #text = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx",text)

    # removing non ascii

    text = re.sub(r'[^\x00-\x7F]+', "", text) 

    

    # Stemming is important to reduce the number of features (variation from a single word), why stemming?

    # Lemmatization takes way longer time to process

    text = text.split()

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in text]

    text = " ".join(stemmed_words)



    return text
X = sel_df_cat['Consumer complaint narrative']

y = sel_df_cat['Product']

print(X.shape)

print(y.shape)
from time import time

t0 = time()

X = X.map(lambda x: clean_str(x))

print ("\nCleaning time: ", round(time()-t0, 1), "s")
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

  print(feature_names)

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
sel_df_cat.head(10)
debt_collection = sel_df[sel_df['Product'] == "Debt collection"].head(10)

print(debt_collection)
print(clf.predict(count_vect.transform(debt_collection['Consumer complaint narrative'])))
student_loan = sel_df[sel_df['Product'] == "Student loan"].head(10)

print(student_loan)
print(clf.predict(count_vect.transform(student_loan['Consumer complaint narrative'])))
# Deep Learning libs import



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.models import Sequential, Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Conv1D, GlobalMaxPooling1D, MaxPooling1D

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping



%matplotlib inline
df = pd.read_csv('../input/Consumer_Complaints.csv', nrows=500000)

print(df.shape)

df.head()
sel_df = df.iloc[:, [1,5]]

sel_df = sel_df.dropna()

exclude = [

    'Credit reporting, credit repair services, or other personal consumer reports', 

    'Credit card or prepaid card', 

    'Money transfer, virtual currency, or money service ',

    'Payday loan, title loan, or personal loan'

]

sel_cat = sel_df.Product.isin(exclude)

sel_df_cat = sel_df[~sel_cat] # select category not in the exclude categories 

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
print(max_doc_len)

print(vocab_size)
token = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ')

token.fit_on_texts(X)

X_train_seq = token.texts_to_sequences(X)

X_train_seq = sequence.pad_sequences(X_train_seq, maxlen=max_doc_len)





print(X_train_seq.shape)
le = LabelEncoder()

y_en = le.fit_transform(y)

print(np.unique(y_en, return_counts=True))



y_en = to_categorical(y_en, num_classes= 15)

print(y_en)

print(y_en.shape)
X_train,X_test,y_train,y_test = train_test_split(X_train_seq, y_en,test_size=0.15)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
max_features = vocab_size

maxlen = max_doc_len

embedding_dims = 50

filters = 250

kernel_size = 5

hidden_dims = 250



def ProductClassifier():

    model = Sequential()

    # Add embedding layer

    model.add(Embedding(max_features,

                        embedding_dims,

                        input_length=maxlen))

    model.add(Dropout(0.2))

    

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

    model.add(Dropout(0.2))

    model.add(Activation('relu'))



    # Using Softmax for multiclass classifications

    # model.add(Dense(18))

    model.add(Dense(15))

    model.add(Activation('softmax'))

    return model
model = ProductClassifier()

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
fit = model.fit(X_train, y_train,

          batch_size=50,

          epochs=5,

          #shuffle=True,

          validation_data=(X_test, y_test))
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
def sentences_to_sequences(X):

    token = Tokenizer(num_words=vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ')

    token.fit_on_texts(X)

    X_seq = token.texts_to_sequences(X)

    X_seq = sequence.pad_sequences(X_seq, maxlen=max_doc_len)

    return X_seq
index = 228

x_test = np.array([sel_df_cat.iloc[index, 1]])

x_result = np.array([sel_df_cat.iloc[index, 0]])

X_test_indices = sentences_to_sequences(x_test)

le = LabelEncoder()

le.fit_transform(sel_df_cat['Product'])

print('Narrative: ' + x_test[0] + ', Expected Product: ' + x_result[0] + ', Prediction Product: '+  le.inverse_transform([np.argmax(model.predict(X_test_indices))]))
x_test = np.array(['I have a problem with my credit. This is really sad.'])

X_test_indices = sentences_to_sequences(x_test)

le = LabelEncoder()

le.fit_transform(sel_df_cat['Product'])

print('Narrative: ' + x_test[0] + ', Prediction Product: '+  le.inverse_transform([np.argmax(model.predict(X_test_indices))]))