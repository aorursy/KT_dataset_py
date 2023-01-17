# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# importing the libraries

import pandas as pd

import numpy as np

import re

from tqdm import tqdm

tqdm.pandas()



import xgboost as xgb

from tqdm import tqdm

from sklearn.svm import SVC

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss, accuracy_score, f1_score

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping

from nltk import word_tokenize

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
def url(x):

    x = re.sub(r'http\S+', '', x) #removing web-links 

    x = re.sub(r'@\w+','', x) #removing tags #removing tags

    return x

train["text"] = train["text"].progress_apply(lambda x: url(x))

test["text"] = test["text"].progress_apply(lambda x: url(x))
def deocde(text):

    for punct in ["\x89Û_","\x89ÛÒ","\x89ÛÓ","\x89ÛÏ","\x89Û÷","\x89Ûª","\x89Û\x9d","å_","\x89Û¢","\x89Û¢åÊ","åÊ","å¨","\x89û","åÈ"]:

        text=text.replace(punct, "")

    text=text.replace(r"åÊ"," ")

    text=text.replace(r"Ì_","a")

    text=text.replace(r"Ì©","e")

    text=text.replace(r"Ì¤","c")

    return text







train["text"] = train["text"].progress_apply(lambda x: deocde(x))

test["text"] = test["text"].progress_apply(lambda x: deocde(x))
train["text"] = train["text"].str.lower()

test["text"] = test["text"].str.lower()
train.head()
train.drop_duplicates(subset = ['text'],inplace=True)

test.drop_duplicates(subset = ['text'],inplace=True)

train.dropna(subset = ['text'],inplace=True)

test.dropna(subset = ['text'],inplace=True)
y = train.target.values

xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, 

                                                  stratify=y, 

                                                  random_state=97, 

                                                  test_size=0.1, shuffle=True)
print (xtrain.shape)

print (xvalid.shape)
tfidf = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')

tfidf.fit(list(xtrain)+ list(xvalid))

xtrain_tf = tfidf.transform(xtrain)

xvalid_tf = tfidf.transform(xvalid)
clf = LogisticRegression(C=1.0)

clf.fit(xtrain_tf, ytrain)

predictions = clf.predict(xvalid_tf)



print("f1_score: %0.3f" % f1_score(yvalid, predictions) )
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 2), stop_words = 'english')



# Fitting Count Vectorizer to both training and test sets (semi-supervised learning) Credit @abhishek thakur

ctv.fit(list(xtrain) + list(xvalid))

xtrain_ctv =  ctv.transform(xtrain) 

xvalid_ctv = ctv.transform(xvalid)
clf = LogisticRegression(C=1.0)

clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict(xvalid_ctv)



print("f1_score: %0.3f" % f1_score(yvalid, predictions) )
# Fitting a simple Naive Bayes on TFIDF

clf = MultinomialNB()

clf.fit(xtrain_tf, ytrain)

predictions = clf.predict(xvalid_tf)



print ("f1_score: %0.3f " % f1_score(yvalid, predictions))
# Fitting a simple Naive Bayes on Counts

clf = MultinomialNB()

clf.fit(xtrain_ctv, ytrain)

predictions = clf.predict(xvalid_ctv)



print ("f1_score: %0.3f " % f1_score(yvalid, predictions))

# print ("logloss: %0.3f " % log_loss(yvalid, predictions))
# Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model. credit @abhishekthakur

svd = decomposition.TruncatedSVD(n_components=120)

svd.fit(xtrain_tf)

xtrain_svd = svd.transform(xtrain_tf)

xvalid_svd = svd.transform(xvalid_tf)

 

# Scale the data obtained from SVD.

scl = preprocessing.StandardScaler()

scl.fit(xtrain_svd)

xtrain_svd_scl = scl.transform(xtrain_svd)

xvalid_svd_scl = scl.transform(xvalid_svd)
# Fitting a simple SVM

clf = SVC(C=1.0, probability=True) 

clf.fit(xtrain_svd_scl, ytrain)

predictions = clf.predict(xvalid_svd_scl)



print ("logloss: %0.3f " % f1_score(yvalid, predictions))

# print ("logloss: %0.3f " % log_loss(yvalid, predictions))
# Fitting a simple xgboost on tf-idf

clf = xgb.XGBClassifier(max_depth=5, n_estimators=100, colsample_bytree=0.6, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf.fit(xtrain_tf.tocsc(), ytrain)



predictions = clf.predict(xvalid_tf.tocsc())

print ("logloss: %0.3f " % f1_score(yvalid, predictions))
# Fitting a simple xgboost on CountVectors



clf = xgb.XGBClassifier(max_depth=5, n_estimators=100, colsample_bytree=0.6, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf.fit(xtrain_ctv.tocsc(), ytrain)

predictions = clf.predict(xvalid_ctv.tocsc())



print ("logloss: %0.3f " % f1_score(yvalid, predictions))
# Fitting a simple xgboost on tf-idf svd features

clf = xgb.XGBClassifier(max_depth=5, n_estimators=100, colsample_bytree=0.6, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

clf.fit(xtrain_svd, ytrain)

predictions = clf.predict(xvalid_svd)



print ("logloss: %0.3f " % f1_score(yvalid, predictions))
# Fitting a simple xgboost on tf-idf svd features

clf = xgb.XGBClassifier(nthread=10)

clf.fit(xtrain_svd, ytrain)

predictions = clf.predict(xvalid_svd)



print ("logloss: %0.3f " % f1_score(yvalid, predictions))
# Initialize SVD

svd = TruncatedSVD()

    

# Initialize the standard scaler 

scl = preprocessing.StandardScaler()



# We will use logistic regression here..

lr_model = LogisticRegression()



# Create the pipeline 

clf = pipeline.Pipeline([('svd', svd),

                         ('scl', scl),

                         ('lr', lr_model)])



param_grid = {'svd__n_components' : [120, 180],

              'lr__C': [0.1, 1.0, 10], 

              'lr__penalty': ['l1', 'l2']}
mll_scorer = metrics.make_scorer(f1_score, greater_is_better=False, needs_proba=False) #True @credit Abhishekthakur
# Initialize Grid Search Model

model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,

                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)



# Fit Grid Search Model

model.fit(xtrain_tf, ytrain)  # we can use the full data here but im only using xtrain

print("Best score: %0.3f" % model.best_score_)

print("Best parameters set:")

best_parameters = model.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))


from keras.preprocessing import sequence, text

from tensorflow.keras.preprocessing.sequence import pad_sequences

embed_size = 100 # how big is each word vector

max_features = 12000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 50 # max number of words in a question to use

batch_size = 256

train_epochs = 5

SEED = 97

tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(xtrain))

train_X = tokenizer.texts_to_sequences(xtrain)

valid_X = tokenizer.texts_to_sequences(xvalid)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

test_X = pad_sequences(valid_X, maxlen=maxlen)

word_index=tokenizer.word_index

print('Number of unique words:',len(word_index))
# load the GloVe vectors in a dictionary:



embeddings_index = {}

f = open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt')

for line in tqdm(f):

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()





print('Found %s word vectors.' % len(embeddings_index))



num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i < num_words:

        emb_vec=embeddings_index.get(word)

        if emb_vec is not None:

            embedding_matrix[i]=emb_vec     


from keras.initializers import Constant

from keras.optimizers import Adam

model=Sequential()



embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),

                   input_length=maxlen,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.3))

model.add(LSTM(120, dropout=0.5, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))





optimzer=Adam(learning_rate=3e-4)





from keras import backend as K

def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# compile the model

model.compile(optimizer=optimzer, loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
X_train,X_test,y_train,y_test=train_test_split(train_X,ytrain,test_size=0.2)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=4,epochs=5,validation_data=(X_test,y_test),verbose=2)