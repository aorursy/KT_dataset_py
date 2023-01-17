# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential, Model # initialize neural network library
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate # build our layers library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing import text, sequence
from sklearn.metrics import f1_score
# load train and test datasets
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train datasets shape:", train_df.shape)
print("Test datasets shape:", test_df.shape)
## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values
def first_nn_model():    
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model
#model = first_nn_model()
from sklearn.cluster import KMeans
def kmeans_model():
    model = KMeans()
    return model
#model = kmeans_model()
from sklearn.naive_bayes import GaussianNB
def nbgauss_model():
    model = GaussianNB()
    return model
#model = nbgauss_model()
from sklearn.naive_bayes import MultinomialNB
def nbmulti_model():
    model = MultinomialNB()
    return model
#model = nbmulti_model()
def second_nn_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, 100)(inp)
    x = CuDNNGRU(64, return_sequences=True)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

#model = gsecond_nn_model()
# Evaluating the ANN 
def simple_nn_model():
    model = Sequential() # initialize neural network
    model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = val_X.shape[1]))
    model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
model = simple_nn_model()
from sklearn.svm import LinearSVC
def svc_model():
    model = LinearSVC()
    return model
#model = svc_model()
from sklearn.linear_model import LinearRegression
def linr_model():
    model = LinearRegression()
    return model
#model = lr_model()
from sklearn.linear_model import LogisticRegression
def logr_model():
    model = LogisticRegression()
    return model
#model = logr_model()
# Tried to run it but it doesn't work (need some preparation with dataset)
import lightgbm as lgb
def lgb_model():
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_eval = lgb.Dataset(test_X, val_y, reference=lgb_train)
    # specify your configurations as a dict
    params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
    }
    
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=20,
                      valid_sets=lgb_eval,
                      early_stopping_rounds=5)
    # predict
    pred_noemb_val_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_val = (pred_noemb_val_y > 0.5).astype(int)
    # eval
    print('The f1score of prediction is:', f1score(val_y, y_pred_val) ** 0.5)
#lgb_model()    
from xgboost import XGBClassifier
def xgb_model():
    model = XGBClassifier()
    return model
#model = xgb_model()
#Create model, train and predict

model.fit(train_X, train_y)
pred_noemb_val_y = model.predict(val_X)
pred_noemb_test_y = model.predict(test_X)

pred_noemb_test_y.shape
train_meta = np.zeros(train_y.shape)
test_meta = np.zeros(test_X.shape[0])
y_pred_val = (pred_noemb_val_y > 0.5).astype(int)
y_pred_test = (pred_noemb_test_y > 0.5).astype(int)
# Print predict using f1score from sklearn
score = f1_score(val_y, y_pred_val)
print(score)
# Collect garbage
import gc; gc.collect()
time.sleep(10)
# Create a submission
sub_df = pd.DataFrame({'qid':test_df.qid.values})
sub_df['prediction'] = y_pred_test
sub_df.to_csv('submission.csv', index=False)