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
train=pd.read_csv('../input/train.tsv',sep='\t')
test=pd.read_csv('../input/test.tsv',sep='\t')
submission=pd.read_csv('../input/sampleSubmission.csv')
train
ytrain=train.copy()
ytrain=ytrain.drop(columns=['PhraseId','SentenceId','Phrase'])
ytrain=np.ravel(ytrain)
from sklearn.feature_extraction.text import TfidfVectorizer

tfid_vector=TfidfVectorizer(analyzer='word')
tfid_vector.fit(train['Phrase'])

xtrain=tfid_vector.transform(train['Phrase'])
xtest=tfid_vector.transform(test['Phrase'])
'''
import xgboost as xgb

model_xgb=xgb.XGBClassifier(eta=0.2)
model_xgb.fit(xtrain,ytrain)

ypred_xgb=model_xgb.predict(xtest)
'''
'''
import lightgbm as lgb

d_train = lgb.Dataset(xtrain, label=ytrain)

params = {}
params['learning_rate'] = 0.002
params['boosting_type'] = 'gbdt'
params['objective'] = 'multiclass'
params['metric'] = 'multi_logloss'
params['num_class'] = 5

model_lgb = lgb.train(params, d_train, 100)

ypred_lgb=model_lgb.predict(xtest)
'''
'''
pred_lgb=[]

for x in ypred_lgb:
    pred_lgb.append(np.argmax(x))
'''
'''
from keras.preprocessing.text import Tokenizer

token=Tokenizer(num_words=20000)
token.fit_on_texts(train['Phrase'])
xtrain=token.texts_to_matrix(train['Phrase'])
xtest=token.texts_to_matrix(test['Phrase'])
'''
from keras.utils.np_utils import to_categorical
ytrain=to_categorical(ytrain)
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM

model_DL=models.Sequential()
model_DL.add(layers.Dense(256,activation='relu',input_shape=(xtrain.shape[1],)))
model_DL.add(Dropout(0.2))
model_DL.add(layers.Dense(256,activation='relu'))
model_DL.add(Dropout(0.2))
model_DL.add(layers.Dense(5,activation='softmax'))


model_DL.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model_DL.fit(xtrain,ytrain,epochs=8,batch_size=512)

ypred_nn=model_DL.predict(xtest)

pred_nn=[]
from numpy import argmax

for x in ypred_nn:
    pred_nn.append(np.argmax(x))
submission['Sentiment']=pred_nn
submission.to_csv('sampleSubmission.csv',index=False)