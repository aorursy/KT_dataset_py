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
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
train_data=pd.read_json('../input/train.json')
test_data=pd.read_json('../input/test.json')
target=train_data.cuisine.values
print(target.shape)
def preprocess(data):
    Mydata=[]
    for i in range(len(data['ingredients'])):
        a=" ".join(data['ingredients'][i])
        Mydata.append(a)
    return Mydata

train=preprocess(train_data)
test=preprocess(test_data)
vect=TfidfVectorizer()
train_TDF=vect.fit_transform(train)
test_TDF=vect.transform(test)

# train_TDF=tfidf_features(train,flag='train')
# test_TDF=tfidf_features(test,flag='test')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
enc_target=le.fit_transform(target)
enc_target.shape
print(train_TDF.shape,enc_target.shape,test_TDF.shape)
# from sklearn.model_selection import train_test_split
# trainX,testX,trainY,testY=train_test_split(train_TDF,enc_target,test_size=0.3)
# print ("Train the model ... ")
# classifier = SVC(C=100, # penalty parameter
# 	 			 kernel='rbf', # kernel type, rbf working fine here
# 	 			 degree=3, # default value
# 	 			 gamma=1, # kernel coefficient
# 	 			 coef0=1, # change to 1 from default value of 0.0
# 	 			 shrinking=True, # using shrinking heuristics
# 	 			 tol=0.001, # stopping criterion tolerance 
# 	      		 probability=False, # no need to enable probability estimates
# 	      		 cache_size=200, # 200 MB cache size
# 	      		 class_weight=None, # all classes are treated equally 
# 	      		 verbose=False, # print the logs 
# 	      		 max_iter=-1, # no limit, let it run
#           		 decision_function_shape=None, # will use one vs rest explicitly 
#           		 random_state=None)
# model = OneVsRestClassifier(classifier, n_jobs=4)
# from xgboost import XGBClassifier
# model=XGBClassifier(gamma=0,learning_rate=0.1,min_child_weight=5,n_jobs=4,eval_metric='merror',n_estimators=1000)

train_data
Y=pd.get_dummies(train_data.cuisine)
input_data=Y.values
train_TDF.toarray()
# from scipy.sparse import csr_matrix
# sparse_dataset = csr_matrix(train_TDF)
# featuresNN = sparse_dataset.todense()
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(train_TDF.toarray())
featuresNN=scaler.transform(train_TDF.toarray())

from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY=train_test_split(featuresNN,input_data,test_size=0.3)
print(trainX.shape, testX.shape, trainY.shape, testY.shape)
numfeat=trainX.shape[1]
numfeat
import keras
from keras.layers import *
from keras.models import Sequential

model=keras.models.Sequential()
model.add(Dense(300,input_dim=numfeat,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(400,activation='relu'))
model.add(Dense(20,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer = 'adam',)
model.fit(featuresNN,input_data,epochs=50,shuffle=True,batch_size=100)
# classifier = SVC(kernel='rbf',degree=3,gamma=1,coef0=1, shrinking=True, tol=0.001,probability=False,class_weight=None,max_iter=-1,decision_function_shape=None,random_state=None)
# #model = OneVsRestClassifier(classifier,n_jobs=2)
# param={'cache_size':[200,300],'C':[10,100]}
# model=GridSearchCV(cv=5,estimator=classifier,iid=True,param_grid=param,error_score='mae')
# %%time
# model.fit(train_TDF, enc_target)

a=model.predict_classes(test_TDF)
predicted=le.inverse_transform(a)
predicted

sub=pd.DataFrame({'id':test_data.id,'cuisine':predicted})
sub.to_csv('sample_submission.csv',index=False)
# tfidf = TfidfVectorizer(binary=True)
# def tfidf_features(text, flag):
#     print ("TF-IDF on text data ... ")
#     if flag == "train":
#         x = tfidf.fit_transform(text)
#     else:
#         x = tfidf.transform(text)
#     x = x.astype('float16')
#     print()
#     return x 
# train_TDF=tfidf_features(train,flag='train')
# test_TDF=tfidf_features(test,flag='test')
# print(train_TDF.shape,test_TDF.shape)