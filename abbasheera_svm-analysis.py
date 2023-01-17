import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import fun_py as fp

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn import metrics

import os

print(os.listdir("../input"))

data=pd.read_csv("../input/univ-bank-svm/uivbank.csv")
fp.data_groupcols(data)
fp.data_head(data,3)
fp.data_shape(data)
data.rename(columns={'ZIP Code': 'Zip_Code','Personal Loan': 'Personal_Loan','Securities Account' : 'Securities_Account' ,'CD Account' : 'CD_Account'},inplace=True)
fp.data_duplicates(data,0)
fp.data_isna(data)
fp.data_nullcols(data,0)
fp.data_corr_trg_col(data,'Personal_Loan')
# ID,Age,Experience,Zip Code,CredictCard,Online Sec Acc and Family are not much related , it's getting removed.
rData=fp.data_drop_cols(data,['ID','Age','Experience','Zip_Code','CreditCard','Online','Securities_Account','Family'])
fp.data_head(rData,3)
fp.data_head(data,3)
x=fp.data_drop_cols(rData,'Personal_Loan')

y=rData['Personal_Loan']
print(x.head(3))

print(y.head(3))
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
y_train = np.ravel(y_train)

from sklearn import svm

clf = svm.SVC(kernel='rbf')

clf.fit(x_train, y_train)
yhat = clf.predict(x_test)

uni,counts = np.unique(yhat)

print(np.asarray((uni, counts)).T)
from sklearn.metrics import classification_report, confusion_matrix

import itertools
from sklearn.metrics import f1_score

f1_score(y_test, yhat, average='weighted')
from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y_test, yhat)