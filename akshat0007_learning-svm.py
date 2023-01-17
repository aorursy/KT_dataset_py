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
df=pd.read_csv("../input/cell_samples.csv")
df.head()
import matplotlib.pyplot as plt


df.dtypes
df=df[pd.to_numeric(df["BareNuc"],errors='coerce').notnull()]
df.dtypes
df["BareNuc"]=df["BareNuc"].astype('int')
X = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

y=df["Class"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn import svm
from sklearn.metrics import jaccard_similarity_score as jsc
clf=svm.SVC(kernel='rbf').fit(x_train,y_train)
yhat=clf.predict(x_test)
yhat
print("The accuracy score for our model when we use RBF kernel is:",jsc(y_test,yhat)*100,"%")
clf=svm.SVC(kernel='linear').fit(x_train,y_train)
yhat1=clf.predict(x_test)
yhat1
print("The accuracy score for our model when we use Linear kernel is:",jsc(y_test,yhat1)*100,"%")
clf=svm.SVC(kernel='sigmoid').fit(x_train,y_train)
yhat2=clf.predict(x_test)

yhat2
print("The accuracy score for our model when we use Polynomial kernel is:",jsc(y_test,yhat2)*100,"%")