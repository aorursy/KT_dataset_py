# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/creditcard.csv")
df.head()
df.info()
y=df.Class.values # Output
x_data=df.drop(["Class"],axis=1) # Features
x_train,x_test,y_train,y_test=train_test_split(x_data,y,test_size=0.2,random_state=42) # %20 test, %80 train data
x_train=(x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train)).values # normalization
x_test=(x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test)).values # normalization
lr=LogisticRegression()
lr.fit(x_train,y_train)
score=lr.score(x_test,y_test) # test accuracy
print("test accuracy {}".format(score))
y_list=list(y_test) # real data
pre_list=list(lr.predict(x_test)) #predict data
print("real non fraud {},predict non fraud {}".format(y_list.count(0),pre_list.count(0)))
print("real fraud {},predict fraud {}".format(y_list.count(1),pre_list.count(1)))
print(len(y_list),"of test applied and {} predict correctly".format(int(len(pre_list)*score)))
