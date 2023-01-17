# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/indian_liver_patient.csv")
print(data.columns)
print(data.shape)
data.head(5)
"""find the maximum age included"""
# print(data.Age.value_counts())
plt.figure(figsize=(25,4))
data.Age.value_counts().plot(kind="bar")
plt.xlabel("age")
plt.ylabel("counts")
#one hot encodeing for the gender
gender_ = pd.get_dummies(data.Gender)
gender_ = gender_.iloc[:,1]
data["Gender"] = gender_
data = data.dropna()
# data.head(10)
data.head(4)
scaler = MinMaxScaler(feature_range=(0,1))
x = data.iloc[:,:10]
y = data.iloc[:,10]

scaled_x = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = RandomForestClassifier(n_estimators=10)
# model = GaussianNB(priors=None)
model.fit(x_train,y_train)
org = y_test
pred = model.predict(x_test)
acc = accuracy_score(org,pred)
print("accuracy with random forest:{}".format(acc*100))
# print("accuracy with naive_bayes:{}".format(acc*100))

