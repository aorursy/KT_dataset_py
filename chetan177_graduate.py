# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as pt

import sklearn.linear_model as sk

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
os.chdir("../input")

data = pd.read_csv('Admission_Predict_Ver1.1.csv')

data.head()
# Visualize data

pt.scatter(data.iloc[:,1:2],data.iloc[:,8])

pt.xlabel("GRE Score")

pt.ylabel("Chance of admit")

pt.show()

pt.scatter(data.iloc[:,2:3],data.iloc[:,8])

pt.xlabel("TOEFL Score")

pt.ylabel("Chance of admit")

pt.show()

pt.scatter(data.iloc[:,3:4],data.iloc[:,8])

pt.xlabel("University Rating")

pt.ylabel("Chance of admit")

pt.show()

pt.scatter(data.iloc[:,4:5],data.iloc[:,8])

pt.xlabel("SOP")

pt.ylabel("Chance of admit")

pt.show()

pt.scatter(data.iloc[:,5:6],data.iloc[:,8])

pt.xlabel("	LOR")

pt.ylabel("Chance of admit")

pt.show()

pt.scatter(data.iloc[:,6:7],data.iloc[:,8])

pt.xlabel("CGPA")

pt.ylabel("Chance of admit")

pt.show()

pt.scatter(data.iloc[:,7:8],data.iloc[:,8])

pt.xlabel("Research")

pt.ylabel("Chance of admit")

pt.show()
lin_reg = sk.LinearRegression()   # Linear Regression Model 

data_norm = data 

data_norm = (data_norm  - data_norm.mean())/data_norm.std()  # Mean Normalization of the data because the differnces in the values is large
X = data.iloc[:,1:8]

Y = data.iloc[:,8:]

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
print(x_train.shape[0],y_train.shape[0])

print(x_test.shape[0],y_test.shape[0])
lin_reg.fit(x_train,y_train)
lin_reg.predict(x_test)
lin_reg.score(x_test,y_test)