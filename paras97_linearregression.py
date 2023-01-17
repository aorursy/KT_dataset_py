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
#Fetching the Dataset
dataset=pd.read_csv("../input/Admission_Predict.csv")
X=dataset.iloc[:,0:8].values
y=dataset.iloc[:,8].values

#Viewing the relation of all the features
X_df=pd.DataFrame(X)
pd.plotting.scatter_matrix(X_df)


#Splitting the data into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#Using Linear Regression 

from sklearn.linear_model import LinearRegression
LinReg=LinearRegression()
LinReg.fit(X_train,y_train)
y_pred=LinReg.predict(X_test)

#Finding the accuracy
LinReg.score(X,y)
LinReg.score(X_test,y_test)
LinReg.score(X_train,y_train)


