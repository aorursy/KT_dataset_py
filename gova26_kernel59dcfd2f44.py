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
import pandas as pd
#importing the dataset:

a=pd.read_csv("../input/pd_speech_features.csv")

print(a.head(n=6))
#checking for the info and null values:

print(a.info())

print(a.isnull())

print(a.isna())

#no null values found.
#checking the summary of the dataset:

print(a.describe())
import seaborn as sns

#print(a.head())

print(type(a))

print(a.corr())

#splitting x and y:

#print(a.columns)

x=a.drop(["Unnamed: 0","Unnamed: 753","Unnamed: 754"],axis=1)

print(x.head())

y=a.iloc[:,754]

print(y.head())
#splitting traing an d testing set:

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

print(x_train.head())

x_train=pd.DataFrame(x_train)

#print(type(y_train))

print(x_train.shape)

print(y_train.shape)
#fitting a logistic regression model:

from sklearn.linear_model import LogisticRegression

lm=LogisticRegression()

model=lm.fit(x_train,y_train)

print(model.coef_)

print(model.intercept_)
pred=model.predict(x_test)