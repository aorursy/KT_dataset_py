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
# import libraries:

import numpy as np

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt
#import dataset:

df=pd.read_csv("../input/Admission_Predict.csv")
df.head()
# visualization of dataset:

sns.pairplot(df,hue='Research')
df.shape
sns.scatterplot(df['University Rating'],df['Chance of Admit '],hue='Research',data=df)
sns.distplot(df['GRE Score'],bins=20,kde=False)
sns.distplot(df['TOEFL Score'],bins=20,kde=False)
sns.lmplot('GRE Score','TOEFL Score',hue='Research',data=df)
sns.heatmap(df)
sns.lineplot('GRE Score','TOEFL Score',hue='Research',data=df)
sns.scatterplot(df['GRE Score'],df['CGPA'],hue='Research',data=df)
sns.scatterplot(df['TOEFL Score'],df['CGPA'],hue='Research',data=df)
df=df.drop('Serial No.',axis=1)
df.head()
# make the dataset into the dependent and independent:

X=df.iloc[:,:-1].values

y=df.iloc[:,7].values
X
y
# spliting the datatset into the train and test set:

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# feacture sacling :

#rom sklearn.preprocessing import StandardScaler

#c=StandardScaler()

#_train=sc.fit_transform(X_train)

#_test=sc.fit_transform(X_test)
# fitting the dataset into the model Logistic_regression:

from sklearn.linear_model import LinearRegression

regression=LinearRegression()

regression.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score

result_score=cross_val_score(regression,X_train,y_train,cv=5).mean()
result_score
score=regression.score(X_test,y_test)
score