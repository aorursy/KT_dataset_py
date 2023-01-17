import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-

#Import Necessary Packages

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix

df=pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')

df1=df.copy()
sns.set(rc={'figure.figsize':(11.7,8.27)})
columns=list(df1.columns)

print(columns)
print(df1.isnull().sum())
print(df1.dtypes)
print(np.unique(df1['Department'])) 
print(np.unique(df1['salary']))
sns.boxplot(df1['left'],df1['satisfaction_level'])
sns.boxplot(df1['left'],df1['last_evaluation'])
sns.boxplot(df1['left'],df1['number_project'])
sns.barplot(x=df1['salary'],y=df1['left'],hue=df1['Department'],ci=False)
regr=LogisticRegression()

#split the dependent and independent variables

newdf=pd.get_dummies(df1,drop_first=True)

colist=list(newdf.columns)

features=list(set(colist)-set(['left']))

y=newdf['left'].values

x=newdf[features].values

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=2)

regr.fit(train_x,train_y)

prediction=regr.predict(test_x)

mat=confusion_matrix(test_y,prediction)

acc=accuracy_score(test_y,prediction)

print(acc)