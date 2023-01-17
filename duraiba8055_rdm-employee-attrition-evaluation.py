# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data manipulation

import numpy as np

import pandas as pd



# data visualisation

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

from sklearn import metrics



# sets matplotlib to inline

%matplotlib inline  



# importing LogisticRegression for Test and Train

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

df = pd.read_csv("../input/HR-Employee-Attrition.csv")
df.head()
df.shape
df.columns
df.dtypes
df['Attrition'] = df['Attrition'].map(lambda x: 1 if x== 'Yes' else 0)
df.isnull().any()
df.isnull().sum()
df.corr()
df.duplicated().sum()
def plot_factorplot(attr,labels=None):

    sns.catplot(data=df,kind='count',height=5,aspect=1.5,x=attr)

cat_df=df.select_dtypes(include='object')



for i in cat_df:

    plt.figure(figsize=(15, 15))

    plot_factorplot
df.drop(labels=['EmployeeCount','EmployeeNumber','StockOptionLevel','StandardHours'],axis=1,inplace=True)

df.head()
df.corr()
df.cov()
cat_col = df.select_dtypes(exclude=np.number)

cat_col
for i in cat_col:

    print(df[i].value_counts())
numerical_col = df.select_dtypes(include=np.number)

numerical_col
for i in cat_col:

    print(df[i].value_counts())
for i in numerical_col:

    print(i)
df.BusinessTravel.value_counts()
df.columns.shape
one_hot_categorical_variables = pd.get_dummies(cat_col)
one_hot_categorical_variables.head()
df = pd.concat([numerical_col,one_hot_categorical_variables],sort=False,axis=1)

df.head()
x = df.drop(columns='Attrition')

y = df['Attrition']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=12)

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

train_Pred = logreg.predict(x_train)
metrics.confusion_matrix(y_train,train_Pred)
metrics.confusion_matrix(y_train,train_Pred)
test_Pred = logreg.predict(x_test)
metrics.confusion_matrix(y_test,test_Pred)
metrics.accuracy_score(y_test,test_Pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, test_Pred))