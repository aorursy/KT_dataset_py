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
df = pd.read_csv("../input/advertising.csv")
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df.head()
sns.scatterplot(x=df['Daily Time Spent on Site'],y=df['Daily Internet Usage'],hue=df['Clicked on Ad'])
plt.hist(x=df['Age'],bins=30)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=df,color='red',kind='kde')
sns.pairplot(df,hue='Clicked on Ad',palette='bwr')
df.head()
example=df['Timestamp'][0].split()[0]
example
df['Date'] = df['Timestamp'].apply(lambda x:x.split()[0])
df['Time'] = df['Timestamp'].apply(lambda x:x.split()[1])
df.head(2)
df.drop('Timestamp',axis=1,inplace = True)
df.head(2)
df.loc[(df['Time']<'12:00:00'),'Daytime']='Morning'
df.loc[(df['Time']>='12:00:00')&(df['Time']<'17:00:00'),'Daytime']='Afternoon'
df.loc[(df['Time']>='17:00:00')&(df['Time']<'21:00:00'),'Daytime']='Evening'
df.loc[(df['Time']>='21:00:00')&(df['Time']<'23:50:00'),'Daytime']='Night'
df.head(2)
df['Month']=df['Date'].apply(lambda x:x.split('-')[1])
df.head(2)
sns.countplot(x=df['Month'],hue=df['Clicked on Ad'])
from sklearn.model_selection import train_test_split
df.drop(['Ad Topic Line','City','Country','Date'],axis=1,inplace = True)
df.head(2)
Daytime = pd.get_dummies(df['Daytime'],drop_first=True)
df = pd.concat([df,Daytime],axis=1)
df.head(2)
df.columns
df.drop(['Daytime'],axis=1,inplace = True)
df.info()
df['MonthInt']=df['Month'].apply(lambda x:int(x))
df.drop(['Month'],axis=1,inplace = True)
df.drop(['Time'],axis=1,inplace = True)
df.info()
X = df.drop(['Clicked on Ad'],axis = 1)
y= df['Clicked on Ad']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train,y_train)
from sklearn.metrics import classification_report,confusion_matrix
predictions = lm.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
