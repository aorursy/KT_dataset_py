# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





df= pd.read_csv('../input/titanic-extended/full.csv')

df.head()





df.shape #shape of the dataset(1309,21)
#names of the columns

df.columns
#Lets see some columns individually



df['SibSp'].head()

#Handling missing values and data types

#datatypes

df.dtypes
df.isnull().sum() #sum of the total missing values

df=df.fillna(method='ffill')

df.isnull().sum()
#Filling some more missing values

df['Body']=df['Body'].fillna(method='bfill')





df['Cabin']=df['Cabin'].fillna(method='bfill')





df['Lifeboat']=df['Lifeboat'].fillna(method='bfill')



df.isnull().sum()
#Changing the datatype

df['Sex']=df['Sex'].astype('category')

df['Sex'].dtypes
df['Survived']=df['Survived'].astype(int)

df['Survived'].dtypes
df['Age']=df['Age'].astype(int)

df['Age'].dtypes
# some data query

df['Survived'].value_counts()
#grouby data



g1=df.groupby('Sex')

g1['Survived'].value_counts()
#Preprocessing: Feature selection

df.drop(['Name','Ticket','Cabin','WikiId','Name_wiki','Age_wiki','Hometown','Boarded','Destination','Lifeboat','Body','Class'],axis=1,inplace=True)

df.head()
#correlation

cor_matrix=df.corr()

cor_matrix['Survived'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes=['Survived','Age','Sex','Pclass']

scatter_matrix(df[attributes],figsize=(12,8))

#one hot encoding 

import pandas as pd



x = pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)

x.head()
y=x['Survived'].copy()

x.drop('Survived', axis=1, inplace=True)

x.head()
y.head()
#StandardScaler

from sklearn import preprocessing

Feature=preprocessing.StandardScaler().fit(x).transform(x)

Feature[0:5]
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test= train_test_split(Feature, y, test_size=0.2, random_state=4)

print ('Train_set:',x_train.shape, y_train.shape)

print ('Test_set:',x_test.shape, y_test.shape)
# building our model

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(C=0.01, solver='liblinear')

model=clf.fit(x_train, y_train)
pred=model.predict(x_train)

pred[0:5]
# cross_validation

from sklearn.model_selection import cross_val_score

cross_val= cross_val_score(model, x_train, y_train, cv=3, scoring='accuracy')

cross_val.mean()

#f1-score

from sklearn.metrics import f1_score

f1_score(y_train, pred)
#predicting on test_set

y_pred=model.predict(x_test)

y_pred[0:5]
#f1-score

from sklearn.metrics import f1_score

f1_score(y_test, y_pred)
#confusion_matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
#DecisionTree

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(criterion='entropy', max_depth=4)

clf.fit (x_train, y_train)
pred_2=clf.predict (x_test)

pred_2[0:5]
#f1-score

from sklearn.metrics import f1_score

f1_score(y_test, pred_2)