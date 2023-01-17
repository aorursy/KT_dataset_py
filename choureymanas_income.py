# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import plotly.graph_objs as go

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score , confusion_matrix ,f1_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot  as plt

df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')

df_copy = df.copy()

df[df=='?'] = np.nan

df.head()
pd.value_counts(df['income'])
pd.value_counts(df['income'])/df.shape[0] *100
#Doing resampling

df1 = df.loc[df['income'] == '<=50K'].sample(frac =0.3)

df2 = df.loc[df['income'] == '>50K']

df = pd.concat([df1,df2])
pd.value_counts(df['income'])
for col in df:

    df[col].fillna(df[col].mode()[0] , inplace = True) 
df.head()
df.describe()
df.columns
pd.value_counts(df['workclass'])
plt.figure(figsize = (20,8))

sns.countplot(x = 'workclass' ,data =df)

plt.show()
#Lets understand data

plt.figure(figsize = (20,15))

sns.countplot(x = 'workclass' ,data =df,hue = 'income')

plt.show()
plt.figure(figsize = (12,8))

sns.boxplot(x = 'income' , y = 'hours.per.week' , data =df)

#loca,labe = plt.xticks()

plt.xticks(np.arange(2) , ['Less than 50k','More than 50k'])

plt.xlabel('Income')

plt.legend()

plt.show()
sns.distplot(df['age'])
df.shape
df.drop(['education'],axis =1,inplace =True)
df.shape
pd.value_counts(df['marital.status'])
plt.figure(figsize = (20,8))

sns.countplot(x = 'marital.status' , hue = 'income' , data =df)

plt.show()
for col in df:

    le = LabelEncoder()

    if df.dtypes[col] == np.object:

        df[col] = le.fit_transform(df[col])
df.describe()
#Models

lr = LogisticRegression()

clf = DecisionTreeClassifier()

rfc = RandomForestClassifier()
plt.figure(figsize =(20,8))

sns.heatmap(data = df.corr() ,annot =True , cmap="YlGnBu")

plt.show()
#Features

X = df.drop(['income'],axis = 1)

Y = df['income']
#split

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2)
pd.value_counts(Y_train)
score =[]

for i in [lr , clf , rfc]:

    i.fit(X_train,Y_train)

    Y_predict = i.predict(X_test)

    s = accuracy_score(Y_test,Y_predict)

    score.append(s)

    #print(confusion_matrix(Y_test,Y_predict))

    print(f1_score(Y_test,Y_predict))
model = ['lr' , 'clf' , 'rfc']

for i in range(0,3):

    print(str(model[i]) + ":" +str(score[i]))
df.nunique()
df = pd.concat([(pd.get_dummies(df['occupation'])) , df ] ,axis =1).drop(['occupation'] ,axis =1)

df = pd.concat([(pd.get_dummies(df['native.country'])) , df ] ,axis =1).drop(['native.country'] ,axis =1)
df.shape
X1 = df.drop(['income'],axis = 1)

Y1 = df['income']

X1_train , X1_test , Y1_train , Y1_test = train_test_split(X1,Y1,test_size = 0.2)
#Models

lr1 = LogisticRegression()

clf1 = DecisionTreeClassifier()

rfc1= RandomForestClassifier()
score1 =[]

for i in [lr1 , clf1 , rfc1]:

    i.fit(X1_train,Y1_train)

    Y1_predict = i.predict(X1_test)

    s1 = accuracy_score(Y1_test,Y1_predict)

    score1.append(s1)

    #print(confusion_matrix(Y_test,Y_predict))

    print(f1_score(Y1_test,Y1_predict))
model = ['lr' , 'clf' , 'rfc']

for i in range(0,3):

    print(str(model[i]) + ":" +str(score1[i]))