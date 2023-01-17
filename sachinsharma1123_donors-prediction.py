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
df=pd.read_csv('/kaggle/input/donorsprediction/Raw_Data_for_train_test.csv')
df
train.isnull().sum()
#target d contains a lot of null values ,so delete this column

df=df.drop(['TARGET_D'],axis=1)
df['DONOR_AGE']=df['DONOR_AGE'].fillna(df['DONOR_AGE'].mean())
df['DONOR_AGE']=df['DONOR_AGE'].astype('int64')
df['INCOME_GROUP']=df['INCOME_GROUP'].fillna(df['INCOME_GROUP'].mode()[0])
df['INCOME_GROUP']=df['INCOME_GROUP'].astype('int64')
df['WEALTH_RATING']=df['WEALTH_RATING'].fillna(df['WEALTH_RATING'].mode()[0])
df['WEALTH_RATING']=df['WEALTH_RATING'].astype('int64')
df=df.dropna()
#we are done with filling null values now there are still some other characters in some columns
df['URBANICITY']=df['URBANICITY'].str.replace('?','S')
df['SES']=df['SES'].str.replace('?','2')

df['SES']=df['SES'].astype('int64')
df['CLUSTER_CODE']=df['CLUSTER_CODE'].str.replace('.','40')

df['CLUSTER_CODE']=df['CLUSTER_CODE'].astype('int64')
import seaborn as sns

import matplotlib.pyplot as plt

sns.barplot(x=df['TARGET_B'],y=df['IN_HOUSE'],data=df)
sns.barplot(x=df['TARGET_B'],y=df['URBANICITY'],data=df)
sns.countplot(x=df['TARGET_B'],

              hue=df['SES'],data=df)
sns.countplot(x=df['TARGET_B'],hue=df['HOME_OWNER'],data=df)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
y=df['TARGET_B']

x=df.drop(['TARGET_B'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr=LogisticRegression(max_iter=10000)

lr.fit(x_train,y_train)

pred_1=lr.predict(x_test)

score_1=accuracy_score(y_test,pred_1)
score_1
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,21):

    

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(x_test)

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)

    
max(list_1)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(x_test)

score_2=accuracy_score(y_test,pred_2)
score_2