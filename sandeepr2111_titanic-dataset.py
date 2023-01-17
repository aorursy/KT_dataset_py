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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score
train_df=pd.read_csv('../input/titanic/train.csv')

test_df=pd.read_csv('../input/titanic/test.csv')

gender_df=pd.read_csv('../input/titanic/gender_submission.csv')
train_df.head()
test_df.head()
gender_df.head()
print(train_df.shape)

print(test_df.shape)

print(gender_df.shape)
titanic_df=pd.concat([train_df,test_df],ignore_index=True)
titanic_df.head()
titanic_df.shape
titanic_df.drop(['Name','Ticket'],axis=1,inplace=True)
titanic_df.isna().sum()
titanic_df.drop('Cabin',axis=1,inplace=True)

titanic_df.dropna(subset=['Survived'],inplace=True)

titanic_df['Age']=titanic_df['Age'].fillna(titanic_df['Age'].median())

titanic_df['Embarked']=titanic_df['Embarked'].fillna('S')

titanic_df['Fare']=titanic_df['Fare'].fillna(titanic_df['Fare'].median())
titanic_df.isna().sum()
print(titanic_df['Sex'].unique())

print(titanic_df['Embarked'].unique())
label_enc=LabelEncoder()



for i in ['Sex','Embarked']:

    titanic_df[i]=label_enc.fit_transform(titanic_df[i])
titanic_df.describe()
titanic_df.info()
titanic_df.corr()
plt.figure(figsize=(10,10))

sns.heatmap(titanic_df.corr(),annot=True,square=True,linewidth=2)
sns.countplot(x='Sex',hue='Survived',data=titanic_df)
plt.figure(figsize=(10,8))

sns.countplot(x='Parch',hue='Survived',data=titanic_df)
sns.countplot(x='Pclass',hue='Survived',data=titanic_df)
sns.countplot(x='Pclass',data=titanic_df)
for i in range(len(titanic_df)):

    if titanic_df['Age'][i]<=10:

        titanic_df['Age'][i]=1

    if (titanic_df['Age'][i]>10) & (titanic_df['Age'][i]<=18):

            titanic_df['Age'][i]=2

    if (titanic_df['Age'][i]>18) & (titanic_df['Age'][i]<50):

            titanic_df['Age'][i]=3

    if titanic_df['Age'][i]>=50:

            titanic_df['Age'][i]=4
plt.figure(figsize=(10,8))

sns.countplot(x='Age',hue='Survived',data=titanic_df)
plt.hist(titanic_df['Fare'])
plt.figure(figsize=(10,7))

sns.boxplot(x='Fare',y='Survived',data=titanic_df,orient='h')
plt.figure(figsize=(10,7))

sns.catplot(x='Survived',y='Fare',hue='Sex',data=titanic_df,palette='magma',kind='violin')
sns.countplot(x='Embarked',hue='Survived',data=titanic_df)
X=titanic_df.drop('Survived',axis=1)

y=titanic_df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)
model=LogisticRegression(C=100)

model.fit(X_train,y_train)

test_result=model.predict(X_test)

roc_auc_score(y_test,test_result)
from sklearn.ensemble import RandomForestClassifier

rand_model=RandomForestClassifier()

rand_model.fit(X_train,y_train)

rand_predict=rand_model.predict(X_test)

roc_auc_score(y_test,rand_predict)
svc_model=SVC()

svc_model.fit(X_train,y_train)

svc_predict=svc_model.predict(X_test)

roc_auc_score(y_test,svc_predict)