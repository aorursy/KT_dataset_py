# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

from xgboost import XGBRegressor

from sklearn import tree





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train.tail()
sns.heatmap(train.isnull(),yticklabels=False,cbar = False, cmap = 'viridis')
train.isnull().sum().sort_values(ascending = False)
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train, palette = 'RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette ='RdBu_r')
train['Age'].hist(bins=30)
sns.countplot(x = 'SibSp', data=train)
sns.countplot(x = 'Parch', data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='spring')
plt.figure(figsize=(12,7))

plt.ylim(0,60)

sns.boxplot(x='Pclass',y='Fare',data=train,palette='spring')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else :

            return 24

    else:

        return Age

def impute_fare(cols):

    Fare = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Fare):

        if Pclass == 1:

            return 59

        elif Pclass == 2:

            return 13

        else:

            return 8

    else:

        return Fare
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis = 1)
sns.heatmap(train.isnull())
train.drop('Cabin', axis= 1, inplace=True)
train.head()
train.dropna(inplace=True)
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embarked = pd.get_dummies(train['Embarked'],drop_first = True)
train.drop(['Sex','Embarked','Name','Ticket'],axis = 1, inplace=True)
train = pd.concat([train,sex,embarked],axis =1)
train.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

sex_test = pd.get_dummies(test_data['Sex'],drop_first=True)

embarked_test = pd.get_dummies(test_data['Embarked'],drop_first = True)

test_data.drop('Cabin',axis=1,inplace=True)



test_data['Age']=test_data[['Age','Pclass']].apply(impute_age,axis=1)

test_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test_data = pd.concat([test_data,sex_test,embarked_test],axis=1)

test_data.dropna()

total_df = pd.merge(test_data,gender_sub)

test_data.head()
classifier = RandomForestClassifier(n_estimators=50)

C = 1.0

targets = train['Survived']

X = train.drop(['Survived','PassengerId'],axis=1)

scale = MinMaxScaler(feature_range=(-1,1)).fit(X)

X= scale.transform(X)



svc = svm.SVC(kernel='linear',C=C).fit(X,targets)

total_df['Fare'] = total_df[['Fare','Pclass']].apply(impute_fare,axis = 1)
targets_test = total_df['Survived']

X_test = total_df.drop(['Survived','PassengerId'],axis=1)

X_test = scale.transform(X_test)

predictions = svc.predict(X_test)

print(svc.score(X_test,targets_test))



data = {'PassengerId':total_df['PassengerId'].values,'Survived':predictions}

res = pd.DataFrame(data)

res.head()
csv_data = res.to_csv(index=False)

mf = open('subSVC.txt','w')

print(csv_data,file = mf)

mf.close()