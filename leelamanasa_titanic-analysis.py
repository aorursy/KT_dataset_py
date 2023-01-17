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
import matplotlib.pyplot as plt

import seaborn as sns



# Machine learning models

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
train.columns
test.columns
train.describe()
test.describe()
train.isna().any()
test.isna().any()
train.dtypes
train.info()
test.info()
all_data = pd.concat([train,test],sort=False)

all_data.info()

print(all_data.shape)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.catplot(x = 'Survived', kind = 'count', data = train)
sns.catplot(x = 'Survived',hue= 'Sex', kind = 'count', data = train)
sns.catplot(x = 'Survived',hue= 'Pclass', kind = 'count', data = train)
#Fill Missing numbers with median

all_data['Age'] = all_data['Age'].fillna(value=all_data['Age'].median())

all_data['Fare'] = all_data['Fare'].fillna(value=all_data['Fare'].median())
all_data.info()
sns.catplot(x = 'Embarked', kind = 'count', data = all_data)
all_data['Embarked'] = all_data['Embarked'].fillna(value='S')
all_data.info()
sns.barplot(x="Sex", y="Survived", data=all_data)
#Age Group

all_data.loc[ all_data['Age'] <= 16, 'Age'] = 0

all_data.loc[(all_data['Age'] > 16) & (all_data['Age'] <= 32), 'Age'] = 1

all_data.loc[(all_data['Age'] > 32) & (all_data['Age'] <= 48), 'Age'] = 2

all_data.loc[(all_data['Age'] > 48) & (all_data['Age'] <= 64), 'Age'] = 3

all_data.loc[ all_data['Age'] > 64, 'Age'] = 4 
sns.barplot(x="Age", y="Survived", data=all_data)
all_data.head()
for all_data in [train, test] :

    all_data['Title'] = all_data.Name.str.extract('([A-Za-z]+)\.')
train['Title'].value_counts()
for all_data in [train, test] :

    all_data['Title'].replace(['Dona','Lady','Mme','Countess','Ms'],'Mrs',inplace=True)

    all_data['Title'].replace(['Mlle'],'Miss',inplace=True)

    all_data['Title'].replace(['Sir','Capt','Don','Jonkheer','Col','Major','Rev','Dr','Major'],'Mr',inplace=True)

    all_data.drop(['Name'], axis=1,inplace=True)
train['Title'].value_counts()
train.head()
#Fill Missing numbers with median

for df in [train,test]:

    df['Age'] = df['Age'].fillna(value=df['Age'].median())

    df['Fare'] = df['Fare'].fillna(value=df['Fare'].median())
for df in [train,test]:

    df.drop(['Cabin'], axis=1,inplace=True)

    df.drop(['PassengerId'],axis=1,inplace=True)

    df.drop(['Ticket'],axis=1,inplace=True)
train.head()
test.head()
train[['Age','Fare']].describe()
scaler = MinMaxScaler()

for df in [train,test]:

    df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])
for df in [train, test]:

    df['Sex'] = (df['Sex'] == "male").astype(int)
for df in [train, test]:

    df['Pclass'] = df['Pclass'].astype('category')
train = pd.get_dummies(train)

test = pd.get_dummies(test)
print(train.shape)

train.head()
print(test.shape)

test.head()
X_train = train.drop(['Survived'], axis = 1)

y_train = train['Survived']
rfr = RandomForestClassifier(random_state=42)



rfr.fit(X_train, y_train)



rfr_pred = rfr.predict(X_train)



rfr_acc = accuracy_score(y_train, rfr_pred)



print(round(rfr_acc*100,2,),'%')
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.15, random_state=42)
rfr = RandomForestClassifier(random_state=42)



rfr.fit(X_train, y_train)



rfr_pred_train = rfr.predict(X_train)

rfr_pred_test = rfr.predict(X_test)



rfr_acc_train = accuracy_score(y_train, rfr_pred_train)

rfr_acc_test = accuracy_score(y_test, rfr_pred_test)



print('Accuracy on train set: ',round(rfr_acc_train*100,2,),'%')

print('Accuracy on test set:  ',round(rfr_acc_test*100,2,),'%')
rfr = RandomForestClassifier(criterion = "entropy", 

                              min_samples_leaf = 5,

                              min_samples_split = 12,

                              n_estimators=500,

                              random_state=42,

                              n_jobs=-1)

rfr.fit(X_train, y_train)



rfr_pred_train = rfr.predict(X_train)

rfr_pred_test = rfr.predict(X_test)



rfr_ac_train = accuracy_score(y_train, rfr_pred_train)

rfr_ac_test = accuracy_score(y_test, rfr_pred_test)



print('Accuracy on train set: ',round(rfr_ac_train*100,2,),'%')

print('Accuracy on test set:  ',round(rfr_ac_test*100,2,),'%')
rfr.fit(X_train, y_train)



rfr_pred = rfr.predict(test)



test_df = pd.read_csv("/kaggle/input/titanic/test.csv")



submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],

                            "Survived": rfr_pred})



submission.to_csv('submission', index=False)