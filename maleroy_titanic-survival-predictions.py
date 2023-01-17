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
import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.head()
train.nunique()
pd.concat([train.isnull().sum(), test.isnull().sum()],  keys=['train', 'test'], axis=1)
#drop 'Cabin' because too many missing and not relevant

train = train.drop('Cabin', axis = 1)

test = test.drop('Cabin', axis = 1)

#drop 'Ticket' also not relevant

train = train.drop('Ticket', axis = 1)

test = test.drop('Ticket', axis = 1)
#display null

display(test.loc[test['Fare'].isnull()])
#fill with mean value of Pclass == 3 and Embarked == 'S'

test.loc[test['Fare'].isnull(), 'Fare'] = round(test.loc[(test['Pclass'] == 3) & (test['Embarked'] == 'S')].Fare.mean())
#display null

display(train.loc[train['Embarked'].isnull()])

display(train.groupby('Embarked').Fare.describe())

sns.countplot(x="Embarked", data=train)

plt.show()

#the two missing have no SibSp or Parch and Fare gives no clue, we fill 'Embarked' with the most common value by far : 'S'

train.loc[train['Embarked'].isnull(), 'Embarked'] = 'S'
tmp = pd.DataFrame()

#extract Title in Name

train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.').squeeze()

test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.').squeeze()



#merge some of them

train['Title'] = train['Title'].replace(['Mlle', 'Miss'], 'Ms')

train['Title'] = train['Title'].replace(['Mme', 'Lady'], 'Mrs')

test['Title'] = test['Title'].replace(['Mlle', 'Miss'], 'Ms')

test['Title'] = test['Title'].replace(['Mme', 'Lady'], 'Mrs')



#concat to calculate mean with maximum data

tmp['Title'] = pd.concat([train.Title.squeeze(), test.Title.squeeze()])

tmp['Age'] = pd.concat([train.Age.squeeze(), test.Age.squeeze()])

mean_age_by_title = tmp.loc[tmp['Age'].notnull()].groupby('Title').mean().squeeze()



#fill NaN 'Age' with the mean 'Age' by 'Title'

train['Age_filled'] = [round(mean_age_by_title[val['Title']]) if str(val['Age']) == str(np.nan) else val['Age'] for k, val in train.iterrows()]

test['Age_filled'] = [round(mean_age_by_title[val['Title']]) if str(val['Age']) == str(np.nan) else val['Age'] for k, val in test.iterrows()]
display(train.head())

display(test.head())
#quick look of survived by differents features

fig, ax = plt.subplots(1,5, figsize=(20,4))

sns.barplot(x="Pclass", y="Survived", data=train, ax=ax[0])

sns.barplot(x="Sex", y="Survived", data=train, ax=ax[1])

sns.barplot(x="SibSp", y="Survived", data=train, ax=ax[2])

sns.barplot(x="Parch", y="Survived", data=train, ax=ax[3])

sns.barplot(x="Embarked", y="Survived", data=train, ax=ax[4])

fig.show()
#'Pclass' / 'Sex' / 'Embarked' ready to One-Hot Encoding

#Too few values > 1 in 'SibSp' and 'Parch' to be trustful, merging

train.loc[train['SibSp'] > 0, 'SibSp'] = 1

train.loc[train['Parch'] > 0, 'Parch'] = 1



test.loc[test['SibSp'] > 0, 'SibSp'] = 1

test.loc[test['Parch'] > 0, 'Parch'] = 1
#split 'Age'

train['Age_Category'] = pd.cut(train['Age_filled'],

                        bins=[0, 16, 25, 40, 80], labels=[0, 1, 2, 3])

test['Age_Category'] = pd.cut(test['Age_filled'],

                        bins=[0, 16, 25, 40, 80], labels=[0, 1, 2, 3])

plt.subplots(figsize=(10,5))

sns.countplot('Age_Category',hue='Survived',data=train, palette='RdBu_r')

plt.show()
#'Fare' already categorized by 'Pclass'

display(train.groupby('Pclass').Fare.describe())

display(test.groupby('Pclass').Fare.describe())
#drop 'Fare'

train = train.drop('Fare', axis = 1)

test = test.drop('Fare', axis = 1)

#drop 'Name'

train = train.drop('Name', axis = 1)

test = test.drop('Name', axis = 1)

#drop 'Age'

train = train.drop('Age', axis = 1)

test = test.drop('Age', axis = 1)

#drop 'Age_filled'

train = train.drop('Age_filled', axis = 1)

test = test.drop('Age_filled', axis = 1)

#drop 'Title'

train = train.drop('Title', axis = 1)

test = test.drop('Title', axis = 1)
from sklearn.preprocessing import OneHotEncoder



object_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Age_Category']

# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train[object_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(test[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = train.index

OH_cols_test.index = test.index



# Remove categorical columns (will replace with one-hot encoding)

num_train = train.drop(object_cols, axis=1)

num_test = test.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_train = pd.concat([num_train, OH_cols_train], axis=1)

OH_test = pd.concat([num_test, OH_cols_test], axis=1)

display(OH_train.head())

display(OH_test.head())
from sklearn.model_selection import train_test_split



training_data = OH_train.drop(['PassengerId', 'Survived'], axis=1)

target = OH_train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(training_data, target, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier



acc = list()

for n in range(10):

    rfc = RandomForestClassifier()

    rfc.fit(X_train, y_train)

    y_pred_rfc = rfc.predict(X_val)

    acc.append(round(rfc.score(X_val, y_val)*100, 2))

print(sum(acc) / len(acc))
from sklearn.linear_model import LogisticRegression



acc = list()

for n in range(10):

    lr = LogisticRegression()

    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_val)

    acc.append(round(lr.score(X_val, y_val) * 100, 2))

print(sum(acc) / len(acc))
from sklearn.svm import SVC



acc = list()

for n in range(10):

    svc = SVC()

    svc.fit(X_train, y_train)

    y_pred_svc = svc.predict(X_val)

    acc.append(round(svc.score(X_val, y_val) * 100, 2))

print(sum(acc) / len(acc))
from sklearn.ensemble import GradientBoostingClassifier



acc = list()

for n in range(10):

    gbc = GradientBoostingClassifier()

    gbc.fit(X_train, y_train)

    y_pred = gbc.predict(X_val)

    acc.append(round(gbc.score(X_val, y_val) * 100, 2))

print(sum(acc) / len(acc))
pred = gbc.predict(OH_test.drop('PassengerId', axis=1))



output = pd.DataFrame({'PassengerId': OH_test.PassengerId, 'Survived': pred})

output.to_csv('submission.csv', index=False)