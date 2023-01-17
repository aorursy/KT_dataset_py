# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
input_path = '../input/titanic/'

df_train = pd.read_csv(input_path + 'train.csv')

df_test = pd.read_csv(input_path + 'test.csv')
# Show 5 first lines

df_train.head()
# Show 5 last lines

df_train.tail()
# Initialize new columns with 0 by default

df_train['YoungAndRich'] = 0

df_test['YoungAndRich'] = 0

df_train['Young'] = 0

df_test['Young'] = 0

df_train['RichYoungMale'] = 0

df_test['RichYoungMale'] = 0

df_train['RichYoungFemale'] = 0

df_test['RichYoungFemale'] = 0

df_train['Deck'] = 0

df_test['Deck'] = 0

df_train['LuckyParch'] = 0

df_test['LuckyParch'] = 0
# New columns value attribution

df_train.loc[df_train['Age'] <= 18, 'Young'] = 1

df_train.loc[df_train['Age'] > 18, 'Young'] = 0

df_test.loc[df_test['Age'] <= 18, 'Young'] = 1

df_test.loc[df_test['Age'] > 18, 'Young'] = 0



df_train.loc[(df_train['Age'] <= 30) & (df_train['Pclass'] <= 2), 'YoungAndRich'] = 1

df_test.loc[(df_test['Age'] <= 30) & (df_test['Pclass'] <= 2), 'YoungAndRich'] = 1



df_train.loc[(df_train['YoungAndRich'] == 1) & (df_train['Sex'] == 'male'),'RichYoungMale'] = 1

df_test.loc[(df_test['YoungAndRich'] == 1) & (df_test['Sex'] == 'male'),'RichYoungMale'] = 1

df_train.loc[(df_train['YoungAndRich'] == 1) & (df_train['Sex'] == 'female'),'RichYoungFemale'] = 1

df_test.loc[(df_test['YoungAndRich'] == 1) & (df_test['Sex'] == 'female'),'RichYoungFemale'] = 1
df_train.head(10)
# Check survivability by sex

sns.barplot(x = 'Sex', y = 'Survived', data = df_train);
# Check survivability by class

sns.barplot(x = 'Pclass', y = 'Survived', data = df_train);
# Survivability based on if is a child or not

sns.barplot(x = 'Young', y = 'Survived', data = df_train);
# Survivability by parent 

sns.barplot(x = 'Parch', y = 'Survived', data = df_train);
# Survivability by Age and social class

sns.barplot(x = "YoungAndRich", y="Survived", data = df_train);
# Surivability by Deck on the boat

sns.barplot(x = df_test['Cabin'].str[0], y = 'Survived', data = df_train);
sub = pd.read_csv(input_path + 'gender_submission.csv')
# Fill NaN columns with average_age variables

average_age = np.sum(df_train['Age'].isnull()) / len(df_train['Age'])

df_train.loc[df_train["Age"].isnull(),"Age"] = average_age 

df_test.loc[df_test["Age"].isnull(),"Age"] = average_age

df_train.loc[df_train["Fare"].isnull(),"Fare"] = 0

df_test.loc[df_test["Fare"].isnull(),"Fare"] = 0

df_train["cabin_is_null"] = np.where(df_train["Cabin"].isnull(), 1, 0)

df_test["cabin_is_null"] = np.where(df_test["Cabin"].isnull(), 1, 0)

df_train.loc[df_train['Cabin'].isnull(),'Cabin'] = 0

df_test.loc[df_test['Cabin'].isnull(),'Cabin'] = 0



df_train.loc[df_train['Cabin'] != 0, 'Deck'] = df_train['Cabin'].str[0]

df_test.loc[df_test['Cabin'] != 0, 'Deck'] = df_test['Cabin'].str[0]



df_train.loc[df_train['Parch'] <= 3, 'LuckyParch'] = 1

df_test.loc[df_train['Parch'] <= 3, 'LuckyParch'] = 1
df_train.head(20)
NUM_FEATURES = ['Age', 'Fare','Pclass', 'Young', 'SibSp', 'Parch', 'cabin_is_null', 'RichYoungFemale', 'RichYoungMale','YoungAndRich', 'LuckyParch']

CAT_FEATURES = ['Sex', 'Embarked', 'Cabin']
for c in CAT_FEATURES:

    df_train[c] = df_train[c].astype('category').cat.codes
# Preparing dataset for model integration

X_train = df_train.loc[:, NUM_FEATURES]

X_train_cat = df_train.loc[:, CAT_FEATURES]

y_train = df_train['Survived']
X_test = df_test.loc[:, NUM_FEATURES]

X_test_cat = df_train.loc[:, CAT_FEATURES]
pd.concat([X_train, X_train_cat], axis = 1)
pd.concat([X_test, X_test_cat], axis = 1)
X_train.head(50)
rf = RandomForestClassifier(n_estimators=100)

print(rf)
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# Cell used to store model result

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index = False)