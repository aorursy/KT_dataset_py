# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
train_path = '../input/titanic/train.csv'

test_path = '../input/titanic/test.csv'

submission_path = '../input/titanic/gender_submission.csv'



df_train = pd.read_csv(train_path)

df_test = pd.read_csv(test_path)

df_submission = pd.read_csv(submission_path)
df_train.head()
df_train.isnull().sum()
#visualizing nan/null values

sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
features_having_nan = [(cols, df_train[cols].dtype) for cols in df_train.columns if df_train[cols].isnull().sum()>1]

features_having_nan
#visualizing how many survived and how many not survived

sns.set_style('whitegrid')

sns.countplot(x='Survived', data=df_train)
#visualizing how many survived and how many not survived based on sex

sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex', data=df_train)
#handling age attribute 

sns.boxplot(x='Pclass', y='Age', data=df_train)
#imputing age based on Pclass

def impute_age(cols):

    age = cols[0]

    pclass = cols[1]

    if pd.isnull(age):

        if pclass == 1:

            return 38

        elif pclass == 2:

            return 29

        else:

            return 24

    else:

        return age



df_train['Age'] = df_train[['Age', 'Pclass']].apply(impute_age, axis=1)
sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#as cabin is having a lot of nan values, therefore droping it

df_train.drop(['Cabin'], axis=1, inplace=True)



#for test set

df_test.drop(['Cabin'], axis=1, inplace=True) # 90% nan values, so no use of this attribute
#one hot encoding categorical attributes

sex = pd.get_dummies(df_train['Sex'], drop_first=True) #dummy trap technique

Embarked = pd.get_dummies(df_train['Embarked'], drop_first=True) #dummy trap technique



#for test set

sex_test = pd.get_dummies(df_test['Sex'], drop_first=True) #dummy trap technique

Embarked_test = pd.get_dummies(df_test['Embarked'], drop_first=True) #dummy trap technique
df_train.drop(['Sex', 'Embarked', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

df_test.drop(['Name', 'Sex', 'Embarked', 'Ticket'], axis=1, inplace=True)
df_train.head()
df_train = pd.concat([df_train, sex, Embarked], axis=1)

df_test = pd.concat([df_test, sex_test, Embarked_test], axis=1)
y = df_train['Survived'].to_list()

X = df_train.drop('Survived', axis=1)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=142)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

import xgboost
classifiers = [LogisticRegression(), RandomForestClassifier(n_estimators=100), xgboost.XGBClassifier()]



for clf in classifiers:

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print('\nClassifier: ', clf.__class__.__name__)

    print(accuracy_score(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))
df_test.head()
scaler.fit(df_test.drop(['PassengerId'], axis=1))

x_test_scaled = scaler.transform(df_test.drop(['PassengerId'], axis=1))
y_submission_preds = clf.predict(x_test_scaled)
data = {'PassengerId':df_test['PassengerId'], 

        'Survived':y_submission_preds}



submission = pd.DataFrame(data)
submission.head()
df_submission.head()
submission.to_csv('gender_submission.csv', index=False)