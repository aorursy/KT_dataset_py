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
import pandas as pd

import numpy as np



df=pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
df.shape
df.columns
y=df['Survived']

df.drop(['Survived'], axis=1, inplace=True)
df.drop(['Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
df['Sex']=[0 if i == 'male' else 1 for i in df.Sex ]
df['Age'].fillna(df.Age.mean(), inplace=True)
df['Cabin'].fillna(0, inplace=True)

df['Cabin']=[0 if i== 0 else 1 for i in df.Cabin]

df.head()
df.drop('Embarked', axis=1, inplace=True)
age=df[['Age']].values

age=age.reshape(-1, 1)
fare=df[['Fare']].values

fare=fare.reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

df['Age']=scaler.fit_transform(age)

df['Fare']=scaler.fit_transform(fare)
df.head()
pclass=df[['Pclass']].values

pclass=pclass.reshape(-1, 1)
from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder(categories='auto')

data_pclass=ohe.fit_transform(pclass).toarray()
data=data_pclass[:,1:]

data
df1=pd.DataFrame(data, columns=['pclass1', 'pclass2'])

df=pd.concat([df, df1], axis=1)

df.drop('Pclass', axis=1, inplace=True)
df.head()
df.isnull().sum()
from sklearn.model_selection import train_test_split, RandomizedSearchCV

X_train, X_test, y_train, y_test= train_test_split(df, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier



rfc=RandomForestClassifier(n_estimators=150, random_state=0)

param_dist = {"max_depth": [8, 10, 12],

              "max_features": [8, 8, 8],

              "min_samples_split": [3, 4, 5],

              'min_samples_leaf': [2, 3, 4],

              "bootstrap": [True, True, True],

              "criterion": ['entropy', "gini", "gini"]}



n_iter_search = 25

random_search = RandomizedSearchCV(rfc, param_distributions=param_dist,

                                   n_iter=n_iter_search, cv=5, iid=False)

random_search.fit(X_train, y_train)
y_pred=random_search.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix

print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
test_df=pd.read_csv('/kaggle/input/titanic/test.csv')

test_df.head()
def clean_dataset(df):

    passengerId=df['PassengerId']

    df.drop(['Name', 'PassengerId', 'Ticket', 'Embarked'], axis=1, inplace=True)

    df['Sex']=[0 if i == 'male' else 1 for i in df.Sex]

    df['Age'].fillna(df.Age.mean(), inplace=True)

    df['Cabin'].fillna(0, inplace=True)

    df['Fare'].fillna(df.Fare.mean(), inplace=True)

    df['Cabin']=[0 if i== 0 else 1 for i in df.Cabin]

    age=df[['Age']].values

    age=age.reshape(-1, 1)

    fare=df[['Fare']].values

    fare=fare.reshape(-1, 1)

    df['Age']=scaler.fit_transform(age)

    df['Fare']=scaler.fit_transform(fare)

    pclass=df[['Pclass']].values

    pclass=pclass.reshape(-1, 1)

    data_pclass=ohe.fit_transform(pclass).toarray()

    data=data_pclass[:,1:]

    df1=pd.DataFrame(data, columns=['pclass1', 'pclass2'])

    df=pd.concat([df, df1], axis=1)

    df.drop('Pclass', axis=1, inplace=True)

    return df
test_df1= clean_dataset(test_df)
test_df1.head()
test_df1.isnull().sum()
pred_test_dataset=random_search.predict(test_df1)
pred_test_dataset
pred_test_dataset_df=pd.DataFrame(pred_test_dataset, columns=['Survived'])
df_test=pd.read_csv('/kaggle/input/titanic/test.csv')


pd.concat([df_test['PassengerId'], pred_test_dataset_df['Survived']], axis=1).to_csv(('submission_file.csv'), index=None, header=True)