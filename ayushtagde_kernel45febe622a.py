import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

train_df = pd.read_csv('../input/titanic/train.csv')

train_df
train_df.shape
train_df.dtypes
#Finding null values

train_df.isnull().sum()

from sklearn.impute import SimpleImputer



imputer = SimpleImputer(np.nan, "mean")



train_df['Age'] = imputer.fit_transform(np.array(train_df['Age']).reshape(891,1))

train_df.Embarked.fillna(method='ffill', inplace=True) # 2nd

train_df.drop(['PassengerId', 'Name', 'Cabin' , 'Ticket'], axis=1, inplace=True) 

train_df
sns.countplot(x = 'Survived' , hue = 'Sex' , data = train_df)
sns.countplot(x = 'Embarked' , hue = 'Sex' , data = train_df)
sns.distplot(train_df['Age'] ,kde = False, bins = 15,color = 'r')
fig = plt.figure(figsize = (25,7))

sns.violinplot(x = 'Sex' , y = 'Age' , hue = 'Survived'

              ,data = train_df , split = True )
train_df.Sex[train_df.Sex == 'female'] = 0

train_df.Sex[train_df.Sex == 'male'] = 1

train_df.head()
train_df['Embarked'].unique()
train_df.Embarked[train_df.Embarked == 'S'] = 0

train_df.Embarked[train_df.Embarked == 'C'] = 1

train_df.Embarked[train_df.Embarked == 'Q'] = 2

train_df.head()

X_train = train_df.drop('Survived' , axis = 1)

y_train = train_df['Survived']
test = pd.read_csv('../input/titanic/test.csv')

test.head()
test.shape
test.isnull().sum()
test['Age'] = imputer.fit_transform(np.array(test['Age']).reshape(418,1))

test['Fare'] = imputer.fit_transform(np.array(test['Fare']).reshape(418,1))

test.drop(['Name', 'Cabin' , 'Ticket'], axis=1, inplace=True)

test.Sex[test.Sex == 'female'] = 0

test.Sex[test.Sex == 'male'] = 1



test.Embarked[test.Embarked == 'S'] = 0

test.Embarked[test.Embarked == 'C'] = 1

test.Embarked[test.Embarked == 'Q'] = 2



test



X_test = test.drop('PassengerId' , axis = 1).copy()

X_test.isnull().sum()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



model = RandomForestClassifier(n_estimators=100)

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

print('Score :', round(model.score(X_train , y_train)*100 ,2))
submission = pd.DataFrame( { "PassengerId" : test['PassengerId'] ,

                           "Survived" : y_pred

                           })

submission