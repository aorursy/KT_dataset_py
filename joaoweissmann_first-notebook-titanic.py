import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
test_data = pd.read_csv("../input/test.csv")
train_data = pd.read_csv("../input/train.csv")

test_data.set_index('PassengerId', inplace=True)
train_data.set_index('PassengerId', inplace=True)
#train_data.head()
train_data.info()
#train_data.describe()
#train_data['Survived'].value_counts()

#test_data.head()
#test_data.info()
#test_data.describe()
y_train = train_data['Survived']
x_train = train_data.copy()
del x_train['Survived']

x_train.head()
x_train['family size'] = x_train['SibSp'] + x_train['Parch'] + 1
test_data['family size'] = test_data['SibSp'] + test_data['Parch'] + 1

x_train['is alone'] = x_train['family size'].apply(lambda x: 1 if x == 1 else 0)
test_data['is alone'] = test_data['family size'].apply(lambda x: 1 if x == 1 else 0)

x_train['salutation'] = x_train['Name'].str.extract(' ([A-Za-z]+)\.')
salutation_mask_train = x_train['salutation'].value_counts() < 10
x_train['title'] = x_train['salutation'].apply(lambda x: 'Misc' if salutation_mask_train.loc[x] == True else x)
test_data['salutation'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.')
salutation_mask_test = test_data['salutation'].value_counts() < 10
test_data['title'] = test_data['salutation'].apply(lambda x: 'Misc' if salutation_mask_test.loc[x] == True else x)

x_train['name len'] = x_train['Name'].apply(lambda x: len(x))
test_data['name len'] = test_data['Name'].apply(lambda x: len(x))

x_train['ticket len'] = x_train['Ticket'].apply(lambda x: len(x))
test_data['ticket len'] = test_data['Ticket'].apply(lambda x: len(x))

x_train['ticket left'] = x_train['Ticket'].apply(lambda x: str(x)[0])
ticket_left_mask_train = x_train['ticket left'].value_counts() < 30
x_train['ticket left'] = x_train['ticket left'].apply(lambda x: 'Misc' if ticket_left_mask_train.loc[x] == True else x)
test_data['ticket left'] = test_data['Ticket'].apply(lambda x: str(x)[0])
ticket_left_mask_test = test_data['ticket left'].value_counts() < 30
test_data['ticket left'] = test_data['ticket left'].apply(lambda x: 'Misc' if ticket_left_mask_test.loc[x] == True else x)

x_train['has cabin'] = x_train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test_data['has cabin'] = test_data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

x_train['cabin num'] = x_train['Cabin'].str.extract('([0-9]+)')
test_data['cabin num']= test_data['Cabin'].str.extract('([0-9]+)')

x_train['cabin left'] = x_train['Cabin'].apply(lambda x: str(x)[0])
cabin_left_mask_train = x_train['cabin left'].value_counts() < 10
x_train['cabin left'] = x_train['cabin left'].apply(lambda x: 'Misc' if cabin_left_mask_train.loc[x] == True else x)
test_data['cabin left'] = test_data['Cabin'].apply(lambda x: str(x)[0])
cabin_left_mask_test = test_data['cabin left'].value_counts() < 10
test_data['cabin left'] = test_data['cabin left'].apply(lambda x: 'Misc' if cabin_left_mask_test.loc[x] == True else x)

x_train['cabin len'] = x_train['Cabin'].apply(lambda x: len(str(x)))
test_data['cabin len'] = test_data['Cabin'].apply(lambda x: len(str(x)))

test_data['title'].value_counts()
x_train.isnull().sum()
for data in [x_train, test_data]:
    for title in data['title'].unique():
        data.loc[(data['title']==title) & (data['Age'].isnull()), 'Age'] = data.groupby('title')['Age'].median()[title]
        data.loc[(data['title']==title) & (data['Fare'].isnull()), 'Fare'] = data.groupby('title')['Fare'].median()[title]
        
x_train['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)
x_train['Cabin'].fillna('0', inplace=True)
test_data['Cabin'].fillna('0', inplace=True)
x_train['cabin num'].fillna(0, inplace=True)
test_data['cabin num'].fillna(0, inplace=True)
x_train['age bin'] = pd.cut(x_train['Age'].astype(int), 5)
test_data['age bin'] = pd.cut(test_data['Age'].astype(int), 5)

x_train['fare bin'] = pd.cut(x_train['Fare'], 4)
test_data['fare bin'] = pd.cut(test_data['Fare'], 4)

x_train['cabin num bin'] = pd.cut(x_train['cabin num'].astype(int), 5)
test_data['cabin num bin'] = pd.cut(test_data['cabin num'].astype(int), 5)

x_train['name len bin'] = pd.cut(x_train['name len'].astype(int), 5)
test_data['name len bin'] = pd.cut(test_data['name len'].astype(int), 5)

x_train['ticket len bin'] = pd.cut(x_train['ticket len'].astype(int), 5)
test_data['ticket len bin'] = pd.cut(test_data['ticket len'].astype(int), 5)

x_train['cabin len bin'] = pd.cut(x_train['cabin len'].astype(int), 5)
test_data['cabin len bin'] = pd.cut(test_data['cabin len'].astype(int), 5)
test_data['name len bin'].value_counts()
#test_data['salutation'].value_counts()
#features = ['Pclass', 'Sex', 'Age', 'age bin', 'Parch', 'SibSp', 'is alone', 'family size', 'Fare', 'fare bin', 'Embarked', 'title', 'name len', 'name len bin', 'ticket len', 'ticket len bin', 'ticket left', 'cabin num', 'cabin num bin', 'has cabin', 'cabin len', 'cabin len bin', 'cabin left']
features = ['Pclass', 'Sex', 'age bin', 'is alone', 'family size', 'fare bin', 'Embarked', 'title', 'name len', 'ticket len', 'ticket left', 'cabin num', 'has cabin', 'cabin len', 'cabin left']
#features = ['Pclass', 'Sex', 'age bin', 'is alone', 'family size', 'fare bin', 'Embarked', 'title']

x_train = x_train[features]
x_test = test_data[features]

x_train.head()
#x_test.head()
x_train.isnull().sum()
#colunas_dummies = ['Sex', 'age bin', 'fare bin', 'title', 'Embarked', 'ticket left', 'cabin left', 'cabin num bin', 'name len bin', 'ticket len bin', 'cabin len bin']
colunas_dummies = ['Sex', 'age bin', 'fare bin', 'title', 'Embarked', 'ticket left', 'cabin left']
#colunas_dummies = ['Sex', 'age bin', 'fare bin', 'title', 'Embarked']

x_train = pd.get_dummies(x_train, columns=colunas_dummies)
x_test = pd.get_dummies(x_test, columns=colunas_dummies)

for col in x_train.columns:
    if col not in x_test.columns:
        x_test[col] = 0
for col in x_test.columns:
    if col not in x_train.columns:
        x_train[col] = 0

col_ord = x_train.columns
x_test = x_test[col_ord]

x_train.head()
#x_test.head()

#x_train.head()
#x_train.info()
#x_train.describe()

#x_test.head()
#x_train.info()
#x_train.describe()
x_test.columns
metrica = 'accuracy'
params = {
            'learning_rate': np.arange(0.0001, 0.01, 0.00001),
            'n_estimators':[1000, 2000],
            }
otm = RandomizedSearchCV(GradientBoostingClassifier(random_state=0), n_iter=100, param_distributions=params, scoring=metrica, cv=3, n_jobs=-1)
otm.fit(x_train, y_train)
clf = otm.best_estimator_.fit(x_train, y_train)
#otm.best_score_
#otm.best_params_

pred = clf.predict(x_test)
submission = pd.DataFrame({'PassengerId':x_test.index,'Survived':pred})
submission.to_csv('submission.csv', index=False)
#submission.head()
