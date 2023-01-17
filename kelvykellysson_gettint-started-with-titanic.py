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
train_data=pd.read_csv('../input/titanic/train.csv')

train_data.head()
test_data=pd.read_csv('../input/titanic/test.csv')

test_data.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

features = ["Pclass", "Sex", "SibSp", "Parch"]

y = train_data["Survived"]

X = pd.get_dummies(train_data[features]).fillna(-1)

X_valid, X_train, y_valid, y_train = train_test_split(X,y,test_size=0.5)









from sklearn.ensemble import RandomForestClassifier

v=[]

for n in range(100):

    model = RandomForestClassifier(n_estimators=n+1*10, max_depth=5, random_state=1)

    model.fit(X_train, y_train)



    predictions = model.predict(X_valid)

    score = mean_absolute_error(y_valid, predictions)

    v.append(score)



n_min = min(v)

n_pos = v.index(n_min)



print("Menor valor: %s" % n_min)

print("Posição: %s" % n_pos)
from sklearn.ensemble import RandomForestRegressor

v=[]

for n in range(100):

    model = RandomForestRegressor(n_estimators=n+1*10, random_state=1)

    model.fit(X_train, y_train)



    predictions = model.predict(X_valid)

    score = mean_absolute_error(y_valid, predictions)

    v.append(score)



n_min = min(v)

n_pos = v.index(n_min)

print("Menor valor: %s" % n_min)

print("Posição: %s" % n_pos)
np.mean(predictions==y_valid)
from xgboost import XGBRegressor



v=[]

for n in range(100):

    model = XGBRegressor(n_estimators=500)

    model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)



    predictions = model.predict(X_valid)

    score = mean_absolute_error(y_valid, predictions)

    v.append(score)



n_min = min(v)

n_pos = v.index(n_min)

print("Menor valor: %s" % n_min)

print("Posição: %s" % n_pos)

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1)

model.fit(X_train, y_train)



predictions = model.predict(X_valid)

score = mean_absolute_error(y_valid, predictions)

score
train_data['Embarked_S'] = (train_data['Embarked']=='S').astype(int)

train_data['Embarked_C'] = (train_data['Embarked']=='C').astype(int)

train_data['Embarked_Q'] = (train_data['Embarked']=='Q').astype(int)



train_data['Cabine_nula'] = train_data['Cabin'].isnull().astype(int)



train_data['Nome_contrem_Miss'] = train_data['Name'].str.contains("Miss").astype(int)

train_data['Nome_contrem_Mrs'] = train_data['Name'].str.contains("Mrs").astype(int)



train_data['Nome_contrem_Master'] = train_data['Name'].str.contains("Master").astype(int)

train_data['Nome_contrem_Col'] = train_data['Name'].str.contains("Col").astype(int)

train_data['Nome_contrem_Major'] = train_data['Name'].str.contains("Major").astype(int)

train_data['Nome_contrem_Mr'] = train_data['Name'].str.contains("Mr").astype(int)
features = ['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', "Fare",'Embarked_S', 'Embarked_C', "Embarked_Q", 'Cabine_nula', 'Nome_contrem_Miss', 'Nome_contrem_Mrs','Nome_contrem_Master','Nome_contrem_Col','Nome_contrem_Major','Nome_contrem_Mr']

X = pd.get_dummies(train_data[features]).fillna(-1)

X_valid, X_train, y_valid, y_train = train_test_split(X,y,test_size=0.5)
from sklearn.ensemble import RandomForestClassifier

v=[]

for n in range(100):

    model = RandomForestClassifier(n_estimators=n+1*10, max_depth=5, random_state=1)

    model.fit(X_train, y_train)



    predictions = model.predict(X_valid)

    score = mean_absolute_error(y_valid, predictions)

    v.append(score)



n_min = min(v)

n_pos = v.index(n_min)



print("Menor valor: %s" % n_min)

print("Posição: %s" % n_pos)
from sklearn.ensemble import RandomForestRegressor

v=[]

for n in range(100):

    model = RandomForestRegressor(n_estimators=n+1*10, random_state=1)

    model.fit(X_train, y_train)



    predictions = model.predict(X_valid)

    score = mean_absolute_error(y_valid, predictions)

    v.append(score)



n_min = min(v)

n_pos = v.index(n_min)

print("Menor valor: %s" % n_min)

print("Posição: %s" % n_pos)
from xgboost import XGBRegressor



v=[]

for n in range(100):

    model = XGBRegressor(n_estimators=500)

    model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], verbose=False)



    predictions = model.predict(X_valid)

    score = mean_absolute_error(y_valid, predictions)

    v.append(score)



n_min = min(v)

n_pos = v.index(n_min)

print("Menor valor: %s" % n_min)

print("Posição: %s" % n_pos)
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
train_data['Title'] = train_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

test_data['Title'] = test_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())



train_data['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)

test_data['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)



train_data['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)

test_data['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
train_data['Fam_size'] = train_data['SibSp'] + train_data['Parch'] + 1

test_data['Fam_size'] = test_data['SibSp'] + test_data['Parch'] + 1
numerical_cols = ['Fare']

categorical_cols = ['Pclass', 'Sex', 'Title', 'Embarked', 'Fam_type']



train_data['Fam_type'] = pd.cut(train_data.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

test_data['Fam_type'] = pd.cut(test_data.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])

train_data.head()
numerical_transformer= SimpleImputer(strategy='median')



categorical_transformer=Pipeline(steps=[

    ('Imputer', SimpleImputer(strategy='most_frequent')),

    ('Onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
y = train_data['Survived']

features = ['Pclass', 'Sex', 'Fare', 'Title', 'Embarked', 'Fam_type']

X = train_data[features]

X.head()
titanic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestClassifier(random_state=0, 

                                                               n_estimators=500, max_depth=5))

                             ])



titanic_pipeline.fit(X,y)



print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline, X, y, cv=10).mean()))
test_data['Embarked_S'] = (test_data['Embarked']=='S').astype(int)

test_data['Embarked_C'] = (test_data['Embarked']=='C').astype(int)

test_data['Embarked_Q'] = (test_data['Embarked']=='Q').astype(int)



test_data['Cabine_nula'] = test_data['Cabin'].isnull().astype(int)



test_data['Nome_contrem_Miss'] = test_data['Name'].str.contains("Miss").astype(int)

test_data['Nome_contrem_Mrs'] = test_data['Name'].str.contains("Mrs").astype(int)



test_data['Nome_contrem_Master'] = test_data['Name'].str.contains("Master").astype(int)

test_data['Nome_contrem_Col'] = test_data['Name'].str.contains("Col").astype(int)

test_data['Nome_contrem_Major'] = test_data['Name'].str.contains("Major").astype(int)

test_data['Nome_contrem_Mr'] = test_data['Name'].str.contains("Mr").astype(int)
X_test = test_data[features]

X_test.head()

predictions = titanic_pipeline.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv_new', index=False)