import pandas as pd

import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import random 

random.seed(50)
input_file = '../input/titanic3.csv'

data = pd.read_csv(input_file)
data['PassengerId'] = data.index+1
data.head()
data.describe()
numeric_features = ['age','fare']

categorical_features = ['embarked','sex','pclass']
data['embarked'].value_counts()
numeric_pipeline = Pipeline(steps=[('miss', SimpleImputer(strategy='mean')),('scale', StandardScaler())])

category_pipeline = Pipeline(steps=[('miss',SimpleImputer(strategy='constant',fill_value='missing')),('hot',OneHotEncoder(handle_unknown='ignore'))])
preprocess = ColumnTransformer(transformers = [

    ('numeric', numeric_pipeline,numeric_features),

    ('categorical', category_pipeline,categorical_features)

])
model_pipeline = Pipeline(steps=[

    ('prep', preprocess),

    ('model',LogisticRegression(solver='lbfgs'))

])
x = data.drop('survived', axis=1)

y = data['survived']
x_train = x.head(621)

y_train = y.head(621)

x_test = x.tail(418)

y_test = y.tail(418)
y_test.shape
model_pipeline.fit(x_train,y_train)

model_pipeline.score(x_test,y_test)
model_pipeline = Pipeline(steps=[

    ('prep', preprocess),

    ('model', RandomForestClassifier())

])
model_pipeline.fit(x_train,y_train)

model_pipeline.score(x_test,y_test)
df = data.tail(418)

df = df.drop(['pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked','boat','body','home.dest','survived'], axis=1)
df['Survived'] = model_pipeline.predict(x_test)

df.tail()
df.to_csv('titanic.csv',index=False)