import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# Vamos iniciar o notebook importanto o Dataset

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# Podemos observar as primeiras linhas dele.

titanic_df.head()
age_median = titanic_df['Age'].median()

print(age_median)
titanic_df['Age'] = titanic_df['Age'].fillna(age_median)

test_df['Age'] = test_df['Age'].fillna(age_median)
from sklearn.preprocessing import LabelEncoder

sex_encoder = LabelEncoder()



sex_encoder.fit(list(titanic_df['Sex'].values) + list(test_df['Sex'].values))
sex_encoder.classes_
titanic_df['Sex'] = sex_encoder.transform(titanic_df['Sex'].values)

test_df['Sex'] = sex_encoder.transform(test_df['Sex'].values)

test_df['Sex'][:10]
titanic_df.head()['Name']
import re

def extract_title(name):

    x = re.search(', (.+)\.', name)

    if x : #and not 'Elizabeth' in x.group(1):

        return x.group(1)

    else:

        return ''
titanic_df['Name'].apply(extract_title).unique()
titanic_df['Title'] = titanic_df['Name'].apply(extract_title)

test_df['Title'] = test_df['Name'].apply(extract_title)
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction import DictVectorizer



feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title']

dv = DictVectorizer()

dv.fit(titanic_df[feature_names].append(test_df[feature_names]).to_dict(orient='records'))

dv.feature_names_
from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(dv.transform(titanic_df[feature_names].to_dict(orient='records')),

                                                     titanic_df['Survived'],

                                                     test_size=0.2,

                                                     random_state=42)
import xgboost as xgb
train_X.todense()
dtrain = xgb.DMatrix(data=train_X.todense(), feature_names=dv.feature_names_, label=train_y)

dvalid = xgb.DMatrix(data=valid_X.todense(), feature_names=dv.feature_names_, label=valid_y)
xgb_clf = xgb.train({'max_depth':20, 'eta':0.1, 'objective':'binary:logistic', 'eval_metric': 'error'}, 

                    num_boost_round=3000,

                    dtrain=dtrain,

                    verbose_eval=True, 

                    early_stopping_rounds=30,

                    evals=[(dtrain, 'train'), (dvalid, 'valid')])
test_df['Fare'] = test_df['Fare'].fillna(0)
test_X = dv.transform(test_df[feature_names].to_dict(orient='records'))

print(test_X.shape)
dtest = xgb.DMatrix(data=test_X.todense(), feature_names=dv.feature_names_)
y_pred = np.round(xgb_clf.predict(dtest)).astype(int)
submission_df = pd.DataFrame()
submission_df['PassengerId'] = test_df['PassengerId']

submission_df['Survived'] = y_pred

submission_df
submission_df.to_csv('xgboost_model_1.csv', index=False)