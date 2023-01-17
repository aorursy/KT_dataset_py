import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/homework1-multiclass-classification/train.csv', sep=';')
test = pd.read_csv('/kaggle/input/homework1-multiclass-classification/test.csv', sep=';')
submission = pd.read_csv('/kaggle/input/homework1-multiclass-classification/sub_baseline.csv')
'train: ', train.shape, 'test: ', test.shape
submission.head()
train.columns
train.head()
train.interest_level.value_counts()
train.interest_level.value_counts(normalize=True)
train.features
train['features']=train['features'].str.replace('[\[\]\']', '').str.split(', ')
train['features']
train['num_features']=train['features'].apply(len)
train['photos']=train['photos'].str.replace('[\[\]\']', '').str.split(', ')
train['num_photos']=train['photos'].apply(len)
train["num_description_words"]=train["description"].fillna('').apply(lambda x: len(x.split(" ")))
train["created"]
train["created"] = pd.to_datetime(train["created"])
train["created_year"] = train["created"].dt.year
train["created_month"] = train["created"].dt.month
train["created_day"] = train["created"].dt.day
train["created_hour"] = train["created"].dt.hour
hours=train["created_hour"].value_counts().reset_index().sort_values(by='index')
hours.plot(x='index', y='created_hour', kind='bar')
months=train["created_month"].value_counts().reset_index().sort_values(by='index')
months.plot(x='index', y='created_month', kind='bar')
features=['price', 'bathrooms',
          'bedrooms', "num_photos", "num_features",
          "num_description_words", "created_hour"]
target='interest_level'
train=train.set_index('listing_id')
train_target=train[target]
X_train, X_test, y_train, y_test=train_test_split(train[features], train_target,
                                                  test_size=0.3, stratify=train_target, 
                                                  random_state=42)
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
model = LogisticRegression()
model.fit(X_train, y_train)
y_predicts=model.predict(X_test)
f1_score(y_test ,y_predicts, average='macro')
test['features']=test['features'].str.replace('[\[\]\']', '').str.split(', ')
test['num_features']=test['features'].apply(len)
test['photos']=test['photos'].str.replace('[\[\]\']', '').str.split(', ')
test['num_photos']=test['photos'].apply(len)
test["num_description_words"]=test["description"].fillna('').apply(lambda x: len(x.split(" ")))
test["created"] = pd.to_datetime(test["created"])
test["created_year"] = test["created"].dt.year
test["created_month"] = test["created"].dt.month
test["created_day"] = test["created"].dt.day
test["created_hour"] = test["created"].dt.hour
test[target]=model.predict(test[features])
mapper={
        'low':0,
        'medium':1,
        'high':2
       }
test[target]=test[target].apply(lambda x: mapper[x])
test[['listing_id', target]].to_csv('sub_baseline.csv', index=None)