import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/test.csv')
df.head()
df1 = pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/train.csv')
df1.head()
targets = df1[['pet_id','breed_category','pet_category']]
df1 = df1.drop('breed_category', axis = True)
df1= df1.drop('pet_category', axis = True)
df1.shape
df.shape
df1.loc[18833:18835]
df.loc[0:1]
df.loc[8071:]
test = pd.concat([df1,df])
test['issue_date'] = pd.to_datetime(test['issue_date'])
test['listing_date'] = pd.to_datetime(test['listing_date'])
test['days'] =  (test['listing_date'] - test['issue_date']).dt.days
test['days'].max()
test['days']=test['days'].mask(test['days']<0).fillna(test['days'].mean())
test['month'] = test['days']/30
test['month'].min()
test[18833:18836]
test['condition'].fillna(3.0, inplace=True)
def encode_and_bind(original_dataframe, feature_to_encode):

    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])

    res = pd.concat([original_dataframe, dummies], axis=1)

    return(res)
test = encode_and_bind(test, 'color_type')
test.drop('color_type',axis = 1, inplace = True)
test['height(m)']=test['height(cm)'].apply(lambda x:x*(.01))
test
test['height(m)']=test['height(m)'].mask(test['height(m)']==0).fillna(test['height(m)'].mean())
test['length(m)']=test['length(m)'].mask(test['length(m)']==0).fillna(test['length(m)'].mean())
test['X1']=test['X1'].mask(test['X1']==0).fillna(test['X1'].median())
test['X2']=test['X2'].mask(test['X2']==0).fillna(test['X2'].median())
test
test.drop('days',axis = 1, inplace = True)
train_df=test[:18834]
test_df= test[18834:]
train_df.shape
test_df.shape
train_df = pd.concat([train_df,targets], axis =1)
train_df.head()
test_df.head()
train_df.drop('height(cm)', axis = 1, inplace = True)
test_df.drop('height(cm)', axis = 1, inplace = True)
train_df.columns.values
train_df

train_df.drop('issue_date', axis = 1, inplace = True)

train_df.drop('listing_date', axis = 1, inplace=True)

test_df.drop('issue_date', axis = 1, inplace = True)

test_df.drop('listing_date', axis = 1, inplace=True)
train_df
train_df=train_df.drop(df.columns[0], axis=1)
train_df.loc[train_df['length(m)'] == 0]
train_df.loc[83]
init_train_df = train_df.drop('pet_category', axis = 1)
init_train_df
sec_train_df = train_df.drop('breed_category', axis = 1)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from sklearn.metrics import f1_score
X1_train = init_train_df.drop('breed_category', axis = 1)

Y1_test = init_train_df['breed_category']
X1_train
params = {

        'min_child_weight': [1,2,3,4 ,5,6,7,8,9,10,12],

        'gamma': [0.5, 1, 1.5, 2, 5, 6],

        'subsample': [0.6, 0.8, 1.0,0.7],

        'colsample_bytree': [0.6, 0.8, 1.0, 0.7],

        'max_depth': [1,2,3, 4, 5,6],

        

        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=700, objective = 'multi:softmax',

                    num_class=3, nthread=1)
folds = 5

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 2001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, n_jobs=-1, cv=skf.split(X1_train,Y1_test), verbose=3, random_state=2001 )

random_search.fit(X1_train,Y1_test)
X2_train = sec_train_df.drop('pet_category', axis = 1)

Y2_test = sec_train_df['pet_category']
params = {

        'min_child_weight': [1,2,3,4 ,5,6,7,8,9,10,12],

        'gamma': [0.5, 1, 1.5, 2, 5, 6],

        'subsample': [0.6, 0.8, 1.0,0.7],

        'colsample_bytree': [0.6, 0.8, 1.0, 0.7],

        'max_depth': [1,2,3, 4, 5,6],

        

        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=700, objective = 'multi:softmax',

                    num_class=4, nthread=1)
folds = 5

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 2001)



random_search_1 = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, n_jobs=-1, cv=skf.split(X2_train,Y2_test), verbose=3, random_state=2001 )

random_search_1.fit(X2_train,Y2_test)
test_df.head()
test1 = test_df.drop('pet_id', axis =1)
test1
y_test = random_search_1.predict(test1)

y_test.shape
test2 = test_df.drop('pet_id', axis =1)
y_test1 = random_search.predict(test1)
y_test1
results_df = pd.DataFrame(data={'pet_id':test_df['pet_id'], 'breed_category':y_test1,'pet_category':y_test})

results_df.to_csv('submission_shuvo.csv', index=False)