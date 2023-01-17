import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,roc_auc_score



def cleanPeople(people):

    

    people = people.drop(['date'],axis=1)

    people['people_id'] = people['people_id'].apply(lambda x : x.split('_')[1])

    people['people_id'] = pd.to_numeric(people['people_id']).astype(int)

    

    fields = list(people.columns)

    cat_data = fields[1:11]

    bool_data = fields[11:]

    

    for data in cat_data:

        people[data] = people[data].fillna('type 0')

        people[data] = people[data].apply(lambda x: x.split(' ')[1])

        people[data] = pd.to_numeric(people[data]).astype(int)

    

    for data in bool_data:

        people[data] = pd.to_numeric(people[data]).astype(int)

        

    

    return people



def cleanAct(data, train=False):

    

    data = data.drop(['date'],axis = 1)

    if train:

        data = data.drop(['outcome'],axis=1)

        

    data['people_id'] = data['people_id'].apply(lambda x : x.split('_')[1])

    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)

    

    data['activity_id'] = data['activity_id'].apply(lambda x: x.split('_')[1])

    data['activity_id'] = pd.to_numeric(data['activity_id']).astype(int)

    

    fields = list(data.columns)

    cat_data = fields[2:13]

    

    for column in cat_data:

        data[column] = data[column].fillna('type 0')

        data[column] = data[column].apply(lambda x : x.split(' ')[1])

        data[column] = pd.to_numeric(data[column]).astype(int)

     

    return data    
people = pd.read_csv("../input/people.csv")

people = cleanPeople(people)



act_train = pd.read_csv("../input/act_train.csv",parse_dates=['date'])

act_train_cleaned = cleanAct(act_train,train=True)



act_test = pd.read_csv("../input/act_test.csv",parse_dates=['date'])

act_test_cleaned = cleanAct(act_test)





train = act_train_cleaned.merge(people,on='people_id', how='left')

test = act_test_cleaned.merge(people, on='people_id', how='left')
train = pd.get_dummies(train,columns=['activity_category'],sparse=True,drop_first=True)

test = pd.get_dummies(test,columns=['activity_category'],sparse=True,drop_first=True)
from sklearn.cross_validation import LabelKFold

# LabelKFold is a sklearn function that creates train/validation folds over the data

# The special thing about is that it will split in a way that the same label never appears in

# both train and test.

# This means that we can use it to split the training set based on people and get good validation.

# Here I am making KFolds and then selecting the first one.

train_mask, valid_mask = list(LabelKFold(train['people_id'], n_folds=10))[0]



x_test = test.drop(['people_id','activity_id'],axis=1)

y = act_train['outcome']

train = train.drop(['people_id', 'activity_id'], axis=1)



kklo=x_train = np.array(train)[train_mask]

y_train = np.array(y)[train_mask]



x_valid = np.array(train)[valid_mask]

y_valid = np.array(y)[valid_mask]



#print(x_train.shape)

#print(x_valid.shape)



# Parameters for XGBoost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'auc'

params['eta'] = 0.1 # Learning rate, lower is usually better but takes longer

params['max_depth'] = 10

params['subsample'] = 0.9

params['colsample_bytree'] = 0.9



# Convert to XGBoost DMatrix format

d_train = xgb.DMatrix(x_train, label=y_train, missing=np.nan)

d_valid = xgb.DMatrix(x_valid, label=y_valid, missing=np.nan)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 600, watchlist, early_stopping_rounds=40)

clf.save_model('mymodel.model')
clf = xgb.train(params, d_train, 800, watchlist, xgb_model ='mymodel.model', early_stopping_rounds=40)
p_test = clf.predict(xgb.DMatrix(np.array(x_test)))



sub = pd.DataFrame()

sub['activity_id'] = act_test['activity_id']

sub['outcome'] = p_test

sub.to_csv('submission.csv', index=False)
