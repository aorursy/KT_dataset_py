## importing the necessary packages

import pandas as pd

import ast

from collections import defaultdict

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats

import datetime as dt

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
## Fixing values that are missing/incorrect for the train and test sets

## These are given pre-release so there is no leakage in the data

## Thanks to the discussion forums for this information

train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning

train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          

train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs

train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven

train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 

train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty

train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood

train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II

train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada

train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol

train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip

train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times

train.loc[train['id'] == 1007,'budget'] = 2              # Zyzzyx Road 

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 1885,'budget'] = 12             # In the Cut

train.loc[train['id'] == 2091,'budget'] = 10             # Deadfall

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'budget'] = 6              # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture

train.loc[train['id'] == 335,'budget'] = 2 

train.loc[train['id'] == 348,'budget'] = 12

train.loc[train['id'] == 470,'budget'] = 13000000 

train.loc[train['id'] == 513,'budget'] = 1100000

train.loc[train['id'] == 640,'budget'] = 6 

train.loc[train['id'] == 696,'budget'] = 1

train.loc[train['id'] == 797,'budget'] = 8000000 

train.loc[train['id'] == 850,'budget'] = 1500000

train.loc[train['id'] == 1199,'budget'] = 5 

train.loc[train['id'] == 1282,'budget'] = 9               # Death at a Funeral

train.loc[train['id'] == 1347,'budget'] = 1

train.loc[train['id'] == 1755,'budget'] = 2

train.loc[train['id'] == 1801,'budget'] = 5

train.loc[train['id'] == 1918,'budget'] = 592 

train.loc[train['id'] == 2033,'budget'] = 4

train.loc[train['id'] == 2118,'budget'] = 344 

train.loc[train['id'] == 2252,'budget'] = 130

train.loc[train['id'] == 2256,'budget'] = 1 

train.loc[train['id'] == 2696,'budget'] = 10000000





test.loc[test['id'] == 6733,'budget'] = 5000000

test.loc[test['id'] == 3889,'budget'] = 15000000

test.loc[test['id'] == 6683,'budget'] = 50000000

test.loc[test['id'] == 5704,'budget'] = 4300000

test.loc[test['id'] == 6109,'budget'] = 281756

test.loc[test['id'] == 7242,'budget'] = 10000000

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee

test.loc[test['id'] == 3033,'budget'] = 250 

test.loc[test['id'] == 3051,'budget'] = 50

test.loc[test['id'] == 3084,'budget'] = 337

test.loc[test['id'] == 3224,'budget'] = 4  

test.loc[test['id'] == 3594,'budget'] = 25  

test.loc[test['id'] == 3619,'budget'] = 500  

test.loc[test['id'] == 3831,'budget'] = 3  

test.loc[test['id'] == 3935,'budget'] = 500  

test.loc[test['id'] == 4049,'budget'] = 995946 

test.loc[test['id'] == 4424,'budget'] = 3  

test.loc[test['id'] == 4460,'budget'] = 8  

test.loc[test['id'] == 4555,'budget'] = 1200000 

test.loc[test['id'] == 4624,'budget'] = 30 

test.loc[test['id'] == 4645,'budget'] = 500 

test.loc[test['id'] == 4709,'budget'] = 450 

test.loc[test['id'] == 4839,'budget'] = 7

test.loc[test['id'] == 3125,'budget'] = 25 

test.loc[test['id'] == 3142,'budget'] = 1

test.loc[test['id'] == 3201,'budget'] = 450

test.loc[test['id'] == 3222,'budget'] = 6

test.loc[test['id'] == 3545,'budget'] = 38

test.loc[test['id'] == 3670,'budget'] = 18

test.loc[test['id'] == 3792,'budget'] = 19

test.loc[test['id'] == 3881,'budget'] = 7

test.loc[test['id'] == 3969,'budget'] = 400

test.loc[test['id'] == 4196,'budget'] = 6

test.loc[test['id'] == 4221,'budget'] = 11

test.loc[test['id'] == 4222,'budget'] = 500

test.loc[test['id'] == 4285,'budget'] = 11

test.loc[test['id'] == 4319,'budget'] = 1

test.loc[test['id'] == 4639,'budget'] = 10

test.loc[test['id'] == 4719,'budget'] = 45

test.loc[test['id'] == 4822,'budget'] = 22

test.loc[test['id'] == 4829,'budget'] = 20

test.loc[test['id'] == 4969,'budget'] = 20

test.loc[test['id'] == 5021,'budget'] = 40 

test.loc[test['id'] == 5035,'budget'] = 1 

test.loc[test['id'] == 5063,'budget'] = 14 

test.loc[test['id'] == 5119,'budget'] = 2 

test.loc[test['id'] == 5214,'budget'] = 30 

test.loc[test['id'] == 5221,'budget'] = 50 

test.loc[test['id'] == 4903,'budget'] = 15

test.loc[test['id'] == 4983,'budget'] = 3

test.loc[test['id'] == 5102,'budget'] = 28

test.loc[test['id'] == 5217,'budget'] = 75

test.loc[test['id'] == 5224,'budget'] = 3 

test.loc[test['id'] == 5469,'budget'] = 20 

test.loc[test['id'] == 5840,'budget'] = 1 

test.loc[test['id'] == 5960,'budget'] = 30

test.loc[test['id'] == 6506,'budget'] = 11 

test.loc[test['id'] == 6553,'budget'] = 280

test.loc[test['id'] == 6561,'budget'] = 7

test.loc[test['id'] == 6582,'budget'] = 218

test.loc[test['id'] == 6638,'budget'] = 5

test.loc[test['id'] == 6749,'budget'] = 8 

test.loc[test['id'] == 6759,'budget'] = 50 

test.loc[test['id'] == 6856,'budget'] = 10

test.loc[test['id'] == 6858,'budget'] =  100

test.loc[test['id'] == 6876,'budget'] =  250

test.loc[test['id'] == 6972,'budget'] = 1

test.loc[test['id'] == 7079,'budget'] = 8000000

test.loc[test['id'] == 7150,'budget'] = 118

test.loc[test['id'] == 6506,'budget'] = 118

test.loc[test['id'] == 7225,'budget'] = 6

test.loc[test['id'] == 7231,'budget'] = 85

test.loc[test['id'] == 5222,'budget'] = 5

test.loc[test['id'] == 5322,'budget'] = 90

test.loc[test['id'] == 5350,'budget'] = 70

test.loc[test['id'] == 5378,'budget'] = 10

test.loc[test['id'] == 5545,'budget'] = 80

test.loc[test['id'] == 5810,'budget'] = 8

test.loc[test['id'] == 5926,'budget'] = 300

test.loc[test['id'] == 5927,'budget'] = 4

test.loc[test['id'] == 5986,'budget'] = 1

test.loc[test['id'] == 6053,'budget'] = 20

test.loc[test['id'] == 6104,'budget'] = 1

test.loc[test['id'] == 6130,'budget'] = 30

test.loc[test['id'] == 6301,'budget'] = 150

test.loc[test['id'] == 6276,'budget'] = 100

test.loc[test['id'] == 6473,'budget'] = 100

test.loc[test['id'] == 6842,'budget'] = 30
train.head()

test.head()
# getting the number of NA values

# test na: homepage 2978, overview 14, poster_path 1, release_date 1, runtime 4, status 2, 

# tagline 863, title 3, 

print("Test NA: ",test.isna().sum())

# train na: homepage: 2054, overview 8, poster_path 1, runtime 2, tagline 597

print("\nTrain NA: ",train.isna().sum())
# certain columns have issues with dictionaries coming through, so those need to

# be adjusted

issue_cols = ['belongs_to_collection', 'genres','production_companies','production_countries',

              'spoken_languages','Keywords','cast','crew']

def fix_cols(df, cols):

    for col in cols:

        df[col] = df[col].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))

    return df



train = fix_cols(train, issue_cols)

test = fix_cols(test, issue_cols)
## adjusting the collection columns

train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x!={} else '')

test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x!={} else '')

train['has_collection'] = train['collection_name'].apply(lambda x: 0 if '' else 1)

test['has_collection'] = test['collection_name'].apply(lambda x: 0 if '' else 1)



train = train.drop(['belongs_to_collection'], axis=1)

test = test.drop(['belongs_to_collection'], axis=1)
# fixing most of the columns at once

def fixing_columns(train, test, original_col, count_col, num_to_return):

    train[original_col] = train[original_col].apply(lambda x: sorted(i['name'] for i in x) if x != {} else '')

    train[count_col] = train[original_col].apply(lambda x: len(x))

    

    col_list = list(train[original_col].apply(lambda x: [i for i in x]))

    col_counts = defaultdict(int)

    for row in col_list:

        for i in row:

            col_counts[i] += 1

        

    top = [i[0] for i in sorted(col_counts.items(), key=lambda kv: kv[1], reverse=True)[:num_to_return]]

    

    # One hot encoding if the production company is in the top 20 and exists in the record

    for top_var in list(top):

        train[top_var] = train[original_col].apply(lambda x: 1 if top_var in x else 0)

    

    # doing the same for test

    # fixing the production_companies column and adding a pc count column

    test[original_col] = test[original_col].apply(lambda x: sorted(i['name'] for i in x) if x != {} else '')

    test[count_col] = test[original_col].apply(lambda x: len(x))

    

    # One hot encoding if the production company is in the top 20 and exists in the record

    for top_var in list(top):

        test[top_var] = test[original_col].apply(lambda x: 1 if top_var in x else 0)
# how many top values to keep for each column

number_to_keep = [20, 20, 5, 3, 10, 10, 10]

issue_cols = ['genres','production_companies','production_countries',

              'spoken_languages','Keywords','cast','crew']

for i in range(len(issue_cols)):

    fixing_columns(train, test, issue_cols[i],'num_'+issue_cols[i],number_to_keep[i])
# dropping the unnecessary columns

drop_columns = ['genres','homepage','imdb_id','original_title','overview','poster_path',

                'production_companies','production_countries','spoken_languages',

                'status','tagline','title','Keywords','cast','crew']

train = train.drop(drop_columns, axis=1)

test = test.drop(drop_columns, axis=1)
# removing the N/A values and just replacing them with 0

print("Test NA: ",test.columns[test.isna().sum()>0].tolist())

print("\nTrain NA: ",train.columns[train.isna().sum()>0].tolist())



train[train.isna().any(axis=1)]   

train[train.isna().any(axis=1)]    



# using 0 to fill the na columns

train.fillna(0, inplace=True)

test.fillna(0, inplace=True)

    
# adjusting the date values

def date_adjust(df):

    # getting the release date into proper format

    df[['release_month', 'release_day','release_year']] = df['release_date'].str.split('/',expand=True).replace(np.nan,0).astype(int)

    

    #df.loc[df['release_year'] <=18, 'release_year'] += 2000

    #df.loc[df['release_year'] > 18, 'release_year'] += 1900

    releaseDate = pd.to_datetime(df['release_date'])

    df['release_week'] = releaseDate.dt.week

    df['release_dayofweek'] = releaseDate.dt.dayofweek

    df['release_quarter'] = releaseDate.dt.quarter

    df['original_english'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)

    df = df.drop(['collection_name','original_language','release_date'], axis=1)





date_adjust(train)

date_adjust(test)



train = train.drop(['collection_name','original_language','release_date','id'], axis=1)

test = test.drop(['collection_name','original_language','release_date','id'], axis=1)
# creating some new features based on combinations of high performing ones



def feature_transform(df):

    df['budget_runtime'] = (df['budget'] + 1)/(df['runtime']+1)

    df['poularity_year'] = (df['popularity']+1)/(df['release_year']+1)

    df['budget_popularity'] = (df['budget']+1)/(df['popularity']+1)

    df['budget_year'] = (df['budget'] + 1)/(df['release_year']+1)

    df['runtime_year'] = (df['runtime']+1)/(df['release_year']+1)



feature_transform(train)

feature_transform(test)
train['budget'] = train['budget'].apply(np.log1p)

test['budget'] = test['budget'].apply(np.log1p)

X = np.asarray(train.drop(['revenue'], axis=1))

y = np.asarray(train.revenue.apply(np.log1p))

# training and running an xgboost model

params = {'objective':'reg:linear',

          'eta':0.01,

          'max_depth':6,

          'min_child_weight':3,

          'subsample':0.8,

          'colsample_bytree':0.8,

          'colsample_bylevel':.5,

          'gamma':1.45,

          'eval_metric':'rmse',

          'seed':1,

          'silent':True}



def train_xgb(X_train, y_train, X_test, y_test):

    xgb_data = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_test, y_test),'valid')]

    xgb_model = xgb.train(params, xgb.DMatrix(X_train,y_train),

          5000, xgb_data,

          verbose_eval=100,

          early_stopping_rounds=500)

    return xgb_model



error = []

kf = list(KFold(n_splits=10, random_state=1, shuffle=True).split(X))

for i, (train_index, test_index) in enumerate(kf):

    X_train, X_test =X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    xgb_model = train_xgb(X_train, y_train, X_test, y_test)

    error.append(xgb_model.best_score)

    

print("Error: {} +/- {}".format(np.mean(error), np.std(error)))

  
# plotting the feature importance of the last fold

xgb_model.feature_names = train.columns.tolist()

fig, ax = plt.subplots(figsize=(20,10))

xgb.plot_importance(xgb_model, max_num_features=30,ax=ax, height=.9)

plt.show()
# fitting on the full training set

xgb_full_train = xgb.XGBRegressor(objective = 'reg:linear',

                                  eta = 0.01,

                                  max_depth = 6,

                                  min_child_weight = 3,

                                  subsample = 0.8,

                                  colsample_bytree = 0.7,

                                  eval_metric = 'rmse',

                                  seed = 1,

                                  n_estimators = 2800)





xgb_full_train.fit(X, y)

y_predictions = xgb_full_train.predict(test.values)
submission = pd.read_csv("../input/sample_submission.csv")

submission['revenue'] = np.exp(y_predictions)





submission.to_csv('xgb_sub.csv',index=False)