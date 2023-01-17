import numpy as np

import pandas as pd

from sklearn import *

import xgboost as xgb



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

train.shape, test.shape, sub.shape
#Additional feature columns for image or sentiment analysis, dropping for now

train.drop(columns=['imdb_id', 'homepage', 'poster_path'], inplace=True)

test.drop(columns=['imdb_id', 'homepage', 'poster_path'], inplace=True)
def date_features(df):

    df['release_date'] = pd.to_datetime(df['release_date'])

    df['release_year'] = df['release_date'].dt.year

    df['release_month'] = df['release_date'].dt.month

    df['release_quarter'] = df['release_date'].dt.quarter

    df['release_dow'] = df['release_date'].dt.dayofweek

    df.drop(columns=['release_date'], inplace=True)

    return df



train = date_features(train)

test = date_features(test)
def get_dictionary(s):

    try:

        d = eval(s)

    except:

        d = {}

    return d
#could test dummies but going with simple encoding

train.belongs_to_collection = train.belongs_to_collection.map(lambda x: len(get_dictionary(x))).clip(0,1)

test.belongs_to_collection = test.belongs_to_collection.map(lambda x: len(get_dictionary(x))).clip(0,1)



#pd.concat((train, train.genres.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).str.get_dummies(sep=',')), axis=1) #need to concat train/test first

train.genres = train.genres.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

test.genres = test.genres.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



train.production_companies = train.production_companies.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

test.production_companies = test.production_companies.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



train.production_countries = train.production_countries.map(lambda x: sorted([d['iso_3166_1'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

test.production_countries = test.production_countries.map(lambda x: sorted([d['iso_3166_1'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



train.spoken_languages = train.spoken_languages.map(lambda x: sorted([d['iso_639_1'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

test.spoken_languages= test.spoken_languages.map(lambda x: sorted([d['iso_639_1'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



train.Keywords = train.Keywords.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))

test.Keywords = test.Keywords.map(lambda x: sorted([d['id'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))



for c in ['original_language', 'status', 'genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords']:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(train[c].fillna('').astype(str)) + list(test[c].fillna('').astype(str)))

    train[c] = lbl.transform(train[c].fillna(''))

    test[c] = lbl.transform(test[c].fillna(''))

    print(c, len(lbl.classes_))
#dropping cast and crew but probably great graph features along with above dictionary items - analysis for another time

train.drop(columns=['cast', 'crew'], inplace=True)

test.drop(columns=['cast', 'crew'], inplace=True)
#dropping text features for now

def standard_text_features(df):

    for c in ['original_title', 'title', 'tagline', 'overview']:

        df[c + '_len'] = df[c].map(lambda x: len(str(x)))

        df[c + '_wlen'] = df[c].map(lambda x: len(str(x).split(' ')))

    df.drop(columns=['original_title', 'title', 'tagline', 'overview'], inplace=True)

    return df



train = standard_text_features(train)

test = standard_text_features(test)
#https://www.kaggle.com/zero92/tmdb-prediction

train.loc[train['id'] == 16,'revenue'] = 192864         

train.loc[train['id'] == 90,'budget'] = 30000000                  

train.loc[train['id'] == 118,'budget'] = 60000000       

train.loc[train['id'] == 149,'budget'] = 18000000       

train.loc[train['id'] == 313,'revenue'] = 12000000       

train.loc[train['id'] == 451,'revenue'] = 12000000      

train.loc[train['id'] == 464,'budget'] = 20000000       

train.loc[train['id'] == 470,'budget'] = 13000000       

train.loc[train['id'] == 513,'budget'] = 930000         

train.loc[train['id'] == 797,'budget'] = 8000000        

train.loc[train['id'] == 335,'budget'] = 2 

train.loc[train['id'] == 348,'budget'] = 12

train.loc[train['id'] == 470,'budget'] = 13000000 

train.loc[train['id'] == 513,'budget'] = 1100000

train.loc[train['id'] == 640,'budget'] = 6 

train.loc[train['id'] == 696,'budget'] = 1

train.loc[train['id'] == 797,'budget'] = 8000000 

train.loc[train['id'] == 850,'budget'] = 1500000

train.loc[train['id'] == 1199,'budget'] = 5 

train.loc[train['id'] == 1282,'budget'] = 9 

train.loc[train['id'] == 1347,'budget'] = 1

train.loc[train['id'] == 1755,'budget'] = 2

train.loc[train['id'] == 1801,'budget'] = 5

train.loc[train['id'] == 1918,'budget'] = 592 

train.loc[train['id'] == 2033,'budget'] = 4

train.loc[train['id'] == 2118,'budget'] = 344 

train.loc[train['id'] == 2252,'budget'] = 130

train.loc[train['id'] == 2256,'budget'] = 1 

train.loc[train['id'] == 2696,'budget'] = 10000000

train.loc[train['id'] == 819,'budget'] = 90000000       

train.loc[train['id'] == 850,'budget'] = 90000000       

train.loc[train['id'] == 1007,'budget'] = 2              

train.loc[train['id'] == 1112,'budget'] = 7500000       

train.loc[train['id'] == 1131,'budget'] = 4300000        

train.loc[train['id'] == 1359,'budget'] = 10000000       

train.loc[train['id'] == 1542,'budget'] = 15800000             

train.loc[train['id'] == 1570,'budget'] = 15800000       

train.loc[train['id'] == 1571,'budget'] = 4000000        

train.loc[train['id'] == 1714,'budget'] = 46000000       

train.loc[train['id'] == 1721,'budget'] = 17500000       

train.loc[train['id'] == 1865,'revenue'] = 25000000      

train.loc[train['id'] == 1885,'budget'] = 12             

train.loc[train['id'] == 2091,'budget'] = 10             

train.loc[train['id'] == 2268,'budget'] = 17500000       

train.loc[train['id'] == 2491,'budget'] = 6              

train.loc[train['id'] == 2602,'budget'] = 31000000       

train.loc[train['id'] == 2612,'budget'] = 15000000       

train.loc[train['id'] == 2696,'budget'] = 10000000      

train.loc[train['id'] == 2801,'budget'] = 10000000     

train.loc[train['id'] == 335,'budget'] = 2 

train.loc[train['id'] == 348,'budget'] = 12

train.loc[train['id'] == 470,'budget'] = 13000000 

train.loc[train['id'] == 513,'budget'] = 1100000

train.loc[train['id'] == 640,'budget'] = 6 

train.loc[train['id'] == 696,'budget'] = 1

train.loc[train['id'] == 797,'budget'] = 8000000 

train.loc[train['id'] == 850,'budget'] = 1500000

train.loc[train['id'] == 1199,'budget'] = 5 

train.loc[train['id'] == 1282,'budget'] = 9              

train.loc[train['id'] == 1347,'budget'] = 1

train.loc[train['id'] == 1755,'budget'] = 2

train.loc[train['id'] == 1801,'budget'] = 5

train.loc[train['id'] == 1918,'budget'] = 592 

train.loc[train['id'] == 2033,'budget'] = 4

train.loc[train['id'] == 2118,'budget'] = 344 

train.loc[train['id'] == 2252,'budget'] = 130

train.loc[train['id'] == 2256,'budget'] = 1 

train.loc[train['id'] == 2696,'budget'] = 10000000



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
col = [c for c in train.columns if c not in ['id', 'revenue']]



params = {'eta': 0.02, 'objective': 'reg:linear', 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8,  'eval_metric': 'rmse', 'seed': 1, 'silent': True}

x1, x2, y1, y2 = model_selection.train_test_split(train[col].fillna(-1), np.log1p(train['revenue']), test_size=0.2, random_state=1)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

model = xgb.train(params, xgb.DMatrix(x1, y1), 2500,  watchlist, verbose_eval=100, early_stopping_rounds=200)

pred1 = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)
model = ensemble.GradientBoostingRegressor(random_state=4,  loss='ls', learning_rate=0.02, n_estimators=1000,  max_depth=7)



model.fit(x1, y1)

print(np.sqrt(metrics.mean_squared_error(y2, model.predict(x2))))



model.fit(train[col].fillna(-1), np.log1p(train['revenue']))

pred2 = model.predict(test[col].fillna(-1))

test['revenue'] = np.expm1((pred1 * 0.9) + (pred2 * 0.1))

test[['id', 'revenue']].to_csv('submission.csv', index=False)