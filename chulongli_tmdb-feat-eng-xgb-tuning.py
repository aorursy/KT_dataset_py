import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import ast

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import scipy



from sklearn import svm, datasets

from sklearn.model_selection import cross_validate



from sklearn.model_selection import GridSearchCV



import xgboost as xgb

from sklearn import metrics   #Additional scklearn functions



import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

#rcParams['figure.figsize'] = 12, 4
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full_data = [train, test]



# Convert string into dict

# 'belongs_to_collection'

dict_columns = ['genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

for dataset in full_data:

    for column in dict_columns:

        dataset[column] = dataset[column].apply(lambda x: [] if x!=x else ast.literal_eval(x)) # x!=x means x is nan
train.head(5)
train.isnull().sum()
for index, row in train['belongs_to_collection'][0:5].iteritems():

    print(index, row)
train['belongs_to_collection'].isnull().value_counts()
for dataset in full_data:

    dataset['HasCollection'] = dataset['belongs_to_collection'].apply(lambda x: 1 if x!=x else 0)

train.drop('belongs_to_collection', axis=1, inplace = True)

test.drop('belongs_to_collection', axis=1, inplace = True)

train.head()
for index, row in train['genres'][0:5].iteritems():

    print(index, row)

#train[train['genres'].isna()]

train['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()
genre_list = train['genres'].apply(lambda x: [i['name'] for i in x])

genre_rank = pd.Series(genre_list.sum()).value_counts()

print(genre_rank)



for dataset in full_data:

    dataset['genres'] = dataset['genres'].apply(lambda x: [i['name'] for i in x])

    dataset['num_genres'] = dataset['genres'].apply(lambda x: len(x))

    for i in genre_rank.keys():

        dataset[i] = dataset['genres'].apply(lambda x: 1 if i in x else 0)

    dataset["all_genres"] = dataset['genres'].apply(lambda x: "_".join(x))



all_genres_rank = train["all_genres"].value_counts()[0:30]



for dataset in full_data:

    for i in all_genres_rank.keys():

        dataset['all_genres_'+i] = dataset['all_genres'].apply(lambda x: 1 if i in x else 0)

train.head()
for index, row in train['production_companies'][0:5].iteritems():

    print(index, row)

company_list = train['production_companies'].apply(lambda x: [i['name'] for i in x])

company_rank = pd.Series(company_list.sum()).value_counts()

print(company_rank[company_rank>30])

for dataset in full_data:

    dataset['production_companies'] = dataset['production_companies'].apply(lambda x: [i['name'] for i in x])

    dataset['num_companies'] = dataset['production_companies'].apply(lambda x:len(x))

train.head(3)
for dataset in full_data:

    for i in company_rank.keys()[0:30]:

        dataset[i] = dataset['production_companies'].apply(lambda x: 1 if i in x else 0)

train.head(3)
for index, row in train.production_countries[0:5].iteritems():

    print(index, row)

country_list = train['production_countries'].apply(lambda x: [i['name'] for i in x])

country_rank = pd.Series(country_list.sum()).value_counts()

print(country_rank[0:15])



for dataset in full_data:

    dataset['production_countries'] = dataset['production_countries'].apply(lambda x: [i['name'] for i in x])

    dataset['num_countries'] = dataset['production_countries'].apply(lambda x:len(x))

    for i in country_rank.keys()[0:30]:

        dataset[i] = dataset['production_countries'].apply(lambda x: 1 if i in x else 0)
for index, row in train.spoken_languages[0:5].iteritems():

    print(index, row)



langauge_list = train['spoken_languages'].apply(lambda x: [i['iso_639_1'] for i in x])

langauge_rank = pd.Series(langauge_list.sum()).value_counts()

print(langauge_rank[0:15])

for dataset in full_data:

    dataset['spoken_languages'] = dataset['spoken_languages'].apply(lambda x: [i['iso_639_1'] for i in x])

    dataset['num_languages'] = dataset['spoken_languages'].apply(lambda x:len(x))

    for i in langauge_rank.keys()[0:30]:

        dataset[i] = dataset['spoken_languages'].apply(lambda x: 1 if i in x else 0)

langauge_rank
for index, row in train['Keywords'][0:5].iteritems():

    print(index, row)

keyword_list = train['Keywords'].apply(lambda x: [i['name'] for i in x])

keyword_rank = pd.Series(keyword_list.sum()).value_counts()

print(keyword_rank[0:15])



for dataset in full_data:

    dataset['Keywords'] = dataset['Keywords'].apply(lambda x: [i['name'] for i in x])

    dataset['num_keywords'] = dataset['Keywords'].apply(lambda x:len(x))

    for i in keyword_rank.keys()[0:30]:

        dataset[i] = dataset['Keywords'].apply(lambda x: 1 if i in x else 0)
#for index, row in train['cast'][0:1].iteritems():

#    print(index, row)



cast_name_list = train['cast'].apply(lambda x: [i['name'] for i in x])

cast_name_rank = pd.Series(cast_name_list.sum()).value_counts()

print(cast_name_rank[0:30])



cast_character_list = train['cast'].apply(lambda x: [i['character'] for i in x])

cast_character_rank = pd.Series(cast_character_list.sum()).value_counts()

print(cast_character_rank[0:30])



cast_gender_list = train['cast'].apply(lambda x: [i['gender'] for i in x])

cast_gender_rank = pd.Series(cast_gender_list.sum()).value_counts()

print(cast_gender_rank)



for dataset in full_data:

    dataset['cast_name'] = dataset['cast'].apply(lambda x: [i['name'] for i in x])

    dataset['cast_character'] = dataset['cast'].apply(lambda x: [i['character'] for i in x])

    dataset['cast_gender'] = dataset['cast'].apply(lambda x: [i['gender'] for i in x])

    

    dataset['num_cast'] = dataset['cast'].apply(lambda x:len(x))

    for i in cast_name_rank.keys()[0:30]:

        dataset[i] = dataset['cast_name'].apply(lambda x: 1 if i in x else 0)

    for i in cast_character_rank.keys()[0:30]:

        dataset[i] = dataset['cast_character'].apply(lambda x: 1 if i in x else 0)

    for i in cast_gender_rank.keys():

        dataset[i] = dataset['cast_gender'].apply(lambda x: 1 if i in x else 0)
train.head(2)
### Temporarily ignore this comlumn

#for index, row in train.crew[0:1].iteritems():

#    print(index, row)

crew_name_list = train['crew'].apply(lambda x: [i['name'] for i in x])

crew_name_rank = pd.Series(crew_name_list.sum()).value_counts()

print(crew_name_rank[0:15])



crew_job_list = train['crew'].apply(lambda x: [i['job'] for i in x])

crew_job_rank = pd.Series(crew_job_list.sum()).value_counts()

print(crew_job_rank[0:15])



crew_department_list = train['crew'].apply(lambda x: [i['department'] for i in x])

crew_department_rank = pd.Series(crew_department_list.sum()).value_counts()

print(crew_department_rank[0:15])



crew_gender_list = train['crew'].apply(lambda x: [i['gender'] for i in x])

crew_gender_rank = pd.Series(crew_gender_list.sum()).value_counts()

print(crew_gender_rank)



for dataset in full_data:

    dataset['crew_name'] = dataset['crew'].apply(lambda x: [i['name'] for i in x])

    dataset['crew_job'] = dataset['crew'].apply(lambda x: [i['job'] for i in x])

    dataset['crew_department'] = dataset['crew'].apply(lambda x: [i['department'] for i in x])

    dataset['crew_gender'] = dataset['crew'].apply(lambda x: [i['gender'] for i in x])

   

    dataset['num_crew'] = dataset['crew'].apply(lambda x:len(x))

    for i in crew_name_rank.keys()[0:30]:

        dataset[i] = dataset['crew_name'].apply(lambda x: 1 if i in x else 0)

    for i in crew_job_rank.keys()[0:30]:

        dataset[i] = dataset['crew_job'].apply(lambda x: 1 if i in x else 0)

    for i in crew_department_rank.keys()[0:30]:

        dataset[i] = dataset['crew_department'].apply(lambda x: 1 if i in x else 0)

    for i in crew_gender_rank.keys():

        dataset[i] = dataset['crew_gender'].apply(lambda x: 1 if i in x else 0)

for dataset in full_data:

    dataset['HasHomepage'] = dataset['homepage'].apply(lambda x: 1 if x==x else 0)
for dataset in full_data:

    le = LabelEncoder()

    dataset['original_langauge_code'] = le.fit_transform(dataset['original_language'])
train.head(5)
# Fill the missing data

test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/98'
for dataset in full_data:

    dataset['real_release_date'] = dataset['release_date'].apply(lambda x: (x[:-2] + '20' + x.split('/')[2]) 

                                                                  if int(x.split('/')[2]) <= 19 else (x[:-2] + '19' + x.split('/')[2]))

    dataset['real_release_date'] = pd.to_datetime(dataset['real_release_date'])

    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']

    for part in date_parts:

        dataset['release_date_' + part] = getattr(dataset['real_release_date'].dt, part).astype(int)

  
train.head()
# data fixes from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3

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

train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   

train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 

train.loc[train['id'] == 1542,'budget'] = 1              # All at Once

train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture

test.loc[test['id'] == 3889,'budget'] = 15000000       # Colossal

test.loc[test['id'] == 6733,'budget'] = 5000000        # The Big Sick

test.loc[test['id'] == 3197,'budget'] = 8000000        # High-Rise

test.loc[test['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2

test.loc[test['id'] == 5704,'budget'] = 4300000        # French Connection II

test.loc[test['id'] == 6109,'budget'] = 281756         # Dogtooth

test.loc[test['id'] == 7242,'budget'] = 10000000       # Addams Family Values

test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee



power_six = train.id[train.budget > 1000][train.revenue < 100]



for k in power_six :

    train.loc[train['id'] == k,'revenue'] =  train.loc[train['id'] == k,'revenue'] * 1000000
fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

plt.hist(train['revenue']);

plt.title('Distribution of revenue');

plt.subplot(1, 2, 2)

plt.hist(np.log1p(train['revenue']));

plt.title('Distribution of log of revenue');
train.head(2)
drop_columns = ['id', 'genres', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path', 'production_companies',

               'production_countries', 'release_date', 'spoken_languages', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew', 'real_release_date',

               'cast_name', 'cast_character', 'cast_gender', 'crew_name', 'crew_job', 'crew_department', 'crew_gender', 'all_genres']

train['runtime'].fillna(train['runtime'].median(), inplace=True)

test['runtime'].fillna(train['runtime'].median(), inplace=True)



train.drop(drop_columns, axis=1, inplace=True)

test.drop(drop_columns, axis=1, inplace=True)
print(train.dtypes)
X = train.drop('revenue', axis=1)

y = train['revenue']

sub = pd.read_csv('../input/sample_submission.csv')

print(X.shape, y.shape, test.shape)





X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
from sklearn.metrics import mean_squared_error

from math import sqrt

import math



#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

def rmsle(real, predicted):

    sum=0.0

    for x in range(len(predicted)):

        if predicted[x]<0 or real[x]<0: #check for negative values

            continue

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p - r)**2

    return (sum/len(predicted))**0.5





def modelfit(alg, X_train, y_train, X_valid, y_valid, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, 

                          metrics='rmse', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(X_train, y_train,eval_metric='rmse')

        

    #Predict training set:

    dtrain_predictions = alg.predict(X_valid)

      

    #Print model report:

    print("\nModel Report")

    print("RMSE : %.4f" % rmsle(y_valid.values, dtrain_predictions))

                    

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)[0:20]

    

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')
#Choose all predictors except target & IDcols

import xgboost as xgb

xgb1 = xgb.XGBRegressor(

    learning_rate =0.1,

    n_estimators=140,

    max_depth=4,

    min_child_weight=4,

    gamma=0,

    subsample=0.8,

    colsample_bytree=0.8,

    nthread=4,

    scale_pos_weight=1,

    seed=27)



modelfit(xgb1, X_train, y_train, X_valid, y_valid)
final = xgb.XGBRegressor(learning_rate =0.003, n_estimators=170, max_depth=4,

                                                  min_child_weight=4, gamma=0, subsample=0.8, colsample_bytree=0.8,nthread=4, 

                                                  scale_pos_weight=1, seed=27)

final.fit(X,y,eval_metric='rmse')

xgb_y_pred = final.predict(test)

sub['revenue'] = xgb_y_pred

sub.to_csv("XGBoost_advance.csv", index=False)
'''

# XGBoosting

import xgboost as xgb

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X,y)

XG_y_pred = xg_reg.predict(test)

sub['revenue'] = XG_y_pred

sub.to_csv("XG.csv", index=False)

print(X.shape, y.shape, test.shape)

'''
'''

import xgboost as xgb

xgb_params = {'objective': 'reg:linear'}

params = {'objective': 'reg:linear', 

    'eta': 0.01, 

    'max_depth': 6, 

    'subsample': 0.6, 

    'colsample_bytree': 0.7,  

    'eval_metric': 'rmse', 

    'seed': 2019, 

    'silent': True,

}



train_data = xgb.DMatrix(X, label=y)

#watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

xgb_model = xgb.train(dtrain=train_data, num_boost_round=200, params=params)

xgb_y_pred = xgb_model.predict(xgb.DMatrix(test))



sub['revenue'] = xgb_y_pred

sub.to_csv("XGBoost.csv", index=False)

'''
# LGB

import lightgbm as lgb

#params = {'objective': 'regression'}

params = {'objective':'regression',

    'num_leaves' : 30,

    'min_data_in_leaf' : 20,

    'max_depth' : 9,

    'learning_rate': 0.004,

    #'min_child_samples':100,

    'feature_fraction':0.9,

    "bagging_freq": 1,

    "bagging_fraction": 0.9,

    'lambda_l1': 0.2,

    "bagging_seed": 2019,

    "metric": 'rmse',

    #'subsample':.8, 

    #'colsample_bytree':.9,

    "random_state" : 2019,

    "verbosity": -1

}

model1 = lgb.LGBMRegressor(**params, n_estimators = 3000)

model1.fit(X_train.values, y_train.values, 

        eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)



LGB_y_pred = model1.predict(test, num_iteration=model1.best_iteration_)

sub['revenue'] = LGB_y_pred

sub.to_csv("LGB.csv", index=False)

print(LGB_y_pred[0:10].astype(int))
# GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingRegressor

GDB = GradientBoostingRegressor().fit(X,y)

GDB_y_pred = GDB.predict(test)



sub['revenue'] = GDB_y_pred

sub.to_csv("GDB.csv", index=False)
###### BaggingRegressor

from sklearn.ensemble import BaggingRegressor

BR = BaggingRegressor().fit(X,y)

BR_y_pred = BR.predict(test)

sub['revenue'] = BR_y_pred

sub.to_csv("BR.csv", index=False)