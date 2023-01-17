import numpy as np

import pandas as pd

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/test.csv')
X_train = train.drop(['revenue'],axis=1)

y_train = train['revenue']

print(X_train.shape, y_train.shape)
X = pd.concat([X_train, test], axis=0)
X.head(3)
X['has_homepage'] = X['homepage'].isnull() == False

X['is_original_english'] = X['original_language'] == 'en'

X['has_collection'] = X['belongs_to_collection'].isnull() == False

X['has_two_titles'] = X['original_title'] != X['title']

X.drop(['status','original_language','poster_path', 'homepage', 'imdb_id','belongs_to_collection', 'id'], axis=1, inplace=True)

X.head(2)
X.columns
X.loc[pd.isnull(X['spoken_languages']) == True,'spoken_languages'] = 0

X['lang'] = list(map(lambda x: [i['iso_639_1'] for i in eval(x)] if x!=0 else [], X['spoken_languages'].values))

X['n_lang'] = X['lang'].apply(lambda x: len(x))



# temp_lang = ' '.join(list(map(lambda x: ' '.join(x), X['lang']))).split(' ')



spoken_features = ['' + i for i in ['', 'la', 'it', 'cs', 'ta', 'pt', 'hu', 'zh', 'pl', 'ar', 'en', 'ja', 'de', 'ko', 'cn', 'tr',

 'he', 'sv', 'el', 'ru', 'fr', 'es', 'hi', 'th']]



for i in spoken_features:

    X[i] = X['lang'].apply(lambda x: i[7:] in x)



X.drop(['original_title', 'spoken_languages', 'lang'], axis=1, inplace=True)
X.head(2)
X.loc[pd.isnull(X['genres']) == True,'genres'] = 0

genres = set(' '.join([' '.join(i) for i in list(map(lambda x: [i['name'] for i in eval(x)] if x!=0 else [], X['genres'].values))]).split())



X['genres'] = list(map(lambda x: [i['name'] for i in eval(x)] if x!=0 else [], X['genres'].values))



for i in genres:

    X['genre_' + i] = X['genres'].apply(lambda x: i in x)
X['n_genres'] =  X['genres'].apply(lambda x: len(x))
X['release_month'] = 0

X['release_day'] = 0

X['release_year'] = 0



X = pd.concat([X, X['release_date'].str.split('/', expand=True)], axis=1)

X.head(2)
X.iloc[:,-1] = X.iloc[:,-1].fillna('0').astype(int)
year_mod = []

for i in X.iloc[:,-1].values:

    if i in range(0, 19):

        year_mod.extend([2000 + i])

    else:

        year_mod.extend([1900 + i])

year_mod



X['release_year'] = year_mod
X.head(2)
X = pd.concat([X, pd.get_dummies(X[0], prefix='release_month')], axis=1)

X.head(2)
X['release_date'] = pd.to_datetime(X['release_date'])
X['release_weekday'] = X['release_date'].dt.weekday.fillna(8).astype(int)
X.loc[:,'production_companies'] = X.loc[:,'production_companies'].fillna('[]')



companies = ','.join([','.join(i) for i in list(map(lambda x: [i['name'] for i in eval(x)], X['production_companies'].values))]).split(',')

unique_companies = set(companies)

# print(companies)



X['production_companies'] = list(map(lambda x: [i['name'] for i in eval(x)], X['production_companies'].values))
prod_count = {i: sum([1 for j in companies if i == j]) for i in unique_companies}



most_famous_prod = [k for k,v in prod_count.items() if v > 100 and k]

famous_prod = [k for k,v in prod_count.items() if 30 <= v < 100 and k]
X['n_production_companies'] = X['production_companies'].apply(lambda x: len(x))

X['most_famous_prod'] = X['production_companies'].apply(lambda x: sum([1 for i in x if i in most_famous_prod]))

X['famous_prod'] = X['production_companies'].apply(lambda x: sum([1 for i in x if i in famous_prod]))

X.head(2)
X.loc[:,'production_countries'] = X.loc[:,'production_countries'].fillna('[]')



countries = ','.join([','.join(i) for i in list(map(lambda x: [i['iso_3166_1'] for i in eval(x)], X['production_countries'].values))]).split(',')

unique_countries = set(countries)

# print(unique_countries)



X['production_countries'] = list(map(lambda x: [i['iso_3166_1'] for i in eval(x)], X['production_countries'].values))
country_count = {i: sum([1 for j in countries if i == j]) for i in unique_countries}

# sorted(country_count.items(), key=lambda x: x[1], reverse=True)



most_famous_countries= [k for k,v in country_count.items() if v > 100 and k]

famous_countries = [k for k,v in country_count.items() if 30 <= v < 100 and k]



X['n_production_countries'] = X['production_countries'].apply(lambda x: len(x))

X['most_famous_countries'] = X['production_countries'].apply(lambda x: sum([1 for i in x if i in most_famous_countries]))

X['famous_countries'] = X['production_countries'].apply(lambda x: sum([1 for i in x if i in famous_countries]))
X.columns
X['has_tagline'] = X['tagline'].apply(lambda x: pd.isnull(x))
X.drop(['genres', 'overview', 'production_companies', 'production_countries', 'release_date', 'tagline', 'release_month', 'release_day', 0, 2,

       'title', 'Keywords', 'cast','crew'], axis=1, inplace=True)

X.head(2)
X['runtime'] = X['runtime'].fillna(X['runtime'].mean())
X[1] = X[1].fillna(1)
for f in X.dtypes[X.dtypes == 'bool'].index:

    X[f] = X[f].astype(int)
X['popularity'] = (X['popularity'] - X['popularity'].mean()) / (X['popularity'].max()-X['popularity'].min())
X['runtime'] = (X['runtime'] - X['runtime'].mean()) / (X['runtime'].max()-X['runtime'].min())
X['inflationBudget'] = X['budget'] + X['budget']*1.8/100*(2019-X['release_year'])
X['budget'] = (X['budget'] - X['budget'].mean()) / (X['budget'].max()-X['budget'].min())

X['inflationBudget'] = (X['inflationBudget'] - X['inflationBudget'].mean()) / (X['inflationBudget'].max()-X['inflationBudget'].min())
X.head(2)
X = X.reset_index()
from sklearn.linear_model import LinearRegression



regressor = LinearRegression(normalize=True)

regressor.fit(X[:X_train.shape[0]], y_train)



y_test_pred = regressor.predict(X[X_train.shape[0]:])
y_test_pred.shape
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_log_error, make_scorer



score = cross_val_score(regressor, X[:X_train.shape[0]], y_train)

(abs(score[0]) + abs(score[1]) + abs(score[2]))/3
# import xgboost as xgb



# def xgb_model(trn_x, trn_y, val_x, val_y, test, verbose) :

    

#     params = {'objective': 'reg:linear', 

#               'eta': 0.01, 

#               'max_depth': 6, 

#               'subsample': 0.6, 

#               'colsample_bytree': 0.7,  

#               'eval_metric': 'rmse', 

#               'seed': random_seed, 

#               'silent': True,

#     }

    

#     record = dict()

#     model = xgb.train(params

#                       , xgb.DMatrix(trn_x, trn_y)

#                       , 100000

#                       , [(xgb.DMatrix(trn_x, trn_y), 'train'), (xgb.DMatrix(val_x, val_y), 'valid')]

#                       , verbose_eval=verbose

#                       , early_stopping_rounds=500

#                       , callbacks = [xgb.callback.record_evaluation(record)])

#     best_idx = np.argmin(np.array(record['valid']['rmse']))



#     val_pred = model.predict(xgb.DMatrix(val_x), ntree_limit=model.best_ntree_limit)

#     test_pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)



#     return {'val':val_pred, 'test':test_pred, 'error':record['valid']['rmse'][best_idx], 'importance':[i for k, i in model.get_score().items()]}
# import lightgbm as lgb



# def lgb_model(trn_x, trn_y, val_x, val_y, test, verbose) :



#     params = {'objective':'regression',

#          'num_leaves' : 30,

#          'min_data_in_leaf' : 20,

#          'max_depth' : 9,

#          'learning_rate': 0.004,

#          #'min_child_samples':100,

#          'feature_fraction':0.9,

#          "bagging_freq": 1,

#          "bagging_fraction": 0.9,

#          'lambda_l1': 0.2,

#          "bagging_seed": random_seed,

#          "metric": 'rmse',

#          #'subsample':.8, 

#           #'colsample_bytree':.9,

#          "random_state" : random_seed,

#          "verbosity": -1}



#     record = dict()

#     model = lgb.train(params

#                       , lgb.Dataset(trn_x, trn_y)

#                       , num_boost_round = 100000

#                       , valid_sets = [lgb.Dataset(val_x, val_y)]

#                       , verbose_eval = verbose

#                       , early_stopping_rounds = 500

#                       , callbacks = [lgb.record_evaluation(record)]

#                      )

#     best_idx = np.argmin(np.array(record['valid_0']['rmse']))



#     val_pred = model.predict(val_x, num_iteration = model.best_iteration)

#     test_pred = model.predict(test, num_iteration = model.best_iteration)

    

#     return {'val':val_pred, 'test':test_pred, 'error':record['valid_0']['rmse'][best_idx], 'importance':model.feature_importance('gain')}
# from catboost import CatBoostRegressor



# def cat_model(trn_x, trn_y, val_x, val_y, test, verbose) :

    

#     model = CatBoostRegressor(iterations=100000,

#                                  learning_rate=0.004,

#                                  depth=5,

#                                  eval_metric='RMSE',

#                                  colsample_bylevel=0.8,

#                                  random_seed = random_seed,

#                                  bagging_temperature = 0.2,

#                                  metric_period = None,

#                                  early_stopping_rounds=200

#                                 )

#     model.fit(trn_x, trn_y,

#                  eval_set=(val_x, val_y),

#                  use_best_model=True,

#                  verbose=False)

    

#     val_pred = model.predict(val_x)

#     test_pred = model.predict(test)

    

#     return {'val':val_pred, 

#             'test':test_pred, 

#             'error':model.get_best_score()['validation_0']['RMSE']}
# result_dict = dict()

# val_pred = np.zeros(train.shape[0])

# test_pred = np.zeros(test.shape[0])

# final_err = 0

# verbose = False



# for i, (trn, val) in enumerate(fold) :

#     print(i+1, "fold.    RMSE")

    

#     trn_x = train.loc[trn, :]

#     trn_y = y[trn]

#     val_x = train.loc[val, :]

#     val_y = y[val]

    

#     fold_val_pred = []

#     fold_test_pred = []

#     fold_err = []

    

#     #""" xgboost

#     start = datetime.now()

#     result = xgb_model(trn_x, trn_y, val_x, val_y, test, verbose)

#     fold_val_pred.append(result['val']*0.2)

#     fold_test_pred.append(result['test']*0.2)

#     fold_err.append(result['error'])

#     print("xgb model.", "{0:.5f}".format(result['error']), '(' + str(int((datetime.now()-start).seconds/60)) + 'm)')

#     #"""

    

#     #""" lightgbm

#     start = datetime.now()

#     result = lgb_model(trn_x, trn_y, val_x, val_y, test, verbose)

#     fold_val_pred.append(result['val']*0.4)

#     fold_test_pred.append(result['test']*0.4)

#     fold_err.append(result['error'])

#     print("lgb model.", "{0:.5f}".format(result['error']), '(' + str(int((datetime.now()-start).seconds/60)) + 'm)')

#     #"""

    

#     #""" catboost model

#     start = datetime.now()

#     result = cat_model(trn_x, trn_y, val_x, val_y, test, verbose)

#     fold_val_pred.append(result['val']*0.4)

#     fold_test_pred.append(result['test']*0.4)

#     fold_err.append(result['error'])

#     print("cat model.", "{0:.5f}".format(result['error']), '(' + str(int((datetime.now()-start).seconds/60)) + 'm)')

#     #"""

    

#     # mix result of multiple models

#     val_pred[val] += np.mean(np.array(fold_val_pred), axis = 0)

#     #print(fold_test_pred)

#     #print(fold_test_pred.shape)

#     #print(fold_test_pred.columns)

#     test_pred += np.mean(np.array(fold_test_pred), axis = 0) / k

#     final_err += (sum(fold_err) / len(fold_err)) / k

    

#     print("---------------------------")

#     print("avg   err.", "{0:.5f}".format(sum(fold_err) / len(fold_err)))

#     print("blend err.", "{0:.5f}".format(np.sqrt(np.mean((np.mean(np.array(fold_val_pred), axis = 0) - val_y)**2))))

    

#     print('')

    

# print("fianl avg   err.", final_err)

# print("fianl blend err.", np.sqrt(np.mean((val_pred - y)**2)))
submission = pd.read_csv('/kaggle/input/tmdb-box-office-prediction/sample_submission.csv')

submission['revenue'] = y_test_pred

submission.to_csv('submission.csv', index=False)