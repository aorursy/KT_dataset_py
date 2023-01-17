import sys

import sklearn



import os



from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import ast



%matplotlib inline 



import warnings

warnings.filterwarnings(action="ignore")



pd.set_option('display.max_columns', 500) 

pd.set_option('display.max_rows', 500)



print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
X_train = train.drop(['revenue'],axis=1)

y_train = train['revenue']

print(X_train.shape, y_train.shape)
dict_columns = ['belongs_to_collection', 'genres', 'production_companies','production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

        

train = text_to_dict(train)

test = text_to_dict(test)
X = pd.concat([X_train, test], axis=0, ignore_index=True)
X['has_homepage'] = X['homepage'].isnull() == False

X['is_original_english'] = X['original_language'] == 'en'

X['has_collection'] = X['belongs_to_collection'].isnull() == False

X['has_two_titles'] = X['original_title'] != X['title']

X.drop(['status','original_language','poster_path', 'homepage', 'imdb_id','belongs_to_collection', 'id'], axis=1, inplace=True)

X.head(2)
X.loc[pd.isnull(X['spoken_languages']) == True,'spoken_languages'] = 0

X['lang'] = list(map(lambda x: [i['iso_639_1'] for i in eval(x)] if x!=0 else [], X['spoken_languages'].values))

X['n_lang'] = X['lang'].apply(lambda x: len(x))



# temp_lang = ' '.join(list(map(lambda x: ' '.join(x), X['lang']))).split(' ')



spoken_features = ['' + i for i in ['', 'la', 'it', 'cs', 'ta', 'pt', 'hu', 'zh', 'pl', 'ar', 'en', 'ja', 'de', 'ko', 'cn', 'tr',

 'he', 'sv', 'el', 'ru', 'fr', 'es', 'hi', 'th']]



for i in spoken_features:

    X[i] = X['lang'].apply(lambda x: i[7:] in x)



X.drop(['original_title', 'spoken_languages', 'lang'], axis=1, inplace=True)
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
X['has_tagline'] = X['tagline'].apply(lambda x: pd.isnull(x))



X.drop(['genres', 'overview', 'production_companies', 'production_countries', 'release_date', 'tagline', 'release_month', 'release_day', 0, 2,

       'title', 'Keywords', 'cast','crew'], axis=1, inplace=True)

X.head(2)
X['budget_log'] = np.log1p(X['budget'])
X['inflationBudget'] = X['budget'] + X['budget']*1.8/100*(2019-X['release_year'])

X['runtime'] = X['runtime'].fillna(X['runtime'].mean())



X[1] = X[1].fillna(1)



for f in X.dtypes[(X.dtypes == 'bool') | (X.dtypes == 'object')].index:

    X[f] = X[f].astype(int)
data_dropping_names = data.drop(['original_title','overview','tagline','title'], axis=1)



train = data_dropping_names[data_dropping_names['source'] == 'train'].copy()

test = data_dropping_names[data_dropping_names['source'] == 'test'].copy()



train_labels = train['revenue_log'] #creating labels, our Y_train, gonna use the log as it works better for skewed data



train.drop(['id', 'revenue', 'source', 'revenue_log'], axis=1, inplace=True) # dropping the target and id



test_final = test.drop(['id', 'source','revenue','revenue_log'], axis=1) #this is the final test, the dataset give for our prediction, notice that I am dropping the revenue column here that has been created when we merged the two datasets together
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,RobustScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('robust_scaler', RobustScaler())

        ])
import time

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import  KFold

from catboost import CatBoostRegressor

import lightgbm as lgb

import xgboost as xgb

import eli5

import gc



n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):



    prediction = np.zeros(X_test.shape[0]) #initializing the prediction matrix with zeros, with the number of training examples in X_test

    scores = [] #this list is gonna be used to store all the scores across different folds

    feature_importance = pd.DataFrame() #initializing this dataframe, it's gonna be used to plot the features importance.

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print('Fold', fold_n, 'started at', time.ctime())

        if model_type == 'sklearn': #if the model type is sklearn then

            X_train, X_valid = X[train_index], X[valid_index]

        else:

            X_train, X_valid = X.values[train_index], X.values[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if model_type == 'lgb': 

            model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

                    verbose=1000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test.values), ntree_limit=model.best_ntree_limit)



        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_squared_error(y_valid, y_pred_valid)

            

            y_pred = model.predict(X_test)

            

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=20000,  eval_metric='RMSE', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)

        

        prediction += y_pred #summing all the prediction which is gonna later be divided by the number of folds   

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold # all the predictions divided by the number of folds(getting the average value)

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    # Clean up memory

    gc.enable()

    del model, y_pred_valid, X_test,X_train,X_valid, y_pred, y_train

    gc.collect()



    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return scores, prediction, feature_importance 

        return scores, prediction

    else:

        return scores, prediction

    


train_dummies = pd.get_dummies(X[:X_train.shape[0]])

test_dummies = pd.get_dummies(X[X_train.shape[0]:])

train_dummies, test_dummies = train_dummies.align(test_dummies, axis=1, join='inner')
params = {

          'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 6,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

}

score_lgb, prediction_lgb, _ = train_model(train_dummies, test_dummies, train_labels, params=params, model_type='lgb', plot_feature_importance=True)
sub = pd.read_csv('../input/sample_submission.csv')

sub['revenue'] = np.expm1(prediction_lgb)

sub.to_csv("lgb_model.csv", index=False)
xgb_params = {'eta': 0.01,

              'objective': 'reg:linear',

              'max_depth': 7,

              'subsample': 0.8,

              'colsample_bytree': 0.8,

              'eval_metric': 'rmse',

              'seed': 11,

              'silent': True}

score_xgb, prediction_xgb = train_model(train_dummies, test_dummies, train_labels, params=xgb_params, model_type='xgb', plot_feature_importance=True)
score_xgb.sort(reverse=True)

dictvalues.update({'RMSE_XGB': score_xgb})
plt.plot(dictvalues['RMSE_lgb'], label = 'RMSE_lgb')

plt.plot(dictvalues['RMSE_XGB'], label = 'RMSE_XGB')

plt.legend()
sub['revenue'] = np.expm1(prediction_xgb)

sub.to_csv("XGB_model.csv", index=False)
cat_params = {'learning_rate': 0.002,

              'depth': 5,

              'l2_leaf_reg': 10,

              'colsample_bylevel': 0.8,

              'bagging_temperature': 0.2,

              'od_type': 'Iter',

              'od_wait': 100,

              'random_seed': 11,

              'allow_writing_files': False}

score_cat, prediction_cat = train_model(train_dummies, test_dummies, train_labels, params=cat_params, model_type='cat', plot_feature_importance=True)
score_cat.sort(reverse=True)

dictvalues.update({'RMSE_CAT': score_cat})

plt.plot(dictvalues['RMSE_lgb'], label = 'RMSE_lgb')

plt.plot(dictvalues['RMSE_XGB'], label = 'RMSE_XGB')

plt.plot(dictvalues['RMSE_CAT'], label = 'RMSE_CAT')

plt.legend()
sub['revenue'] = np.expm1(prediction_cat)

sub.to_csv("cat_model.csv", index=False)
sub['revenue'] = np.expm1((prediction_lgb + prediction_xgb + prediction_cat) / 3)

sub.to_csv("combined.csv", index=False)