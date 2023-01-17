#Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

import ast

from tqdm import tqdm

import time

from collections import Counter

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression,Lasso

from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.feature_selection import SelectKBest,chi2

import xgboost as xgb

import lightgbm as lgb

from catboost import CatBoostRegressor

from xgboost.sklearn import XGBRegressor

from xgboost import plot_importance

from types import FunctionType

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

%matplotlib inline

seed = 123
# Data import

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('train dataset size:', train.shape)

print('test dataset size:', test.shape)
train.info()
train.head(3)
'''Total rows:3000

Data columns (total 23 columns):

id  &nbsp;                       3000 non-null int64   unique id given to movie  

belongs_to_collection    604 non-null object   title,poster path etc given in jason format  

budget                   3000 non-null int64   budget of the movie,assuming in USD  

genres                   2993 non-null object  one movie can have multiple genre given in list of dictionaries  

homepage                 946 non-null object   homepage of production company,I think  

imdb_id                  3000 non-null object  unique id given to movie on IMDB website  

original_language        3000 non-null object  original_language of the movie 

original_title           3000 non-null object  title of the movie 

overview                 2992 non-null object  short overview about movie story 

popularity               3000 non-null float64 score based on popularity,how this score is calculated is not given to us

poster_path              2999 non-null object  path to image of movie poster

production_companies     2844 non-null object  one movie can have multiple production companies  

production_countries     2945 non-null object  name of the country where movie production took place 

release_date             3000 non-null object  movie release date in mm/dd/yy format

runtime                  2998 non-null float64 movie runtime in minutes

spoken_languages         2980 non-null object  language spoken in movie given in list of dictionary

status                   3000 non-null object  Status of the movie.Either 'Released' or 'Rumored'

tagline                  2403 non-null object  Tagline given in String format

title                    3000 non-null object  Title of the movie

Keywords                 2724 non-null object  List of keywords related to movie plot

cast                     2987 non-null object  List of cast and its details

crew                     2984 non-null object  list of crew and their detail

revenue                  3000 non-null int64   revenue earned by the movie,this is our taget variable.'''
columns_to_keep = set()
#This method will clean feature with dictionary data.

#Create new feature with total number of values,onehot encoded feature

def clean_dictionary_features(feature_name,train,test):

    #convert string to dictionary

    train[feature_name] = train[feature_name].apply(lambda x:{} if pd.isna(x) else ast.literal_eval(x))

    test[feature_name] = test[feature_name].apply(lambda x:{} if pd.isna(x) else ast.literal_eval(x))

    

    #create new feature of total count of values

    train[feature_name+'_number'] = train[feature_name].apply(lambda x:len(x) if x!={} else 0)

    test[feature_name+'_number'] = test[feature_name].apply(lambda x:len(x) if x!={} else 0)

    columns_to_keep.add(feature_name+'_number')

    

    #get list of all values

    list_of_values = list(train[feature_name].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)



    train[feature_name+'_all'] = train[feature_name].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

    top_values = [m[0] for m in Counter([i for j in list_of_values for i in j]).most_common(10)]

    

    #Create one hot encoded feature

    for val in top_values:

        train[feature_name +'_'+val] = train[feature_name+'_all'].apply(lambda x: 1 if val in x else 0)

        columns_to_keep.add(feature_name +'_'+val)

    

    test[feature_name+'_all'] = test[feature_name].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

    for val in top_values:

        test[feature_name +'_'+val] = test[feature_name+'_all'].apply(lambda x: 1 if val in x else 0)

    

    #Create Lable encoded feature 

    le = LabelEncoder()

    le.fit(list(train[feature_name+'_all'].fillna('')) + list(test[feature_name+'_all'].fillna('')))

    train[feature_name+'_all'] = le.transform(train[feature_name+'_all'].fillna('').astype(str))

    test[feature_name+'_all'] = le.transform(test[feature_name+'_all'].fillna('').astype(str))

    columns_to_keep.add(feature_name+'_all')

    return train,test
train,test = clean_dictionary_features('genres',train,test)

train,test = clean_dictionary_features('production_companies',train,test)

train,test = clean_dictionary_features('production_countries',train,test)

train,test = clean_dictionary_features('spoken_languages',train,test)

train,test = clean_dictionary_features('Keywords',train,test)

train,test = clean_dictionary_features('cast',train,test)

train,test = clean_dictionary_features('crew',train,test)

print("-"*40,'\n',train.isnull().sum())

print("-"*40,'\n',test.isnull().sum())
train.belongs_to_collection.describe()
train['belongs_to_collection'][0]
train['got_collection'] = train['belongs_to_collection'].apply(lambda x:0 if pd.isnull(x) else 1)

test['got_collection'] = test['belongs_to_collection'].apply(lambda x:0 if pd.isnull(x) else 1)
sns.catplot(x='got_collection', y='revenue', data=train);
columns_to_keep.add('got_collection')
train.budget.describe()
sns.distplot(train['budget'])
sns.distplot(np.log1p(train['budget']))
len(train[train['budget']==0])
sns.jointplot(x=np.log1p(train['budget']), y=np.log1p(train['revenue']), data=train, height=8, ratio=4, color="b")
train['budget_log'] = np.log1p(train.budget.values)

test['budget_log'] = np.log1p(test.budget.values)
columns_to_keep.add('budget_log')
train.revenue.corr(train.budget_log,method='pearson')
train.original_language.describe()
sns.catplot('original_language',data=train,kind='count')
train.groupby(['original_language']).mean()[['revenue']].plot(kind='bar')
train['belongs_to_three_lang'] = train.original_language.apply(lambda x:1 if str(x) in['en','zh','tr'] else 0)

test['belongs_to_three_lang'] = test.original_language.apply(lambda x:1 if str(x) in['en','zh','tr'] else 0)
columns_to_keep.add('belongs_to_three_lang')
le = LabelEncoder()

le.fit(list(train['original_language'].fillna('')) + list(test['original_language'].fillna('')))

train['original_language_encoded'] = le.transform(train['original_language'].fillna('').astype(str))

test['original_language_encoded'] = le.transform(test['original_language'].fillna('').astype(str))
columns_to_keep.add('original_language_encoded')
train.original_title.describe()
sns.scatterplot(x=train.original_title.str.len(),y=train.revenue)
train['original_title_length'] = train.original_title.str.len()

test['original_title_length'] = test.original_title.str.len()
train.revenue.corr(train.original_title_length)
columns_to_keep.add('original_title_length')
train.overview.describe()
train.overview[3]
train.popularity.describe()
sns.distplot(train.popularity)
sns.distplot(np.log1p(train.popularity))
plt.boxplot(np.log1p(train.popularity))
sns.scatterplot(x=train.popularity,y=np.log1p(train.revenue))
train.popularity.corr(train.revenue)
columns_to_keep.add('popularity')
train.release_date.describe()
#as year is in yy format we have to handle movies after 20xx.So this method will help to add century to year

def clean_date(date):

    year = date.split('/')[2]

    if int(year) <= 19:

        return date[:-2] + '20' + year

    else:

        return date[:-2] + '19' + year
test.loc[test['release_date'].isnull() == True, 'release_date'] = '05/01/00'

train['release_date'] = train['release_date'].apply(lambda x:clean_date(x))

test['release_date'] = test['release_date'].apply(lambda x:clean_date(x))

train['release_date'] = pd.to_datetime(train['release_date'])

test['release_date'] = pd.to_datetime(test['release_date'])
#get time period features from date value

def date_features(dataset):

    date_sections = ["year", "weekday", "month", 'weekofyear', 'day']

    for sec in date_sections:

        section_col = 'release_date' + "_" + sec

        dataset[section_col] = getattr(dataset['release_date'].dt, sec).astype(int)

        columns_to_keep.add(section_col)

    return dataset



train = date_features(train)

test = date_features(test)
train.runtime.describe()
train.runtime.isnull().sum()
test.runtime.isnull().sum()
train.runtime = train.runtime.fillna(np.mean(train.runtime))

test.runtime = test.runtime.fillna(np.mean(test.runtime))
sns.scatterplot(train.runtime,np.log1p(train.revenue))
train.revenue.corr(train.runtime)
columns_to_keep.add('runtime')
train.status.describe()
train.status.value_counts()
train.tagline.describe()
train.tagline[:3]
sns.scatterplot(x=train.tagline.str.len(),y=train.revenue)
train['tagline_count'] = train['tagline'].apply(lambda x: 0 if pd.isnull(x) else len(x))

test['tagline_count'] = test['tagline'].apply(lambda x: 0 if pd.isnull(x) else len(x))
columns_to_keep.add('tagline_count')
train.title.describe()
sns.scatterplot(x=train.title.str.len(),y=train.revenue)
train['title_count'] = train['title'].apply(lambda x: 0 if pd.isnull(x) else len(x))

test['title_count'] = test['title'].apply(lambda x: 0 if pd.isnull(x) else len(x))
columns_to_keep.add('title_count')
#budget must be high for popular movies

train['budget_popularity'] = train['budget']/train['popularity']

test['budget_popularity'] = test['budget']/test['popularity']

columns_to_keep.add('budget_popularity')



#budget increased since past

train['budget_release_year'] = train['budget']/train['release_date_year']

test['budget_release_year'] = test['budget']/test['release_date_year']

columns_to_keep.add('budget_release_year')



#popularity increased since past

train['popularity_release_year'] = train['popularity']/train['release_date_year']

test['popularity_release_year'] = test['popularity']/test['release_date_year']

columns_to_keep.add('popularity_release_year')



#popularity and day on which movie releases must be related

train['popularity_release_weekday'] = np.sqrt(train['popularity']*train['release_date_weekday'])

test['popularity_release_weekday'] = np.sqrt(test['popularity']*test['release_date_weekday'])

columns_to_keep.add('popularity_release_weekday')



#movies with more generes in it are recently being made

train['genres_number_release_year'] = train['genres_number']/train['release_date_year']

test['genres_number_release_year'] = test['genres_number']/test['release_date_year']

columns_to_keep.add('genres_number_release_year')





#movie runtime reduced w.r.t time

train['runtime_release_year'] = np.sqrt(train['runtime']*train['release_date_year'])

test['runtime_release_year'] = np.sqrt(test['runtime']*test['release_date_year'])

columns_to_keep.add('runtime_release_year')



#high runtime movies may require high budget

train['budget_runtime'] = np.sqrt(train['budget']*train['runtime'])

test['budget_runtime'] = np.sqrt(test['budget']*test['runtime'])

columns_to_keep.add('budget_runtime')



train['budget_tagline_count'] = np.sqrt(train['budget']*train['tagline_count'])

test['budget_tagline_count'] = np.sqrt(test['budget']*test['tagline_count'])

columns_to_keep.add('budget_tagline_count')
len(columns_to_keep)
columns_to_keep
numerical_features = ['Keywords_number','runtime','spoken_languages_number','production_countries_number',

                     'production_companies_number','popularity','genres_number','crew_number','cast_number','budget_log',

                     'budget_popularity','budget_release_year','popularity_release_year','popularity_release_weekday',

                     'genres_number_release_year','runtime_release_year','budget_runtime','budget_tagline_count']

date_features = ['release_date_day','release_date_month','release_date_weekday','release_date_weekofyear',

                        'release_date_year']

feature_text_length = ['title_count','tagline_count','original_title_length']

categorical_features = ['spoken_languages_all','production_companies_all','production_countries_all','original_language',

                       'got_collection','genres_all','crew_all','cast_all','belongs_to_three_lang','Keywords_all']

spoken_language_features = ['spoken_languages_','spoken_languages_Deutsch','spoken_languages_English','spoken_languages_Español',

                             'spoken_languages_Français','spoken_languages_Italiano', 'spoken_languages_Pусский',

                            'spoken_languages_हिन्दी','spoken_languages_日本語','spoken_languages_普通话']

production_countries_features = ['production_countries_Australia','production_countries_Canada','production_countries_France',

                                'production_countries_Germany','production_countries_India','production_countries_Italy',

                                'production_countries_Japan','production_countries_Russia','production_countries_United Kingdom',

                                'production_countries_United States of America' ]

production_companies_features = ['production_companies_Columbia Pictures','production_companies_Columbia Pictures Corporation',

                                 'production_companies_Metro-Goldwyn-Mayer (MGM)', 'production_companies_New Line Cinema',

                                 'production_companies_Paramount Pictures', 'production_companies_Touchstone Pictures',

                                 'production_companies_Twentieth Century Fox Film Corporation','production_companies_Universal Pictures',

                                 'production_companies_Walt Disney Pictures', 'production_companies_Warner Bros.']

genres_features = ['genres_Action', 'genres_Adventure','genres_Comedy', 'genres_Crime', 'genres_Drama', 'genres_Family',

                 'genres_Horror', 'genres_Romance', 'genres_Science Fiction', 'genres_Thriller']

crew_features = ['crew_Avy Kaufman', 'crew_Deborah Aquila', 'crew_Francine Maisler','crew_James Newton Howard',

                 'crew_Jerry Goldsmith', 'crew_Luc Besson', 'crew_Mary Vernieu', 'crew_Robert Rodriguez','crew_Steven Spielberg',

                 'crew_Tricia Wood']

cast_features = ['cast_Bruce McGill', 'cast_Bruce Willis','cast_J.K. Simmons','cast_John Turturro','cast_Liam Neeson',

                 'cast_Morgan Freeman','cast_Robert De Niro','cast_Samuel L. Jackson', 'cast_Susan Sarandon','cast_Willem Dafoe']

keywords_features = ['Keywords_aftercreditsstinger', 'Keywords_based on novel', 'Keywords_biography',

                     'Keywords_duringcreditsstinger', 'Keywords_independent film', 'Keywords_murder',

                     'Keywords_revenge', 'Keywords_sport', 'Keywords_violence', 'Keywords_woman director']

print(len(numerical_features)+len(date_features)+len(feature_text_length)+len(categorical_features)+len(spoken_language_features)

+len(production_countries_features)+len(production_companies_features)+len(genres_features)+len(crew_features)+len(cast_features)

+len(keywords_features))
numerical_data = numerical_features

numerical_data.append('revenue')
plt.figure(figsize=(12,6))

sns.heatmap(train[numerical_data].corr(), annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.title('Correlation between Numerical Features')

plt.show()
date_data = date_features

date_data.append('revenue')

plt.figure(figsize=(12,6))

sns.heatmap(train[date_data].corr(), annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.title('Correlation between date Features')

plt.show()
text_length_data = feature_text_length

text_length_data.append('revenue')

plt.figure(figsize=(12,6))

sns.heatmap(train[text_length_data].corr(), annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.title('Correlation between text length Features')

plt.show()
categorical_features_data = categorical_features

categorical_features_data.append('revenue')

plt.figure(figsize=(12,6))

sns.heatmap(train[categorical_features_data].corr(), annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.title('Correlation between categorical Features')

plt.show()
spoken_language_features_data = spoken_language_features

spoken_language_features_data.append('revenue')

plt.figure(figsize=(12,6))

sns.heatmap(train[spoken_language_features_data].corr(), annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.title('Correlation between spoken_language Features')

plt.show()
production_countries_features_data = production_countries_features

production_countries_features_data.append('revenue')

plt.figure(figsize=(12,6))

sns.heatmap(train[production_countries_features_data].corr(), annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.title('Correlation between production_countries Features')

plt.show()
production_companies_features_data = production_companies_features

production_companies_features_data.append('revenue')

plt.figure(figsize=(12,6))

sns.heatmap(train[production_companies_features_data].corr(), annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.title('Correlation between production_companies Features')

plt.show()
genres_features_data = genres_features

genres_features_data.append('revenue')

plt.figure(figsize=(12,6))

sns.heatmap(train[genres_features_data].corr(), annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.title('Correlation between genres_features Features')

plt.show()
crew_features_data = crew_features

crew_features_data.append('revenue')

plt.figure(figsize=(12,6))

sns.heatmap(train[crew_features_data].corr(), annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.title('Correlation between crew_features Features')

plt.show()
corr_features = list(columns_to_keep)

corr_features.append('revenue')

corrs = abs(train[corr_features].corr()['revenue']).sort_values(ascending=False)

corr_selected_features = corrs[:50].index.tolist()

corr_selected_features.remove('revenue')

#corr_selected_features
def select_model(X_train, X_val, y_train, y_val):



    best_models = {}

    models = [

        {

            'name': 'LinearRegression',

            'estimator': LinearRegression(),

            'hyperparameters': {},

        },

       

        {

            'name': 'GradientBoostingRegressor',

            'estimator': GradientBoostingRegressor(),

            'hyperparameters':{

                'n_estimators': range(100, 200, 10),

                'criterion': ['friedman_mse'],

                'max_depth': [3, 5, 7, 9],

                'max_features': ['log2', 'sqrt'],

                'min_samples_leaf': [1, 2, 4],

                'min_samples_split': [3, 5, 7]

            }

            

        },



        {

            'name': 'XGBoost',

            'estimator': xgb.XGBRegressor(),

            'hyperparameters':{

                'booster': ['gbtree', 'gblinear', 'dart'],

                'max_depth': range(5, 50, 5),

                'n_estimators': [200],

                'nthread': [4],

                'min_child_weight': range(1, 8, 2),

                'learning_rate': [.05, .1, .15],

            }

        },

        {

            'name': 'Light GBM',

            'estimator': lgb.LGBMRegressor(),

            'hyperparameters':{

                'max_depth': range(20, 85, 15),

                'learning_rate': [.01, .05, .1],

                'num_leaves': [300, 600, 900, 1200],

                'n_estimators': [200]

            }

        }

    ]

    

    for model in tqdm(models):

        print('\n', '-'*25, '\n', model['name'])

        start = time.perf_counter()

        grid = GridSearchCV(model['estimator'], param_grid=model['hyperparameters'], cv=5, scoring = "neg_mean_squared_error", verbose=False, n_jobs=-1)

        grid.fit(X_train, y_train)

        best_models[model['name']] = {'score': grid.best_score_, 'params': grid.best_params_}

        mse_val = mean_squared_error(y_val, grid.predict(X_val))

        mse_train = mean_squared_error(y_train, grid.predict(X_train))

        print("RMSLE train:{}".format(np.sqrt(mse_train))) 

        print("RMSLE validation:{}".format(np.sqrt(mse_val)))

        print("best_params_:{}".format(grid.best_params_))

        run = time.perf_counter() - start

        

        

    return best_models
def get_best_parameters(train,features_list):



    X_train = train[features_list]

    y_train = np.log1p(train["revenue"]).values

      

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)



    models = select_model(X_train, X_val, y_train, y_val)

    return models
# A class that will define all the regression models as methods



class Models(object):

    

    

    

    # Initialization 

    def __init__(self, x_train, x_validation, y_train, y_validation):

        # changing input as dataframe to list

        self.x_train = [x_train.iloc[i].tolist() for i in range(len(x_train))]

        self.x_validation = [x_validation.iloc[i].tolist() for i in range(len(x_validation))]

        self.y_train = y_train.tolist()

        self.y_validation = y_validation.tolist()

        

    

    

    @staticmethod

    def print_info(cross_val_scores, mse_train,mse_val):

        print("Cross Validation Scores: ", cross_val_scores)

        print("RMSLE train:{}".format(np.sqrt(mse_train))) 

        print("RMSLE validation:{}".format(np.sqrt(mse_val)))

        #print("Mean Squared Error: ", np.sqrt(mse))

        

        

  

    # Gradient Boosting Regressor

    def GBR(self, x_train, x_validation,  y_train, y_validation):

        gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.08,max_features='sqrt',criterion='friedman_mse',

                                        min_samples_leaf=1,min_samples_split=3, max_depth=7, random_state=seed)

        gbr.fit(self.x_train, self.y_train)

        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

        cross_val_scores = cross_val_score(gbr, self.x_train, self.y_train, cv=kfold)

        mse_val = mean_squared_error(self.y_validation, gbr.predict(self.x_validation))

        mse_train = mean_squared_error(self.y_train, gbr.predict(self.x_train))

        print('\nGradient Boosting Regressor')

        self.print_info(cross_val_scores, mse_train,mse_val)

        return cross_val_scores, mse_val, gbr

    

    

    # LGBM Regressor 

    def lgbm(self, x_train, x_validation,  y_train, y_validation):

        lgbm =lgb.LGBMRegressor(n_estimators=10000,objective="regression", metric="rmse",num_leaves=20, 

                             min_child_samples=100,learning_rate=0.01, bagging_fraction=0.8,feature_fraction=0.8, 

                             bagging_frequency=1,importance_type='gain', bagging_seed=seed,subsample=.9, 

                             colsample_bytree=.9,use_best_model=True)

                                

        lgbm.fit(x_train, y_train,eval_set=(x_validation, y_validation),verbose=False)

    

        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

        cross_val_scores = cross_val_score(lgbm, self.x_train, self.y_train, cv=kfold)

        mse_val = mean_squared_error(self.y_validation, lgbm.predict(self.x_validation))

        mse_train = mean_squared_error(self.y_train, lgbm.predict(self.x_train))

        print('\nLGBM Regressor')

        self.print_info(cross_val_scores, mse_train,mse_val)

        return cross_val_scores, mse_val, lgbm

    

    

    # XgBoost Regressor 

    def xgBoost(self, x_train, x_validation,  y_train, y_validation):

        params = {'objective': 'reg:linear','eta': 0.01,'max_depth': 6,'subsample': 0.6,'colsample_bytree': 0.7,  

              'eval_metric': 'rmse', 'seed': seed,'silent': True,}

    

        record = dict()

        xg = xgb.train(params, xgb.DMatrix(x_train, y_train), 100000, [(xgb.DMatrix(x_train, y_train), 'train'),

                                                                      (xgb.DMatrix(x_validation, y_validation), 'valid')]

                      , verbose_eval=False, early_stopping_rounds=500, callbacks = [xgb.callback.record_evaluation(record)])

        best_idx = np.argmin(np.array(record['valid']['rmse']))

    

        #kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

        #cross_val_scores = cross_val_score(xg, self.x_train, self.y_train, cv=kfold)

        cross_val_scores= 0

        mse_val = mean_squared_error(self.y_validation, xg.predict(xgb.DMatrix(x_validation), ntree_limit=xg.best_ntree_limit))

        mse_train = mean_squared_error(self.y_train, xg.predict(xgb.DMatrix(x_train), ntree_limit=xg.best_ntree_limit))

        print('\nXgBoost Regressor')

        self.print_info(cross_val_scores, mse_train,mse_val)

        #plot_importance(xg)

        #plt.show()

        return cross_val_scores, mse_val, xg
def evaluate_models(train, test,features_list):



    X_train = train[features_list]

    y_train = np.log1p(train["revenue"]).values

      

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)



    methods = [x for x, y in Models.__dict__.items() if type(y) == FunctionType]

    methods.remove('__init__')

    # Now calling the all regression methods

    cross_scores_list, mse_list = [], []

    models = {}

    for model in methods:

        reg = Models(X_train, X_val, y_train, y_val)

        cross_val_scores, mse, return_model = getattr(reg, model)(X_train, X_val, y_train, y_val)

        cross_scores_list.append(cross_val_scores)

        models[model] = return_model

        mse_list.append(mse)

    return models
#get_best_parameters(train,list(columns_to_keep))
#Evaluate models with best parameters and top 50 features based on correlation

corr_features_models = evaluate_models(train, test,corr_selected_features)
#Evaluate models with best parameters and  all features we created so far

all_features_models = evaluate_models(train, test,list(columns_to_keep))
def calculate_test_results(models,test,model_names,features_list):

        

    X_test = test[features_list]

    pred = np.empty(shape=len(X_test)).tolist()

    

    for model in model_names:

        mod = models[model]

        if(model=='xgBoost'):

            pred = pred + np.expm1(mod.predict(xgb.DMatrix(X_test), ntree_limit=mod.best_ntree_limit))

        else:   

            pred = pred + np.expm1(mod.predict(X_test))

    

    return pred
xgBoost_results = calculate_test_results(all_features_models,test,['xgBoost'],list(columns_to_keep))

gbr_results = calculate_test_results(all_features_models,test,['GBR'],list(columns_to_keep))

lgbm_results = calculate_test_results(all_features_models,test,['lgbm'],list(columns_to_keep))



#final_pred = 0.4*xgBoost_results + 0.4*gbr_results + 0.2*lgbm_results  2.11835

#final_pred = 0.7*xgBoost_results + 0.3*gbr_results 2.05313

#final_pred = 0.3*xgBoost_results + 0.7*gbr_results 2.10757

#final_pred = 0.6*xgBoost_results + 0.2*gbr_results + 0.2*lgbm_results  2.06723

#final_pred = 0.8*xgBoost_results + 0.2*gbr_results 2.05289

final_pred = 0.8*xgBoost_results + 0.2*lgbm_results 

#final_pred = 0.9*xgBoost_results + 0.1*gbr_results

#final_pred = xgBoost_results  2.05923



submission = pd.DataFrame()

submission['id'] = test['id']

submission['revenue'] = final_pred

submission.to_csv('submission.csv', index=False)