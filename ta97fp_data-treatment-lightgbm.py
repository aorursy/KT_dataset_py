# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split

from dask.distributed import Client

import os

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.dtypes
sns.heatmap(train.corr())
train[['Month','Day', 'Year']] = train['release_date'].str.split('/',expand=True)

trainset_release_date = train[['Month', 'Day', 'Year', 'revenue']]



trainset_release_date = trainset_release_date.astype(float)



cor = trainset_release_date.corr()



sns.heatmap(cor)



train = train.drop(['Year'], axis=1)



train.isna().sum()
runtime_mean = train['runtime'].mean()

print(runtime_mean)

train['runtime'] = train['runtime'].replace(0, runtime_mean)



train['homepage'] = train['homepage'].fillna(0)

train.loc[train['homepage'] != 0, 'homepage'] = 1
# Using function "from https://www.kaggle.com/tijlkindt/simple-tmdb-prediction-with-linear-regression - Simple TMDB prediction with Linear Regression" for JSON Treatment 



def proc_json(string, key):

    try:

        data = eval(string)

        return ",".join([d[key] for d in data])

    except:

        return ''



def proc_json_len(string):

    try:

        data = eval(string)

        return len(data)

    except:

        return 0





train.belongs_to_collection = train.belongs_to_collection.apply(lambda x: proc_json(x, 'name'))



train.genres = train.genres.apply(lambda x: proc_json(x, 'name'))



train.production_companies = train.production_companies.apply(lambda x: proc_json(x, 'name'))



train.production_countries = train.production_countries.apply(lambda x: proc_json(x, 'iso_3166_1'))



train.spoken_languages = train.spoken_languages.apply(lambda x: proc_json(x, 'iso_639_1'))



train.Keywords = train.Keywords.apply(lambda x: proc_json(x, 'name'))



train.cast = train.cast.apply(proc_json_len)



train.crew = train.crew.apply(proc_json_len)

#Genres



genres = []

for idx, val in train.genres.iteritems():

    gen_list = val.split(',')

    for gen in gen_list:

        if gen == '':

            continue



        if gen not in genres:

            genres.append(gen)

            



genre_column_names = []

for gen in genres:

    col_name = 'genres' + gen.replace(' ', '_')

    train[col_name] = train.genres.str.contains(gen).astype('uint8')

    genre_column_names.append(col_name)



train.drop(['genres'], axis=1)
#Target Encoding for categorical variables





means = train.groupby('overview')['revenue'].mean()

train['overview'] = train['overview'].map(means)



means = train.groupby('Keywords')['revenue'].mean()

train['Keywords'] = train['Keywords'].map(means)



means = train.groupby('original_language')['revenue'].mean()

train['original_language'] = train['original_language'].map(means)



means = train.groupby('original_title')['revenue'].mean()

train['original_title'] = train['original_title'].map(means)



means = train.groupby('production_countries')['revenue'].mean()

train['production_countries'] = train['production_countries'].map(means)



means = train.groupby('production_companies')['revenue'].mean()

train['production_companies'] = train['production_companies'].map(means)



means = train.groupby('status')['revenue'].mean()

train['status'] = train['status'].map(means)



means = train.groupby('original_title')['revenue'].mean()

train['original_title'] = train['original_title'].map(means)
#Dropping least important features



train = train.drop(['belongs_to_collection', 'genres', 'imdb_id', 'poster_path', 'release_date', 'spoken_languages', 'tagline', 'title'], axis=1)



train = train.astype(np.float64)
#Using TPOT for model selection with Dask support



#client = Client(n_workers=4, threads_per_worker=1)

#client





#trainset_features = train.drop(['revenue'], axis=1).values

#trainset_target = train['revenue'].values



#X_train, X_test, y_train, y_test = train_test_split(trainset_features, trainset_target, train_size=0.75, test_size=0.25)





#tp = TPOTClassifier(generations=5, n_jobs=-1, random_state=0, verbosity=2, use_dask=True)



#tp.fit(X_train, y_train)
trainset_features = train.drop(['revenue'], axis=1)

trainset_target = train['revenue']



X_train, X_test, y_train, y_test = train_test_split(trainset_features, trainset_target,

                                                    train_size=0.75, test_size=0.25)





gbm = lgb.LGBMRegressor(num_leaves=31,

                        learning_rate=0.05,

                        n_estimators=20)

gbm.fit(X_train, y_train,

        eval_set=[(X_test, y_test)],

        eval_metric='l1',

        early_stopping_rounds=5)





y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)



print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)





print('Feature importances:', list(gbm.feature_importances_))



estimator = lgb.LGBMRegressor(num_leaves=31)



param_grid = {

    'learning_rate': [0.01, 0.1, 1],

    'n_estimators': [20, 40]

}



gbm = GridSearchCV(estimator, param_grid, cv=3)

gbm.fit(X_train, y_train)



print('Best parameters found by grid search are:', gbm.best_params_)
#Making validation Regression with best parameters



lr = LGBMRegressor(boosting_type='dart',num_leaves=20,max_depth=-1,min_data_in_leaf=20, learning_rate=0.2,n_estimators=500,subsample_for_bin=200000,

                   class_weight=None,min_split_gain=0.0,min_child_weight=0.001,subsample=0.1,subsample_freq=0,colsample_bytree=0.75,reg_alpha=0.0,reg_lambda=0.0,

                   random_state=101,n_jobs=-1)



lr.fit(X_train, y_train,eval_set=[(X_test, y_test)])

pred = lr.predict(X_test, num_iteration=lr.best_iteration_)
#Submission



#submit = pd.DataFrame({'id': test.id, 'revenue':np.expm1(predict)})

#submit.to_csv('submission.csv', index=False)
