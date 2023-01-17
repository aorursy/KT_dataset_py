import numpy as np

import pandas as pd



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate



from sklearn.feature_selection import RFECV

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2, f_regression



from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



import random

import time

from tqdm import tqdm



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print('train dataset size:', train.shape)

print('test dataset size:', test.shape)

train.sample(4)
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



    

def feature_engineering(df):

    # missing values

    df.runtime.fillna(0, inplace=True)

    df.status.fillna('Released', inplace=True)

    df.release_date.fillna(df.release_date.mode()[0], inplace=True)

    

    # create count features

    df['count_genre'] = df.genres.apply(proc_json_len)

    df['count_country'] = df.production_countries.apply(proc_json_len)

    df['count_company'] = df.production_companies.apply(proc_json_len)

    df['count_splang'] = df.spoken_languages.apply(proc_json_len)

    df['count_cast'] = df.cast.apply(proc_json_len)

    df['count_crew'] = df.crew.apply(proc_json_len)

    df['count_staff'] = df.count_cast + df.count_crew

    df['count_keyword'] = df.Keywords.apply(proc_json_len)

    

    # convert json features

    df.belongs_to_collection = df.belongs_to_collection.apply(lambda x: proc_json(x, 'name'))

    df.genres = df.genres.apply(lambda x: proc_json(x, 'name'))

    df.production_companies = df.production_companies.apply(lambda x: proc_json(x, 'name'))

    df.production_countries = df.production_countries.apply(lambda x: proc_json(x, 'iso_3166_1'))

    df.spoken_languages = df.spoken_languages.apply(lambda x: proc_json(x, 'iso_639_1'))

    df.Keywords = df.Keywords.apply(lambda x: proc_json(x, 'name'))

    

    # create length of text features

    df['len_title'] = df.title.str.len()

    df.len_title.fillna(0, inplace=True)

    df['len_overview'] = df.overview.str.len()

    df.len_overview.fillna(0, inplace=True)

    df['len_tagline'] = df.tagline.str.len()

    df.len_tagline.fillna(0, inplace=True)

    

    # create category code features

    df['code_origlang'] = df.original_language.astype('category').cat.codes

    

    # create date related features

    df.release_date = pd.to_datetime(df.release_date)

    df['release_year'] = df.release_date.dt.year

    df['release_year'] = df.release_year.apply(lambda x: x-100 if x > 2020 else x)

    df['release_month'] = df.release_date.dt.month

    df['release_wday'] = df.release_date.dt.dayofweek



    # create boolean features

    df['in_collection'] = (df.belongs_to_collection != '').astype('uint8')

    df['us_country'] = df.production_countries.str.contains('US').astype('uint8')

    df['en_lang'] = (df.original_language == 'en').astype('uint8')

    df['has_hompage'] = df.homepage.apply(lambda x: 1 if pd.isnull(x) == False else 0)



    # log money values

    if 'revenue' in df.columns:

        df.revenue = np.log1p(df.revenue)

    df.budget = np.log1p(df.budget)

    df.popularity = np.log1p(df.popularity)

    

    return df

train = feature_engineering(train)

test = feature_engineering(test)
train.info()
all_features = train.select_dtypes(include=['int64', 'float64', 'uint8', 'int8']).columns.tolist()

all_features.remove('id')



plt.figure(figsize=(18,18))

correlations = train[all_features].corr()

sns.heatmap(correlations, annot=True, fmt='.2', center=0.0, cmap='RdBu_r')

plt.show()



target = 'revenue'

all_features.remove(target)
def select_model(X, Y):



    best_models = {}

    models = [

        {   'name': 'LinearRegression',

            'estimator': LinearRegression() 

        },

        {   'name': 'KNeighborsRegressor',

            'estimator': KNeighborsRegressor(),

        },

        {   'name': 'RandomForestRegressor',

            'estimator': RandomForestRegressor(),

        },

        {   'name': 'MLPRegressor',

            'estimator': MLPRegressor(),

        },

        {   'name': 'GradientBoostingRegressor',

            'estimator': GradientBoostingRegressor(),

        },

        {   'name': 'XGBoost',

            'estimator': XGBRegressor(),

        },

        {   'name': 'LightGBM',

            'estimator': LGBMRegressor(),

        },

        {   'name': 'CatBoost',

            'estimator': CatBoostRegressor(verbose=False),

        }

        

    ]

    

    for model in tqdm(models):

        start = time.perf_counter()

        grid = GridSearchCV(model['estimator'], param_grid={}, cv=5, scoring = "neg_mean_squared_error", verbose=False, n_jobs=-1)

        grid.fit(X, Y)

        best_models[model['name']] = {'score': grid.best_score_, 'params': grid.best_params_, 'model':model['estimator']}

        run = time.perf_counter() - start

        

    return best_models



models = select_model(train[all_features], train[target])

models
best_model = None

max_score = -100

best_model_name = ''



for m in models:

    if models[m]['score'] > max_score:

        max_score = models[m]['score']

        best_model = models[m]['model']

        best_model_name = m

        

print(best_model_name, max_score)
base_model = XGBRegressor()
corr_features = correlations.loc[correlations.revenue >= 0.1, 'revenue'].sort_values(ascending=False).index.tolist()

corr_features.remove('revenue')

corr_features
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

fit = pca.fit(train[all_features])



print(pca.explained_variance_ratio_)
# the first PC is enough to show the variance.

feature_df = pd.DataFrame({'feature': all_features, 'importance': abs( pca.components_[0])})

feature_df.sort_values(by='importance', ascending=False, inplace=True)



pca_features = feature_df.feature[:15]

pca_features
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



X, y = train[all_features], train[target]



sfs = SFS(estimator=base_model, 

           k_features=(3, 15),

           forward=True, 

           floating=False, 

           scoring='neg_mean_squared_error',

           cv=5)



sfs.fit(X, y, custom_feature_names=all_features)



print('best combination (ACC: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_names_))



fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')

plt.grid()

plt.show()



sfs_features = list(sfs.k_feature_names_)
def get_accuracy(features):

    X, y = train[features], train['revenue']

    

    result = cross_validate(base_model, X, y, cv=5, scoring="neg_mean_squared_error", verbose=False, n_jobs=-1)

    return np.mean(result['test_score'])





best_features = None

best_accuracy = None

best_idx = -1



feature_candidates = [all_features, corr_features, pca_features, sfs_features]

for idx, flist in enumerate(feature_candidates):

    acc = get_accuracy(flist)

    if best_accuracy is None or acc > best_accuracy:

        best_accuracy = acc

        best_features = flist

        best_idx = idx

        

print(best_idx)

print(best_features)

print(best_accuracy)

hyperparameters = {

    'max_depth': range(1, 12, 2),

    'n_estimators': range(90, 201, 10),

    'min_child_weight': range(1, 8, 2),

    'learning_rate': [.05, .1, .15],

}



grid = GridSearchCV(base_model, param_grid=hyperparameters, cv=5, scoring = "neg_mean_squared_error", verbose=True, n_jobs=-1)

grid.fit(train[best_features], train[target])

print('score = {}\nparams={}'.format(grid.best_score_, grid.best_params_))
opt_model = XGBRegressor(learning_rate=0.15, max_depth=3, min_child_weight=5, n_estimators=100)

opt_model.fit(train[best_features], train[target], eval_metric='rmse')

predict = opt_model.predict(test[best_features])



submit = pd.DataFrame({'id': test.id, 'revenue':np.expm1(predict)})

submit.to_csv('submission.csv', index=False)