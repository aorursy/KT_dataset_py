import numpy as np

import pandas as pd 



import pickle

from collections import namedtuple, defaultdict

import datetime

from math import sin, cos, sqrt, atan2, radians

from lightgbm import LGBMClassifier

from tqdm import tqdm_notebook

import folium

%matplotlib inline

import sklearn

import scipy.sparse

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from multiprocessing import Pool

import lightgbm as lgb

from sklearn.model_selection import KFold

from itertools import product

import warnings 

warnings.simplefilter('ignore')
Y_train =  pd.read_csv('../input/ozonmasters-ml2-2020-c1/1_data/train_target.csv')

X_train = pd.read_csv('../input/ozonmasters-ml2-2020-c1/1_data/train_data.csv', parse_dates=['due'])

X_test = pd.read_csv('../input/ozonmasters-ml2-2020-c1/1_data/test_data.csv', parse_dates=['due'])
X_train['target'] = Y_train.astype(int)

X_test['target'] = np.nan

X_train['isTrain'] = True

X_test['isTrain'] = False



X = pd.concat([X_train, X_test])



X['old_index'] = X.index 

X.reset_index(drop=True, inplace=True)
cities = pd.DataFrame([['moscow', 55.755814, 37.617635],

                       ['kazan',  55.796289, 49.108795],

                       ['nnovgorod',  56.326797, 44.006516],

                       ['spb',  59.939095, 30.315868],

                       ['voronezh',  51.660781, 39.200269]], columns = ['city', 'lat', 'lon'] ) 



NN = NearestNeighbors(n_neighbors=5, metric='euclidean', n_jobs=-1)

NN.fit(cities[['lat', 'lon']])
def get_city_features(X):

    dist, neighbours = NN.kneighbors(X[['lat', 'lon']])

    X['centr_dist'] = dist[:, 0]

    X['city'] = neighbours[:, 0]

    X['city'] = X['city'].map(cities.city)
get_city_features(X)
X['time'] = X.due.dt.floor("H")
def add_weather(X):

    result = []

    for city in ['moscow', 'kazan', 'nnovgorod', 'spb', 'voronezh']:

        city_weather = pd.read_json(f'../input/ozonmasters-ml2-2020-c1/weather_data/weather_data/group-city-{city}.jsonl', lines=True)

        city_weather.time = pd.to_datetime(city_weather.time)

        city_weather['city'] = city 

        result.append(city_weather)

    result = pd.concat(result)

    return pd.merge(X, result, on=['city', 'time'], how='left')    
X = add_weather(X)
def get_time_features(df):

    df['date'] = df.due.dt.date

    df['dow'] = df.due.dt.dayofweek

    df['hour'] = df.due.dt.hour

    df['minute'] = df.due.dt.minute

    df['second'] = df.due.dt.second
get_time_features(X)  
def get_atipic_hour(X):

    

    # количество вызовов в данный день и час в данном городе 

    ctc = X.sort_values(by=['due']).groupby(['city', 'time']).due.count().rename('counts_per_hour').reset_index()



    

    # cреднее количество вызовов в данный час в данном городе 

    ctc['hour'] = ctc.time.dt.hour

    ctc['mean_count'] = ctc.groupby(['city', 'hour']).counts_per_hour.transform('mean')



    ctc['atipic_hour'] = ctc.counts_per_hour/ctc.mean_count



    return pd.merge(X, ctc[['city', 'time', 'atipic_hour']], on = ['city', 'time'], how = 'left')

    
X = get_atipic_hour(X)
def get_atipic_day(X):

    

    # количество вызовов в данный день и час в данном городе 

    ctc = X.sort_values(by=['due']).groupby(['city', 'date']).due.count().rename('counts_per_day').reset_index()



    

    # cреднее количество вызовов в данный час в данном городе 

    ctc['mean_count'] = ctc.groupby('city').counts_per_day.transform('mean')



    ctc['atipic_day'] = ctc.counts_per_day/ctc.mean_count



    return pd.merge(X, ctc[['city', 'date', 'atipic_day', 'counts_per_day']], on = ['city', 'date'], how = 'left')
X = get_atipic_day(X)
lags = [1, 7]
def lag_feature(df, lags):

    for i in lags:

        shifted = df[['city', 'date', 'counts_per_day']].drop_duplicates()

        feature = 'ratio_counts_lag_'+str(i)

        shifted.columns = ['city', 'date', feature]   

        shifted['date'] = shifted['date'].apply(lambda x: x + datetime.timedelta(days=i))

        df = pd.merge(df, shifted, on=['city', 'date'], how='left')

        df[feature] = df['counts_per_day']/df[feature] 

    return df
X = lag_feature(X, lags)
X_train = X[X.isTrain] 

X_test = X[~X.isTrain] 
knc = KNeighborsClassifier(metric='euclidean', n_neighbors=100)
prediction  = cross_val_predict(knc, X_train[['lat', 'lon']], X_train.target, cv=10, method='predict_proba')
X_train['knn_prediction'] = prediction[:,0]
n = round(X_train.shape[0] * 0.9)

X_train_sample = X_train.sample(n, random_state=42)

knc.fit(X_train_sample[['lat', 'lon']], X_train_sample.target)
test_prediction = knc.predict_proba(X_test[['lat', 'lon']])



X_test['knn_prediction'] = test_prediction[:,0]
class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):

    ''' 

        Этот класс реализует создание KNN признаков.

    '''



    def __init__(self, k_list, metric, n_jobs = 4,  n_classes=None, n_neighbors=None, eps=1e-10):

        self.n_jobs = n_jobs

        self.k_list = k_list

        self.metric = metric

        self.n_neighbors = n_neighbors or max(k_list)

        self.eps = eps

        self.n_classes_ = n_classes



    def fit(self, X, y):

        # Создание объекта-классификатора  

        self.NN = NearestNeighbors(n_neighbors=max(self.k_list),

                                   metric=self.metric,

                                   n_jobs=-1,

                                   algorithm='brute' if self.metric == 'cosine' else 'auto')

        self.NN.fit(X)



        # Сохраниение меток 

        self.y_train = y.values



        # Определение количества классов 

        self.n_classes = len(np.unique(y)) if self.n_classes_ is None else self.n_classes_



    def predict(self, X):

        '''

            Создание признаков для каждого объекта в наборе данных

        '''

        result = []

        for k in self.k_list:

            

            neighs_dist, neighs = self.NN.kneighbors(X)

            neighs_dist, neighs = neighs_dist[:, :k], neighs[:, :k] 



            neighs_y = self.y_train[neighs]



            # 1. Доля объектов каждого класса среди ближайших соседей

            fraction = np.mean(neighs_y, axis = 1)



            # 2. Минимальная дистанция до объектов каждого из классов



            # где y=1  не трогаем значение дистанции, где y=0 прибалвяем к дистанции np.inf

            ones = np.min(neighs_dist + np.where(neighs_y, 0, np.inf), axis =1)

            zeros = np.min(neighs_dist + np.where(neighs_y, np.inf, 0), axis =1)



            # 3. Средняя дистанция 



            mean_distance = np.median(neighs_dist, axis=1)



            # 4. Минимальная дистанция до объектов каждого класса деленная на расстояние до среднего объекта



            norm_ones = ones/(mean_distance + self.eps)



            norm_zeros = zeros/(mean_distance + self.eps)



            # 6. Средняя дистанция до объекта каждого класса из k ближайших соседей



            ones_mean = (np.sum(neighs_dist*neighs_y, axis=1) + self.eps) / np.sum(neighs_y, axis=1)



            mask = 1 * ~neighs_y.astype(bool)



            zeros_mean = (np.sum(neighs_dist*mask, axis=1) + self.eps) / np.sum(mask, axis=1)

            

            column_names = ['fraction_ones', 'min_distance_one', 'min_distance_zero',

                            'mean_distance', 'norm_min_distance_one', 'norm_min_distance_zero',

                            'mean_distance_one', 'min_distance_zero']

            

            result.append(pd.DataFrame(data = np.c_[[fraction, ones, zeros, mean_distance,

                                                      norm_ones, norm_zeros, ones_mean, zeros_mean]].T, 

                                        columns = column_names, 

                                        index = X.index).add_suffix(f'_{k}'))



        return pd.concat(result, axis=1)
nnf = NearestNeighborsFeats(n_jobs = 10, k_list=[10, 50], metric='euclidean')
prediction = cross_val_predict(nnf, X_train[['lat', 'lon']], X_train.target, cv=10, method='predict')
X_train = pd.concat([X_train, pd.DataFrame(prediction, index=X_train.index).add_prefix('nnf_')], axis=1)
n = round(X_train.shape[0]*0.9)



X_train_sample = X_train.sample(n, random_state=42)



nnf.fit(X_train_sample[['lat', 'lon']], X_train_sample.target)
prediction_test = nnf.predict(X_test[['lat', 'lon']])



X_test = pd.concat([X_test, pd.DataFrame(prediction_test.values, index=X_test.index).add_prefix('nnf_')], axis=1)
sample = X_train.sample(3000)

Map = folium.Map(location=(55.751244, 37.618423), zoom_start=12)

for lat, long, target in zip(sample.lat, sample.lon, sample.target):

    folium.Circle((lat, long),

                   radius=5,

                   color='blue' if target else 'red',

                   fill_color='#3186cc',

                   ).add_to(Map)
Map
class ThrColumnEncoder:

    def __init__(self, thr=0.5):

        self.thr = thr

        self.categories = defaultdict(lambda: -1)

    def fit(self, x):

        values = x.value_counts(dropna=False)

        values = values*100/len(x) if self.thr < 1 else values

        for value, key in enumerate(values[values >= self.thr].index):

            self.categories[key] = value

        for value, key in enumerate(values[values < self.thr].index):

            self.categories[key] = -1

            

    def transform(self, x):   

        return x.apply(self.categories.get)

    

    def fit_transform(self, x):

        self.fit(x)

        return self.transform(x)
class ThrLabelEncoder:

    """

    Работает с pd.DataFrame.

    """

    def __init__(self, thr=0.5):

        self.thr = thr

        self.column_encoders = {}

        self.features = None

        

    def fit(self, X, features):

        self.features = features

        for feature in self.features:

            ce = ThrColumnEncoder(thr=self.thr)

            ce.fit(X.loc[:, feature])

            self.column_encoders[feature] = ce

            

    def transform(self, X):

        for feature in self.features: 

            ce = self.column_encoders[feature]

            X.loc[:, feature] = ce.transform(X[feature]).values



    def fit_transform(self, X, features):

        self.features = features

        self.fit(X, features)

        self.transform(X)
cat_features = ['f_class', 's_class', 't_class', 'city', 'summary', 'icon', 'precip_type'] 
tle = ThrLabelEncoder()



tle.fit_transform(X_train, features=cat_features)



tle.transform(X_test)
class MetaFeatureConstructor:

    def __init__(self, n_folds=10, model=None, eval_metric='auc', early_stopping_rounds=100):

        self.random_state = np.random.randint(1e7)

        self.n_folds = n_folds

        self.model = model

        self.eval_metric = eval_metric

        self.early_stopping_rounds = early_stopping_rounds

        self.kf = StratifiedKFold(

            n_folds, random_state=self.random_state, shuffle=True)

        self.models = list()

        self.feature_importance_ = pd.DataFrame()



    def cross_val_predict(self, X, Y):

        result = []

        # осторожно стратификация по выбросам

        for train_index, test_index in tqdm_notebook(self.kf.split(X, Y)):

            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]

            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            train_data = lgb.Dataset(X_train, label=Y_train)

            valid_data = lgb.Dataset(X_test, label=Y_test)

            one_fold_model = lgb.train(params = self.model.get_params(), train_set = train_data, num_boost_round =10000,

                                       valid_sets=[train_data, valid_data],

                                       verbose_eval=100, early_stopping_rounds=self.early_stopping_rounds)

            Y_predict = pd.Series(

                one_fold_model.predict(X_test), index=Y_test.index)



            fold_importance = pd.DataFrame(np.c_[np.array(X.columns), one_fold_model.feature_importance()],

                                           columns=['feature', 'importance'])

            self.feature_importance_ = pd.concat(

                [self.feature_importance_, fold_importance], axis=0)



            result.append(Y_predict[:])

            self.models.append(one_fold_model)



        return pd.concat(result, sort=False, axis=0)



    def predict(self, X_test):

        result = pd.concat([pd.Series(model.predict(

            X_test), index=X_test.index) for model in self.models], axis=1)

        return result.mean(axis=1)



    @property

    def feature_importance(self):

        return self.feature_importance_.groupby("feature").sum().reset_index()



    @property

    def scores(self):

        return {'training': [model.best_score['training'][self.eval_metric] for model in self.models],

                'valid': [model.best_score['valid_1'][self.eval_metric] for model in self.models]

                }
params = {'task': 'train',

          'boosting': 'gbdt',

          'n_estimators': 10000,

          'class_weight': 'balanced',

          'metric': {'logloss', 'auc'},

          'learning_rate': 0.1,

          'feature_fraction': 0.66, 

          'max_depth': 8,

          'num_leaves': 64,

          'min_data_in_leaf': 25,

          'verbose': -1,

          'seed': None,

          'bagging_seed': None,

          'drop_seed': None

          }
X_train.columns
train_columns = ['dist',  'f_class', 'lat', 'lon', 's_class', 't_class', 

                 'centr_dist', 'city', 'temperature', 'apparent_temperature',

                 'wind_speed', 'wind_gust', 'wind_bearing', 'cloud_cover',  'visibility',

                 'dow', 'hour', 'minute', 'second', 'atipic_hour', 'atipic_day',

                 'counts_per_day', 'ratio_counts_lag_1', 'ratio_counts_lag_7',

                 'knn_prediction', 'nnf_0', 'nnf_1', 'nnf_2', 'nnf_3', 'nnf_4', 'nnf_5',

                 'nnf_6', 'nnf_7', 'nnf_8', 'nnf_9', 'nnf_10', 'nnf_11', 'nnf_12',

                  'nnf_13', 'nnf_14', 'nnf_15']
mfc = MetaFeatureConstructor(model=lgb.LGBMClassifier(**params), n_folds=5)



prediction = mfc.cross_val_predict(X_train.loc[:, train_columns], X_train.loc[:, 'target'])
mfc.feature_importance.sort_values('importance')
mfc.scores
y_train_prediction = mfc.predict(X_test.loc[:, train_columns])
X_test.index = X_test.old_index
X_test['target'] = y_train_prediction.values
target = pd.read_csv('../input/ozonmasters-ml2-2020-c1/1_data/sample_submission.csv')



target.target = X_test['target'] 



target.to_csv('weather_5folds.csv', index=False)