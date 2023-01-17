import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import datetime

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from collections import Counter

from sklearn.preprocessing import StandardScaler

import os

import xgboost as xgb

from sklearn import model_selection

import ast

from sklearn.preprocessing import LabelEncoder

import time

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn import linear_model
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
#Посмотрим на исходные данные

print(train.shape)

print(test.shape)

train.head()
train.info()

test.info()
#Посмотрим на зависимость некоторых признаков и целевой переменной, как она распределена

features = ['budget', 'popularity', 'runtime', 'revenue']

sns.pairplot(train[features])
#построим матрицу корреляции

sns.heatmap(train[features].corr(), linewidths=.5, cmap="Reds")
#от признаков в денежном выражении возьмем логарифм

train['log_budget'] = np.log1p(train['budget'])

test['log_budget'] = np.log1p(test['budget'])



train['log_revenue'] = np.log1p(train['revenue'])



fig, ax = plt.subplots(figsize = (15, 5))

plt.subplot(1, 3, 1)

plt.title('Распределение бэджета')

sns.distplot(train['log_budget'], color='Green');

plt.subplot(1, 3, 2)

plt.title('Распределение дохода')

sns.distplot(train['log_revenue'], color='Orange');

plt.subplot(1, 3, 3)

plt.title('Зависимость дохода от бюджета')

plt.scatter(train['log_budget'], train['log_revenue'])

#Видим, что многие признаки содержат большое количество пустых значений

fig = plt.figure(figsize=(10, 8))

train.isna().sum().sort_values(ascending=True).plot(kind='barh',colors='LightGreen')
train.head()
#Попробуем из данных, представленных в виде json выделить важные составляющие



#функция преобразования дат, т.к. у всех только две последние цифры

def fix_date(x):

    year = x.split('/')[2]

    if int(year) <= 19:

        return x[:-2] + '20' + year

    else:

        return x[:-2] + '19' + year



df = pd.concat([train, test]).reset_index(drop = True)

print('All data shape')

print(df.shape)



json_features=["belongs_to_collection", "genres", "production_companies", "production_countries", "spoken_languages"]



for feature in json_features:

    df.loc[df[feature].notnull(),feature]=df.loc[df[feature].notnull(),feature].apply(lambda x : ast.literal_eval(x)).apply(lambda x : [y["name"] for y in x])



df["in_collection"]=1

df.loc[df["belongs_to_collection"].isnull(),"in_collection"]=0

df["genres_len"]=df.loc[df["genres"].notnull(),"genres"].apply(lambda x : len(x))

df["production_companies_len"]=df.loc[df["production_companies"].notnull(),"production_companies"].apply(lambda x : len(x))

df["production_countries_len"]=df.loc[df["production_countries"].notnull(),"production_countries"].apply(lambda x : len(x))

df["spoken_languages_len"]=df.loc[df["spoken_languages"].notnull(),"spoken_languages"].apply(lambda x : len(x))



df.loc[df["cast"].notnull(),"cast"]=df.loc[df["cast"].notnull(),"cast"].apply(lambda x : ast.literal_eval(x))

df.loc[df["crew"].notnull(),"crew"]=df.loc[df["crew"].notnull(),"crew"].apply(lambda x : ast.literal_eval(x))

df["cast_len"] = df.loc[df["cast"].notnull(),"cast"].apply(lambda x : len(x))

df["crew_len"] = df.loc[df["crew"].notnull(),"crew"].apply(lambda x : len(x))



df.loc[df["homepage"].notnull(),"homepage"]=1

df["homepage"]=df["homepage"].fillna(0)



df["has_tagline"]=1

df.loc[df["tagline"].isnull(),"has_tagline"]=0



df["title_different"]=1

df.loc[df["title"]==df["original_title"],"title_different"]=0



df.loc[df["release_date"].notnull(),"release_date"]=df.loc[df["release_date"].notnull(),"release_date"].apply(lambda x : fix_date(x))

release_date=pd.to_datetime(df["release_date"])

df["release_year"]=release_date.dt.year

df["release_month"]=release_date.dt.month

df["release_day"]=release_date.dt.day

df["release_wd"]=release_date.dt.dayofweek

df["release_quarter"]=release_date.dt.quarter



#кодируем строковый атрибут как число

encoder = LabelEncoder()

encoder.fit(list(df['original_language'].fillna('')))

df['original_language'] = encoder.transform(df['original_language'].fillna('').astype(str))



df['log_popularity']=np.log1p(df['popularity'])



#заполним пропуски

df.fillna(value=0.0, inplace = True)



train = df.loc[:train.shape[0] - 1,:]

test = df.loc[train.shape[0]:,:]

print(train.shape)

print(test.shape)
train.head()
#Таким образом, скорее всего признаки, где большая часть данных пустая не повлияют на результат - удалим их

#Также видим, что есть очевидные бесполезные признаки, как например imdb_id или status

useless_features = ['belongs_to_collection', 'homepage', 'tagline', 

                    'Keywords', 'id', 'imdb_id', 'status', 'poster_path', 

                    'title', 'original_title', 'genres', 'production_companies', 

                    'production_countries', 'spoken_languages', 'cast', 

                    'crew', 'release_date', 'overview', 'budget', 'popularity']

train = train.drop(useless_features, axis=1)

train = train.drop('revenue', axis=1)

test = test.drop(useless_features + ['revenue', 'log_revenue'], axis=1)

print(train.shape)

print(test.shape)
train.head()
test.head()
#сформируем выборку данных для обучения модели

X = train.drop(['log_revenue'], axis=1)

y = train['log_revenue']
print(X.shape)

print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12, shuffle=False)

print('Train data shape')

print(X_train.shape)

print('Test data shape')

print(X_test.shape)
def rmsle(y_test, y_pred):

    return np.sqrt(mean_squared_error(y_test, y_pred))



def predict(model):

    model.fit(X_train.values, y_train)

    y_pred = model.predict(X_test.values)

    print(rmsle(y_test, y_pred))

    return y_pred

    

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)

    rmse = np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))

    return(rmse)



def eval_model(model, name):

    start_time = time.time()

    score = rmsle_cv(model)

    print("{} score: {:.4f} ({:.4f}),     execution time: {:.1f}".format(name, score.mean(), score.std(), time.time()-start_time))
mod_xgb = xgb.XGBRegressor(objective  = 'reg:linear', 

          eta = 0.01, 

          max_depth = 6,

          min_child_weight = 3,

          subsample = 0.8, 

          colsample_bytree = 0.8,

          colsample_bylevel = 0.50, 

          gamma = 0.1, 

          eval_metric = 'rmse',

          seed = 12, n_estimators = 2000)

eval_model(mod_xgb, "xgb")
train_pred = predict(mod_xgb)
plt.figure(figsize=(30,10))

plt.plot(np.array(y_test[:100]),label="Реальная")

plt.plot(train_pred[:100],label="Предсказанная")

plt.legend(fontsize=15)

plt.title("Значения предсказанной и реальной выручки",fontsize=24)

plt.show()
#Пройдемся еще раз алгоритмом XGBoost с использованием интерфейса враппера

params = {'objective': 'reg:linear', 

          'eta': 0.01, 

          'max_depth': 6, 

          'min_child_weight': 3,

          'subsample': 0.8,

          'colsample_bytree': 0.8,

          'colsample_bylevel': 0.50, 

          'gamma': 0.1, 

          'eval_metric': 'rmse', 

          'seed': 12, 

          'silent': True    

}

xgb_data = [(xgb.DMatrix(X_train, y_train), 'train'), (xgb.DMatrix(X_test, y_test), 'valid')]

mod_xgb_base = xgb.train(params, 

                  xgb.DMatrix(X_train, y_train),

                  5000,  

                  xgb_data, 

                  verbose_eval=200,

                  early_stopping_rounds=200)
train_pred = mod_xgb_base.predict(xgb.DMatrix(X_test), ntree_limit=mod_xgb_base.best_ntree_limit)

plt.figure(figsize=(30,10))

plt.plot(np.array(y_test[:100]),label="Реальная")

plt.plot(train_pred[:100],label="Предсказанная")

plt.legend(fontsize=15)

plt.title("Значения предсказанной и реальной выручки",fontsize=24)

plt.show()
fig, ax = plt.subplots(figsize=(20,12))

xgb.plot_importance(mod_xgb_base, max_num_features=40, height = 0.5, ax = ax)

plt.title('XGBOOST распределение самых важных фич')

plt.show()
nr_cv = 5

linreg = LinearRegression()

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

grid_linear = GridSearchCV(linreg, parameters, cv=nr_cv, verbose=1 , scoring = "neg_mean_squared_error")

grid_linear.fit(X, y)



print(grid_linear.best_params_)

print(grid_linear.best_estimator_)



linreg = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)

eval_model(linreg, 'Linear Regression')
train_pred = predict(linreg)
plt.figure(figsize=(30,10))

plt.plot(np.array(y_test[:100]),label="Реальная")

plt.plot(train_pred[:100],label="Предсказанная")

plt.legend(fontsize=15)

plt.title("Значения предсказанной и реальной выручки",fontsize=24)

plt.show()