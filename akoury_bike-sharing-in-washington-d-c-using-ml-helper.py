!pip install ml_helper
import warnings

import numpy as np

import pandas as pd

import seaborn as sns

from tempfile import mkdtemp

from sklearn.base import clone

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.cluster import KMeans

from ml_helper.helper import Helper

from imblearn import FunctionSampler

from imblearn.combine import SMOTEENN

from sklearn.decomposition import PCA

from imblearn.pipeline import Pipeline

from sklearn.ensemble import IsolationForest

from sklearn.compose import ColumnTransformer

from gplearn.genetic import SymbolicTransformer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score as metric_scorer

from sklearn.feature_selection import RFE, SelectFromModel

from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, PowerTransformer, OneHotEncoder, FunctionTransformer



warnings.filterwarnings('ignore')
MEMORY = mkdtemp()



KEYS = {

    'SEED': 1,

    'DATA_H': '../input/bike-sharing-dataset/hour.csv',

    'DATA_D' : '../input/bike-sharing-dataset/day.csv',

    'DATA_P': 'https://gist.githubusercontent.com/akoury/6fb1897e44aec81cced8843b920bad78/raw/b1161d2c8989d013d6812b224f028587a327c86d/precipitation.csv',

    'TARGET': 'cnt',

    'METRIC': 'r2',

    'TIMESERIES': True,

    'SPLITS': 3,

    'ESTIMATORS': 150,

    'MEMORY': MEMORY

}



hp = Helper(KEYS)
def read_data(input_path):

    return pd.read_csv(input_path, parse_dates=[1])



data = read_data(KEYS['DATA_H'])

data_daily = read_data(KEYS['DATA_D'])



data.head()
data.describe()
precipitation = read_data(KEYS['DATA_P'])

data = pd.merge(data, precipitation,  how='left', on=['dteday','hr'])

data['precipitation'].fillna(0, inplace=True)

data['precipitation'][data['precipitation'] > 0] = 1

data['precipitation'] = data['precipitation'].astype(int).astype('category')



data_hourly = data.copy()

data_hourly = data_hourly[data_hourly['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))]
data.dtypes
hp.missing_data(data)
data = hp.convert_to_category(data, data.iloc[:,2:10])



data.dtypes
palette_tot_cas_reg = ['darkgreen', 'darkred', 'darkblue']



palette_cas = ['darkred', 'salmon']

palette_reg = ['darkblue', 'skyblue']
plt.figure(figsize=(16, 8))

sns.distplot(data['cnt'])
data_daily = data_daily[data_daily['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))]

data_daily = hp.convert_to_category(data_daily, data_daily.iloc[:,2:9])

data_daily.set_index('dteday')



plt.figure(figsize=(16, 5))



ax = sns.lineplot(data = data_daily, x = 'dteday', y = 'cnt', color='darkgreen', size = 1,label = 'count')

ax = sns.lineplot(data = data_daily, x = 'dteday', y = 'casual', color='darkred', size = 1, label = 'casual')

ax = sns.lineplot(data = data_daily, x = 'dteday', y = 'registered', color='darkblue', size = 1, label = 'registered')



handles, labels = ax.get_legend_handles_labels()

l = plt.legend(handles[0:1]+handles[3:4]+handles[6:7], labels[0:1]+labels[3:4]+labels[6:7], loc=2)

plt.xlabel('Date')

plt.ylabel('Users')
df_month = pd.DataFrame(data_daily.groupby("mnth")[["cnt", 'casual', 'registered']].mean()).reset_index()

months = pd.Series(["January","February","March","April","May","June","July","August","September","October","November","December"]).rename("months")

df_month = pd.concat([df_month, months], axis = 1)





plt.figure(figsize=(12, 5))

ax = sns.pointplot(data = df_month, x = "months", y = "cnt", color = 'darkgreen')

ax = sns.pointplot(data = df_month, x = "months", y = "casual", color = 'darkred')

ax = sns.pointplot(data = df_month, x = "months", y = "registered", color = 'darkblue')



plt.xlabel('')

plt.ylabel('Users')
df_week = pd.DataFrame(data_daily.groupby("weekday")[["cnt", 'casual', 'registered']].mean()).reset_index()

df_week = pd.melt(df_week, id_vars = ['weekday'], value_vars = ['cnt', 'casual', 'registered'], var_name = 'type', value_name = 'users')



plt.figure(figsize=(12, 5))

ax = sns.lineplot(data = df_week, x = "weekday", y = "users", hue = "type", palette = palette_tot_cas_reg)

plt.xlabel('Weekday')

plt.ylabel('Users')
data_hourly = hp.convert_to_category(data_hourly, data_hourly.iloc[:,2:9])

data_hourly.set_index('dteday')



df_day = pd.DataFrame(data_hourly.groupby("hr")[["cnt", 'casual', 'registered']].mean()).reset_index()

df_day = pd.melt(df_day, id_vars = ['hr'], value_vars = ['cnt', 'casual', 'registered'], var_name = 'type', value_name = 'users')



plt.figure(figsize=(12, 5))

sns.lineplot(data = df_day, x = "hr", y = "users", hue = "type", palette = palette_tot_cas_reg)



plt.xlabel('Hour')

plt.ylabel('Users')
plt.figure(figsize=(12, 5))

sns.lineplot(data = data_hourly, x = "hr", y = "casual", hue = 'workingday', palette = palette_cas)

sns.lineplot(data = data_hourly, x = "hr", y = "registered", hue = 'workingday', palette = palette_reg)

plt.xlabel('Hour')

plt.ylabel('Users')
atemp_binned = pd.cut(x = data_hourly['atemp'], bins = 4).rename('atemp_binned')

data_hourly_binned = pd.concat([data_hourly, atemp_binned], axis = 1)



df_day_by_day_atemp = pd.DataFrame(data_hourly_binned.groupby(["hr", "atemp_binned"])[["cnt", 'casual', 'registered']].mean()).reset_index()

df_day_by_day_atemp.head()



plt.figure(figsize=(12, 5))

sns.lineplot(data = df_day_by_day_atemp, x = 'hr', y = 'casual', hue = 'atemp_binned', palette = 'husl')
plt.figure(figsize=(12, 5))

sns.lineplot(data = df_day_by_day_atemp, x = 'hr', y = 'registered', hue = 'atemp_binned', palette = 'husl')
sns.scatterplot(data = data_daily, x = 'atemp', y = 'casual', hue = 'workingday', alpha = .3)

plt.title('Casual Users')
sns.scatterplot(data = data_daily, x = 'atemp', y = 'registered', hue = 'workingday', alpha = .3)

plt.title('Registered Users')
plt.figure(figsize=(12, 5))

ax = sns.lineplot(data = data_hourly, x = "hr", y = "casual", hue = 'precipitation', palette = palette_cas, label = 'casual')



ax = sns.lineplot(data = data_hourly, x = "hr", y = "registered", hue = 'precipitation', palette = palette_reg, label = 'registered')



handles, labels = ax.get_legend_handles_labels()

l = plt.legend(handles[0:2]+handles[5:7], labels[0:2]+labels[5:7], loc=2)

plt.xlabel('Hour')

plt.ylabel('Users')
hp.boxplot(data, ['instant'])
invariant = hp.coefficient_variation(data, threshold = 0.05, exclude = ['instant'])
base_holdout = data[data['dteday'].isin(pd.date_range('2012-10-01','2012-12-31'))].copy()

base_holdout = hp.drop_columns(base_holdout, ['dteday', 'casual', 'registered'])

base_data = data[data['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))].copy()

base_data = hp.drop_columns(data, ['dteday', 'casual', 'registered'])



y, predictions = hp.predict(base_data, base_holdout, LinearRegression())

base_score = metric_scorer(y, predictions)

print('Baseline score: ' + str(base_score))
training_data = data[data['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))].copy()

correlated_cols = hp.correlated(training_data, 0.9)
under_rep = hp.under_represented(data, 0.97)
hp.plot_pca_components(data.drop('dteday', axis=1))
hp.feature_importance(hp.drop_columns(data, ['dteday', 'registered', 'casual']), RandomForestRegressor(n_estimators=KEYS['ESTIMATORS'], random_state = KEYS['SEED'], n_jobs = -1), convert = True)
holdout = data[data['dteday'].isin(pd.date_range('2012-10-01','2012-12-31'))].copy().reset_index()

holdout_final_plots = holdout.copy() # we will use this for plots at the end

train_data = data[data['dteday'].isin(pd.date_range('2011-01-01','2012-09-30'))]
def day(df):

    df = df.copy()

    df['day'] = df['dteday'].dt.day

    df = hp.convert_to_category(df, ['day'])

    

    return df



def drop_features(df, cols):

    return df[df.columns.difference(cols)]



def kmeans(df, clusters = 3):

    clusterer = KMeans(clusters, random_state=KEYS['SEED'])

    cluster_labels = clusterer.fit_predict(df)

    df = np.column_stack([df, cluster_labels])

    

    return df



def outlier_rejection(X, y):

    model = IsolationForest(random_state=KEYS['SEED'], behaviour='new', n_jobs = -1)

    model.fit(X)

    y_pred = model.predict(X)

    

    return X[y_pred == 1], y[y_pred == 1]



num_pipeline = Pipeline([ 

    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True))

])



categorical_pipeline = Pipeline([

    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))

])



pipe = Pipeline([

    ('day', FunctionTransformer(day, validate=False)),

    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),

    ('column_transformer', ColumnTransformer([

        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),

        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),

    ], remainder='passthrough'))

])



models = [

    {'name':'linear_regression', 'model': LinearRegression()},

    {'name':'random_forest', 'model': RandomForestRegressor(n_estimators = KEYS['ESTIMATORS'], random_state = KEYS['SEED'], n_jobs = -1)},

    {'name':'xgb', 'model': XGBRegressor(random_state = KEYS['SEED'])}

]
all_scores = hp.pipeline(train_data, models, pipe)
num_pipeline = Pipeline([ 

    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True)),

    ('binning', KBinsDiscretizer(n_bins = 5, encode = 'onehot-dense')),

    ('polynomial', PolynomialFeatures(degree = 2, include_bias = False)),

])



categorical_pipeline = Pipeline([

    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))

])



pipe = Pipeline([

    ('day', FunctionTransformer(day, validate=False)),

    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),

    ('column_transformer', ColumnTransformer([

        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),

        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),

    ], remainder='passthrough')),

])



all_scores = hp.pipeline(train_data, models, pipe, all_scores)
num_pipeline = Pipeline([ 

    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True)),

])



categorical_pipeline = Pipeline([

    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))

])



pipe = Pipeline([

    ('day', FunctionTransformer(day, validate=False)),

    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),

    ('column_transformer', ColumnTransformer([

        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),

        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),

    ], remainder='passthrough')),

    ('pca', PCA(n_components = 6))

])



all_scores = hp.pipeline(train_data, models, pipe, all_scores)
num_pipeline = Pipeline([ 

    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True)),

])



categorical_pipeline = Pipeline([

    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))

])



pipe = Pipeline([

    ('day', FunctionTransformer(day, validate=False)),

    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),

    ('column_transformer', ColumnTransformer([

        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),

        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),

    ], remainder='passthrough')),

    ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators = KEYS['ESTIMATORS'], random_state = KEYS['SEED'], n_jobs = -1), threshold = 0.005)),

])



all_scores = hp.pipeline(train_data, models, pipe, all_scores)
num_pipeline = Pipeline([ 

    ('power_transformer', PowerTransformer(method='yeo-johnson', standardize = True)),

    ('polynomial', PolynomialFeatures(degree = 2, include_bias = False)),

])



categorical_pipeline = Pipeline([

    ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'))

])



pipe = Pipeline([

    ('day', FunctionTransformer(day, validate=False)),

    ('drop_features', FunctionTransformer(drop_features, kw_args={'cols': ['dteday','casual', 'registered'] + correlated_cols + under_rep}, validate=False)),

    ('column_transformer', ColumnTransformer([

        ('numerical_pipeline', num_pipeline, ['hum', 'temp', 'windspeed']),

        ('categorical_pipeline', categorical_pipeline, ['day', 'hr', 'mnth', 'precipitation', 'season', 'weathersit', 'weekday', 'yr']),

    ], remainder='passthrough')),

    ('outliers', FunctionSampler(func = outlier_rejection)),

])



all_scores = hp.pipeline(train_data, models, pipe, all_scores)
hp.plot_models(all_scores)
hp.show_scores(all_scores, top = True)
rf_grid = {

    'random_forest__criterion': ['mse', 'mae'],

    'random_forest__max_depth': [50, 100],

    'random_forest__min_samples_leaf': [5,10],

    'random_forest__min_samples_split': [10, 20],

    'random_forest__max_leaf_nodes': [None, 80],

}



final_scores, f_pipe = hp.cross_val(train_data, model = clone(hp.top_pipeline(all_scores)), grid = rf_grid)

final_scores
print(f_pipe.best_params_)

final_pipe = f_pipe.best_estimator_
y, predictions = hp.predict(train_data, holdout, clone(hp.top_pipeline(all_scores)))

score = metric_scorer(y, predictions)

score
hp.scatter_predict(y, predictions)
hp.plot_predict(y, predictions, subset = (3*7*24), x_label = 'Hour', y_label = 'Users')
hp.plot_predict(y, predictions, group = 24, x_label = 'Day', y_label = 'Users')