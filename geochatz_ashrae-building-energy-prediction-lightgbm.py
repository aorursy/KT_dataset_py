# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns

sns.set(style="darkgrid")



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot figures

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12



print('Libraries imported.')
## Memory optimization

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin



def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                #    df[col] = df[col].astype(np.float16)

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('object')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
def load_ashrae_energy_data(filename, ashrae_path = '../input/ashrae-energy-prediction/'):

    csv_path = os.path.join(ashrae_path, filename)

    return reduce_mem_usage(pd.read_csv(csv_path))
weather_train = load_ashrae_energy_data('weather_train.csv')
weather_train.head()
weather_train.info()
weather_train.describe()
# missing data

total = weather_train.isnull().sum().sort_values(ascending=False)

percent = (weather_train.isnull().sum()/weather_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
weather_train.drop(['cloud_coverage', 'precip_depth_1_hr'], axis=1, inplace=True)

weather_train.head()
attributes = ["air_temperature", "dew_temperature", "sea_level_pressure", "wind_direction", "wind_speed"]
from sklearn.base import BaseEstimator, TransformerMixin



class LagWeatherFeatureCalculator(BaseEstimator, TransformerMixin):

    def __init__(self, frequency='W', shift=1, attributes=['air_temperature']):

        self.frequency = frequency

        self.shift = shift

        self.attributes = attributes

    

    def fit(self, X, y=None):

        print('LagFeatureCalculator fit')

        return self

    

    def transform(self, X, y=None):

        print('LagFeatureCalculator transform')

        print("Frequency is: {}".format(self.frequency))

        

        X['timestamp'] = pd.to_datetime(X['timestamp'])

        

        frame = X.set_index(keys=['timestamp', 'site_id'])

        frame_shifted = frame[self.attributes].unstack().resample(self.frequency).mean().shift(self.shift,freq=self.frequency).resample('H').ffill()

        

        columns_shifted = [col+'_'+ self.frequency +'%s' % 1 for col in frame.columns]

        frame_shifted.columns.set_levels(columns_shifted, level=0, inplace=True)

        

        return frame.merge(frame_shifted.stack(), left_index=True, right_index=True, how='left').reset_index()
from sklearn.pipeline import Pipeline



lag_pipeline = Pipeline([

    ("hist_D1", LagWeatherFeatureCalculator(frequency='D',shift=1, attributes=attributes)),

    ("hist_W1", LagWeatherFeatureCalculator(frequency='W',shift=1, attributes=attributes)),

])
lag_pipeline.fit_transform(weather_train).tail()
weather_train_full = lag_pipeline.fit_transform(weather_train)

weather_train_full.head()
building_metadata = load_ashrae_energy_data('building_metadata.csv')
building_metadata.head()
building_metadata.info()
# missing data

total = building_metadata.isnull().sum().sort_values(ascending=False)

percent = (building_metadata.isnull().sum()/building_metadata.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
building_metadata.drop(['floor_count', 'year_built'], axis=1, inplace=True)

building_metadata.head()
train = load_ashrae_energy_data('train.csv')
train.info(verbose=True, null_counts=True)
train.head()
train['timestamp'] = pd.to_datetime(train['timestamp'])
ashrae = train.merge(building_metadata, on=['building_id'], how='left').merge(weather_train_full, on=['timestamp', 'site_id'], how='left')

ashrae.head()
ashrae.info()
del building_metadata, train, weather_train, weather_train_full
ashrae['meter_reading_log1p'] = np.log1p(ashrae['meter_reading'])
train_data_meter0_site0 = ashrae.query('meter == 0 & site_id ==0')

train_data_meter0_site0.groupby('timestamp').sum()['meter_reading_log1p'].plot(figsize=(10, 5))

plt.show()
ashrae.drop(['meter_reading_log1p'], axis=1, inplace=True)
train_data_meter0_site0.building_id.sort_values().unique()
print(ashrae.query('site_id==0 & meter==0 & timestamp<="2016-05-20"').shape)

ashrae = ashrae.query('not (site_id==0 & meter==0 & timestamp<="2016-05-20")').reset_index(drop=True)

ashrae.info(verbose=True, null_counts=True)
print(ashrae.query("(meter==0 & meter_reading==0)").shape)

ashrae = ashrae.query("not (meter==0 & meter_reading==0)").reset_index(drop=True)

ashrae.info()
ashrae.query("building_id==1099 and meter==2").set_index(keys=['timestamp'])['meter_reading'].hist()

plt.show()
print(ashrae.query("building_id==1099 & meter==2 & meter_reading>=3e4").shape)

ashrae = ashrae.query("not (building_id==1099 & meter==2 & meter_reading>=3e4)").reset_index(drop=True)

ashrae.info()
del train_data_meter0_site0, missing_data, total, percent
train_data = ashrae.copy()
train_data['meter_reading_log1p'] = np.log1p(train_data['meter_reading'])
train_data[['meter_reading', 'meter_reading_log1p']].describe()
sns.distplot(train_data[['meter_reading']])

plt.show()
#skewness and kurtosis

print("Skewness: %f" % train_data['meter_reading'].skew())

print("Kurtosis: %f" % train_data['meter_reading'].kurt())
sns.distplot(train_data[['meter_reading_log1p']])

plt.show()
#skewness and kurtosis

print("Skewness: %f" % train_data[['meter_reading_log1p']].skew())

print("Kurtosis: %f" % train_data[['meter_reading_log1p']].kurt())
date_attributes = ['timestamp']

train_data[date_attributes].head()
train_data['hour'] = train_data['timestamp'].dt.hour

train_data['weekday'] = train_data['timestamp'].dt.weekday

train_data['month'] = train_data['timestamp'].dt.month

train_data[['timestamp','hour','weekday','month']].head()
date_features = ['hour', 'weekday', 'month']
cat_features = ['meter', 'primary_use', 'site_id']
plt.figure(figsize=(20,5))



ax1=plt.subplot(131)

sns.countplot(x='meter', data=train_data[['meter']].replace(

    {0:'electricity', 1:'chilledwater', 2:'steam', 3:'hotwater'}), ax=ax1)

plt.xlabel('meter')

plt.xticks(rotation=90)



ax2=plt.subplot(132)

sns.countplot(x='primary_use', data=train_data, ax=ax2)

plt.xlabel('primary_use')

plt.xticks(rotation=90)



ax3=plt.subplot(133)

sns.countplot(x='site_id', data=train_data, ax=ax3)

plt.xlabel('site_id')



plt.show()
train_data.info()
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer



from statsmodels.stats.outliers_influence import variance_inflation_factor



class ReduceVIF(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):

        # From looking at documentation, values between 5 and 15 are "okay".

        # Above 10 is too high and so should be removed.

        self.thresh = thresh

        

        # The statsmodel function will fail with NaN values, as such we have to impute them.

        # By default we impute using the median value.

        if impute:

            self.imputer = SimpleImputer(strategy=impute_strategy)



    def fit(self, X, y=None):

        print('ReduceVIF fit')

        print(self.imputer)

        if hasattr(self, 'imputer'):

            self.imputer.fit(X)

        return self



    def transform(self, X, y=None):

        print('ReduceVIF transform')

        columns = X.columns.tolist()

        if hasattr(self, 'imputer'):

            X = pd.DataFrame(self.imputer.transform(X), columns=columns)

        return ReduceVIF.calculate_vif(X, self.thresh)



    @staticmethod

    def calculate_vif(X, thresh=5.0):

        dropped=True

        while dropped:

            variables = X.columns

            dropped = False

            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            

            max_vif = max(vif)

            if max_vif > thresh:

                maxloc = vif.index(max_vif)

                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')

                X = X.drop([X.columns.tolist()[maxloc]], axis=1)

                dropped=True

        print(X.columns)

        return X
np.random.seed(24)

m = 10000000

idx = np.random.permutation(len(train_data))[:m]
#dropper = ReduceVIF(thresh=10)

#num_features = list(dropper.fit_transform(train_data.drop(['meter_reading','meter_reading_log1p'], axis=1).select_dtypes(include=['int32','float32']).iloc[idx,:]).columns.values)

#del dropper



num_features = ['square_feet', 'air_temperature', 

                'dew_temperature', 'wind_direction', 'wind_speed', 'dew_temperature_D1', 'wind_direction_D1', 'wind_speed_D1', 'dew_temperature_W1', 'wind_speed_W1']
num_features
np.random.seed(24)

m = 30000

idx = np.random.permutation(len(train_data))[:m]
sns.pairplot(data=train_data.loc[idx, num_features+['meter_reading','meter_reading_log1p']].dropna())

plt.show()
for feature in (num_features+['meter_reading','meter_reading_log1p']):

    print('Feature:{}\tSkewness:{:.3f}\tKurtosis:{:.3f}'.format(feature, train_data[feature].skew(), train_data[feature].kurt()))

    

transf_features = [feature for feature in num_features if abs(train_data[feature].skew()) > 1 and abs(train_data[feature].kurt()) > 1]

transf_features
non_transf_features = list(set(num_features).difference(set(transf_features)))

non_transf_features
train_data_transform = pd.concat([train_data[non_transf_features + ['meter_reading','meter_reading_log1p']],

                                  train_data[transf_features].apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)))], axis=1)

train_data_transform
for feature in num_features+['meter_reading','meter_reading_log1p']:

    print('Feature:{}\tSkewness:{:.3f}\tKurtosis: {:.3f}'.format(feature, train_data_transform[feature].skew(), train_data_transform[feature].kurt()))
train_data_transform.drop(['meter_reading'], axis=1, inplace=True)

corrmat = train_data_transform.corr()

f, ax = plt.subplots(figsize=(7, 7))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'meter_reading_log1p')['meter_reading_log1p'].index

cm = np.corrcoef(train_data_transform[cols].dropna().values.T)

f, ax = plt.subplots(figsize=(7, 7))

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
del train_data, train_data_transform, cm, corrmat
from sklearn.base import BaseEstimator, TransformerMixin



# A class to select numerical, categorical or datetime columns 

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y=None):

        print(self.attribute_names)

        return self

    

    def transform(self, X):

        return X[self.attribute_names]
cat_features = ['meter', 'primary_use', 'site_id', 'building_id']
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[attr].value_counts().index[0] for attr in X], index=X.columns)

        return self

    

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline



cat_pipeline = Pipeline([

    ("selector", DataFrameSelector(cat_features)),

    ('imputer', CategoricalImputer()),

    #('encoder', OneHotEncoder(sparse=True))

    ("encoder", OrdinalEncoder())

])
date_attributes = ['timestamp']
date_features = ['hour', 'weekday', 'month']



class TimeInfoExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        return np.c_[X['timestamp'].dt.hour.astype(int),

                     X['timestamp'].dt.weekday.astype(int),

                     X['timestamp'].dt.month.astype(int)]
date_pipeline = Pipeline([

    ("selector", DataFrameSelector(date_attributes)),

    ('extractor', TimeInfoExtractor()),

    #('encoder', OneHotEncoder(sparse=True))

    ('encoder', OrdinalEncoder())

])
print(num_features)

print(transf_features)

print(non_transf_features)
transf_features_idx = [num_features.index(elem) for elem in transf_features]

non_transf_features_idx = [num_features.index(elem) for elem in non_transf_features]
print(transf_features_idx)

print(non_transf_features_idx)
class LogModulusTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        return np.c_[X[:,non_transf_features_idx], 

                     np.apply_along_axis(lambda x: np.sign(x) * np.log(1 + np.abs(x)), 1, X[:,transf_features_idx])]

        # return X.apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)))
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ('selector', DataFrameSelector(num_features)),

    ('imputer', SimpleImputer(strategy='mean')),

    ('transformer', LogModulusTransformer()),

    ('scaler', StandardScaler()),

])
train_data = ashrae.copy()
train_data.info()
from sklearn.pipeline import FeatureUnion



preprocess_pipeline = FeatureUnion(transformer_list=[

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline),

    ("date_pipeline", date_pipeline)

])
X = preprocess_pipeline.fit_transform(train_data)

y = train_data['meter_reading'].apply(lambda x: np.log1p(x)).values



del  train_data
print(X.shape, y.shape)
total_features = num_features + cat_features + date_features

print(total_features)



categorical_features = cat_features + date_features

print(categorical_features)
np.random.seed(24)

m = 5000000

idx = np.random.permutation(len(X))[:m]



X_subset = X[idx, :]

y_subset = y[idx]
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)



del X_subset, y_subset



print(X_train.shape, y_train.shape)

print(X_valid.shape, y_valid.shape)
fit_params={"early_stopping_rounds": 5, 

            "eval_metric" : 'rmse', 

            "eval_set" : [(X_valid, y_valid)],

            'eval_names': ['validation'],

            'verbose': 100,

            'categorical_feature': categorical_features, 

            'feature_name':total_features}
from scipy.stats import randint, uniform, expon



params ={'num_leaves': randint(10, 50), 

         'min_child_samples': randint(500, 1000), 

         'colsample_bytree': uniform(loc=0.4, scale=0.6),

         'learning_rate' : [0.001, 0.01, 0.1, 0.9, 1.5],

         'subsample': uniform(loc=0.2, scale=0.8), 

         'colsample_bytree': uniform(loc=0.4, scale=0.6),

         'reg_alpha': expon(scale=1.0)}
import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV



lgb_reg = lgb.LGBMRegressor(max_depth=-1, random_state=42, silent=True, metric='None', n_jobs=4, n_estimators=1000)



random_grd = RandomizedSearchCV(estimator=lgb_reg, param_distributions=params,

                                n_iter=10, scoring='neg_mean_squared_error', cv=3, refit=True, random_state=42, verbose=2)



random_grd.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(random_grd.best_score_, random_grd.best_params_))
optimal_params = random_grd.best_params_

#optimal_params = {'colsample_bytree': 0.7905330837693118, 'learning_rate': 0.9, 'min_child_samples': 757, 'num_leaves': 33, 'reg_alpha': 1.786429543354675, 'subsample': 0.36987128854262097} 
cvres = random_grd.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
# Use the full dataset

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)



del X, y



print(X_train.shape, y_train.shape)

print(X_valid.shape, y_valid.shape)
lgb_opt = lgb.LGBMRegressor(max_depth=-1, random_state=42, silent=True, metric='None', n_jobs=4, n_estimators=1000)



# set optimal parameters

lgb_opt.set_params(**optimal_params)



print(lgb_opt)



fit_params={"early_stopping_rounds": 10, 

            "eval_metric" : 'rmse', 

            "eval_set" : [(X_valid, y_valid)],

            'eval_names': ['validation'],

            'verbose': 100,

            'categorical_feature': categorical_features, 

            'feature_name':total_features}



t0, t1 = 900, 1000

def learning_schedule(t):

    return t0 / (t + t1)



#lgb_opt.fit(X_train, y_train, **fit_params, callbacks=[lgb.reset_parameter(learning_rate=learning_schedule)])

lgb_opt.fit(X_train, y_train, **fit_params)
del X_train, y_train, X_valid, y_valid, ashrae
feature_importance = pd.DataFrame()

feature_importance["features"] = total_features

feature_importance["importance"] = lgb_opt.feature_importances_





plt.figure(figsize=(10, 5))

sns.barplot(x="importance", y="features", data=feature_importance.sort_values(by="importance", ascending=False))

plt.title("LightGBM Feature Importance")

plt.show()
weather_test = load_ashrae_energy_data('weather_test.csv')

weather_test.head()
weather_test.drop(['cloud_coverage', 'precip_depth_1_hr'], axis=1, inplace=True)

weather_test.head()
weather_test_full = lag_pipeline.transform(weather_test)

weather_test_full.head()
building_metadata = load_ashrae_energy_data('building_metadata.csv')
building_metadata.drop(['floor_count', 'year_built'], axis=1, inplace=True)

building_metadata.head()
test = load_ashrae_energy_data('test.csv')

test.head()
test['timestamp'] = pd.to_datetime(test['timestamp'])
test_data = test.merge(building_metadata, on=['building_id'], how='left').merge(weather_test_full, on=['timestamp', 'site_id'], how='left')

test_data.head()
del building_metadata, test, weather_test, weather_test_full
n_instances = len(test_data)

print(n_instances)
batch_size = 100000

n_batches = n_instances // batch_size

n_batches
batches = np.array_split(test_data, n_batches)

del test_data
y_pred =[]

for n, batch in enumerate(batches):

    if n % 50 == 0:

        print("batch number: ", n)

    y_pred.extend(np.expm1(lgb_opt.predict(preprocess_pipeline.transform(batch))))



del batches
y_pred = np.array(y_pred)

print(y_pred.shape)

y_pred.ravel()

print(y_pred)
pd.DataFrame(y_pred).describe()
sample_submission = load_ashrae_energy_data('sample_submission.csv')

sample_submission.head()
submission = sample_submission.copy()

del sample_submission



submission['meter_reading'] = np.clip(y_pred, 0, a_max=None)

submission.head()
submission.to_csv('submission.csv', index=False)