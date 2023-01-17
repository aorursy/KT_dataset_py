# for data manipulation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# pandas options

pd.set_option('display.max_columns', 50)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.set_option('mode.use_inf_as_na', True)

pd.options.mode.chained_assignment = None



# for date manipulation

from datetime import datetime



# for visualization: matplotlib

from matplotlib import pyplot as plt

from IPython.core.pylabtools import figsize

%matplotlib inline

# to display visuals in the notebook



# for visualization: seaborn

import seaborn as sns

sns.set_context(font_scale=2)



# for feature engineering: itertools

from itertools import combinations



# for data preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler



# for building the model and calculate RMSE

import lightgbm as lgb

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt



# for hyperparamter optimization and the cross-validation search

from hyperopt import hp, tpe

from hyperopt.fmin import fmin

from sklearn.model_selection import KFold

# from sklearn.model_selection import cross_val_score

# from sklearn.model_selection import TimeSeriesSplit



# for model explainability

import shap



# to cleanup memory usage

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load train  data

building = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")

weather_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")



# drop floor_count

building.drop(columns=["floor_count"], inplace=True)



# convert timestamp column of weather_train and train

train["timestamp"] = pd.to_datetime(train["timestamp"],

                                   format='%Y-%m-%d %H:%M:%S')

weather_train["timestamp"] = pd.to_datetime(weather_train["timestamp"],

                                            format='%Y-%m-%d %H:%M:%S')
## Function to reduce the DF size

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

def reduce_memory_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
reduce_memory_usage(building)

reduce_memory_usage(weather_train)

reduce_memory_usage(train)
# add building age column

current_year = datetime.now().year

building['building_age'] = current_year - building['year_built']

building.drop(columns=['year_built'], inplace=True)



# since NA values only present in building age fillna can be used

building.fillna(round(building.building_age.mean(),0),

                inplace=True)
# check if any NA values left

building.isna().sum()
# create label encoder object and transform the column

le = LabelEncoder()

le_primary_use = le.fit_transform(building.primary_use)



# add label encoded column to dataframe

building['primary_use'] = le_primary_use



del le, le_primary_use

gc.collect()
# add month, day of week, day of month and hour

weather_train['month'] = weather_train['timestamp'].dt.month.astype(np.int8)

weather_train['day_of_week'] = weather_train['timestamp'].dt.dayofweek.astype(np.int8)

weather_train['day_of_month']= weather_train['timestamp'].dt.day.astype(np.int8)

weather_train['hour'] = weather_train['timestamp'].dt.hour



# add is weekend column

weather_train['is_weekend'] = weather_train.day_of_week.apply(lambda x: 1 if x>=5 else 0)
def convert_season(month):

    if (month <= 2) | (month == 12):

        return 0

    # as winter

    elif month <= 5:

        return 1

    # as spring

    elif month <= 8:

        return 2

    # as summer

    elif month <= 11:

        return 3

    # as fall
weather_train['season'] = weather_train.month.apply(convert_season)
weather_train = weather_train.set_index(

    ['site_id','day_of_month','month'])
# create dataframe of daily means per site id

air_temperature_filler = pd.DataFrame(weather_train

                                      .groupby(['site_id','day_of_month','month'])

                                      ['air_temperature'].mean(),

                                      columns=["air_temperature"])

air_temperature_filler.isna().sum()
# create dataframe of air_temperatures to fill

temporary_df = pd.DataFrame({'air_temperature' : weather_train.air_temperature})



# update NA air_temperature values

temporary_df.update(air_temperature_filler, overwrite=False)



# update in the weather train dataset

weather_train["air_temperature"] = temporary_df["air_temperature"]



del temporary_df, air_temperature_filler

gc.collect()
# create dataframe of daily means per site id

cloud_coverage_filler = pd.DataFrame(weather_train

                                     .groupby(['site_id','day_of_month','month'])

                                     ['cloud_coverage'].mean(),

                                     columns = ['cloud_coverage'])

cloud_coverage_filler.isna().sum()
round(cloud_coverage_filler.cloud_coverage.mean(),0)
cloud_coverage_filler.fillna(round(cloud_coverage_filler.cloud_coverage.mean(),0), 

                             inplace=True)



# create dataframe of cloud_coverages to fill

temporary_df = pd.DataFrame({'cloud_coverage' : weather_train.cloud_coverage})



# update NA cloud_coverage values

temporary_df.update(cloud_coverage_filler, overwrite=False)



# update in the weather train dataset

weather_train["cloud_coverage"] = temporary_df["cloud_coverage"]



del temporary_df, cloud_coverage_filler

gc.collect()
# create dataframe of daily means per site id

dew_temperature_filler = pd.DataFrame(weather_train

                                      .groupby(['site_id','day_of_month','month'])

                                      ['dew_temperature'].mean(),

                                      columns=["dew_temperature"])

dew_temperature_filler.isna().sum()
# create dataframe of dew_temperatures to fill

temporary_df = pd.DataFrame({'dew_temperature' : weather_train.dew_temperature})



# update NA dew_temperature values

temporary_df.update(dew_temperature_filler, overwrite=False)



# update in the weather train dataset

weather_train["dew_temperature"] = temporary_df["dew_temperature"]



del temporary_df, dew_temperature_filler

gc.collect()
# create dataframe of daily means per site id

precip_depth_filler = pd.DataFrame(weather_train

                                   .groupby(['site_id','day_of_month','month'])

                                   ['precip_depth_1_hr'].mean(),

                                   columns=['precip_depth_1_hr'])

precip_depth_filler.isna().sum()
round(precip_depth_filler['precip_depth_1_hr'].mean(),0)
precip_depth_filler.fillna(round(precip_depth_filler['precip_depth_1_hr'].mean(),0)

                           , inplace=True)



# create dataframe of precip_depth_1_hr to fill

temporary_df = pd.DataFrame({'precip_depth_1_hr' : weather_train.precip_depth_1_hr})



# update NA precip_depth_1_hr values

temporary_df.update(precip_depth_filler, overwrite=False)



# update in the weather train dataset

weather_train["precip_depth_1_hr"] = temporary_df["precip_depth_1_hr"]



del precip_depth_filler, temporary_df

gc.collect()
# create dataframe of daily means per site id

sea_level_filler = pd.DataFrame(weather_train

                                .groupby(['site_id','day_of_month','month'])

                                ['sea_level_pressure'].mean(),

                                columns=['sea_level_pressure'])

sea_level_filler.isna().sum()
mean_sea_level_pressure = round(

    sea_level_filler

    ['sea_level_pressure']

    .astype(float)

    .mean(),2)
sea_level_filler.fillna(mean_sea_level_pressure, inplace=True)



# create dataframe of sea_level_pressure to fill

temporary_df = pd.DataFrame({'sea_level_pressure' : weather_train.sea_level_pressure})



# update NA sea_level_pressure values

temporary_df.update(sea_level_filler, overwrite=False)



# update in the weather train dataset

weather_train["sea_level_pressure"] = temporary_df["sea_level_pressure"]



del sea_level_filler, temporary_df

gc.collect()
# create dataframe of daily means per site id

wind_direction_filler = pd.DataFrame(weather_train

                                     .groupby(['site_id','day_of_month','month'])

                                     ['wind_direction'].mean(),

                                     columns=['wind_direction'])

wind_direction_filler.isna().sum()
# create dataframe of wind_direction to fill

temporary_df = pd.DataFrame({'wind_direction' : weather_train.wind_direction})



# update NA wind_direction values

temporary_df.update(wind_direction_filler, overwrite=False)



# update in the weather train dataset

weather_train["wind_direction"] = temporary_df["wind_direction"]



del temporary_df, wind_direction_filler

gc.collect()
# create dataframe of daily means per site id

wind_speed_filler = pd.DataFrame(weather_train

                                 .groupby(['site_id','day_of_month','month'])

                                 ['wind_speed'].mean(),

                                 columns=['wind_speed'])

wind_speed_filler.isna().sum()
# create dataframe of wind_speed to fill

temporary_df = pd.DataFrame({'wind_speed' : weather_train.wind_speed})



# update NA wind_speed values

temporary_df.update(wind_speed_filler, overwrite=False)



# update in the weather train dataset

weather_train["wind_speed"] = temporary_df["wind_speed"]



del temporary_df, wind_speed_filler

gc.collect()
# check if NA values left

weather_train.isna().sum()
weather_train = weather_train.reset_index()
weather_train.head()
def convert_direction(degrees):

    if degrees <= 90:

        return 0

    # as norteast direction

    elif degrees <= 180:

        return 1

    # as southeast direction

    elif degrees <= 270:

        return 2

    # as southwest direction

    elif degrees <= 360:

        return 3

    # as northwest direction
weather_train['wind_compass_direction'] = weather_train.wind_direction.apply(convert_direction)

weather_train.drop(columns=['wind_direction'], inplace=True)
# create list of weather variables

weather_variables = ["air_temperature", "cloud_coverage", "dew_temperature",

                    "precip_depth_1_hr", "sea_level_pressure", "wind_speed"]
for i, j in combinations(weather_variables, 2):

    train["mean" + i + "_" + j] = (weather_train[i] + weather_train[j]) / 2
# merge dataframes on train dataframe

train = train.merge(building, on = "building_id", how = "left")

train = train.merge(weather_train, on = ["site_id", "timestamp"], how='left')
del building

del weather_train

del weather_variables

gc.collect()
print("Number of unique columns in the train dataset:", train.shape[1])
train.isna().sum()
correlations_transformed = pd.DataFrame(train.corr())

correlations_transformed = pd.DataFrame(correlations_transformed["meter_reading"]).reset_index()



# format, and display sorted correlations_transformed

correlations_transformed.columns = ["Feature", "Correlation with meter_reading"]

corr_df = (correlations_transformed[correlations_transformed["Feature"] != "meter_reading"]

                .sort_values(by="Correlation with meter_reading", ascending=True))

corr_df
del corr_df, correlations_transformed

gc.collect()
# add log_meter_reading column to the dataframe

train['log_meter_reading'] = np.log1p(train.meter_reading)
correlations_transformed = pd.DataFrame(train.corr())

correlations_transformed = pd.DataFrame(correlations_transformed["log_meter_reading"]).reset_index()



# format, and display sorted correlations_transformed

correlations_transformed.columns = ["Feature", "Correlation with log_meter_reading"]

corr_df = (correlations_transformed[correlations_transformed["Feature"] != "log_meter_reading"]

                .sort_values(by="Correlation with log_meter_reading", ascending=True))

corr_df
initial_feature_list = (corr_df[

    (corr_df["Correlation with log_meter_reading"] >= 0.004) | 

    (corr_df["Correlation with log_meter_reading"] <= -0.004)]["Feature"].

                        to_list())
#original_feature_set = ['building_age', 'le_primary_use', 'cloud_coverage', 

#                        'is_weekend','wind_speed', 'day_of_week',

#                        'wind_compass_direction', 'sea_level_pressure', 'air_temperature',

#                        'day_of_month', 'dew_temperature', 'hour', 

#                        'month', 'meter', 'building_id', 

#                        'site_id', 'square_feet']
# we included meter_reading in the initial feature set

# which is not a feature

# replace meter_reading with precip_depth_1_hr

for n, i in enumerate(initial_feature_list):

    if i == "meter_reading":

        initial_feature_list[n] = "precip_depth_1_hr"

initial_feature_list
del corr_df, correlations_transformed

gc.collect()
X = train[initial_feature_list]

y = train['log_meter_reading']
print("Pearson coefficient based feature selection leaves us with {} features.".

      format(len(X.columns)))
del train

gc.collect()
X = X.fillna(method='ffill', axis=1)
X.isna().sum()
# split train and validation set into 75 and 25 percent sequentially

X_train = X[:int(3 * X.shape[0] / 4)]

X_valid = X[int(3 * X.shape[0] / 4):]



y_train = y[:int(3 * y.shape[0] / 4)]

y_valid = y[int(3 * y.shape[0] / 4):]
# make sure train and validation sets shape align

print("Shape of the training set is: ", X_train.shape)

print("Shape of the validation set is: ", X_valid.shape)

print("Shape of the training labels are: ", y_train.shape)

print("Shape of the validation labels are: ", y_valid.shape)
def rmse(y_true, y_pred):

    return np.sqrt(

        np.mean(

            np.square(y_true - y_pred)

        )

    )
baseline_guess = np.median(y_train)

print('The baseline guess is a score of %0.2f' % baseline_guess)

print("Baseline Performance on the valid set: RMSE = %0.4f" % rmse(y_valid, baseline_guess))
print("Min value of meter_reading is:", y.min())

print("Median value of meter_reading is:", y.median())

print("Max value of meter_reading is:", y.max())
def fit_evaluate_model(model, X_train, y_train, X_valid, y_valid):

    model.fit(X_train, y_train)

    y_predicted = model.predict(X_valid)

    return sqrt(mean_squared_error(y_valid, y_predicted))
%%time

# create model apply fit_evaluate_model

linear_regression = LinearRegression()

lr_rmse = fit_evaluate_model(linear_regression, X_train, y_train, X_valid, y_valid)

print("RMSE of the linear regression model is:", lr_rmse)
del linear_regression

del lr_rmse

gc.collect()
# %%time

# create scaler

# scaler = MinMaxScaler()



# apply min_max_scaler to training set and transform training set

# X_train_scaled = scaler.fit_transform(X_train, y_train)



# transform validation set

# X_valid_scaled = scaler.transform(X_valid)



# knn_regressor = KNeighborsRegressor()

# knn_rmse = fit_evaluate_model(knn_regressor, X_train_scaled, y_train, X_valid_scaled, y_valid)

# print("RMSE of the k nearest neighbors regressor is:", knn_rmse)
%%time

# create model apply fit_evaluate_model

lgbm_regressor = lgb.LGBMRegressor(random_state=42)

lgbm_rmse = fit_evaluate_model(lgbm_regressor, X_train, y_train, X_valid, y_valid)

print("RMSE of the light gbm regressor is:", lgbm_rmse)
del lgbm_regressor

del lgbm_rmse

gc.collect()
# create categorical features 

categorical_features = ['building_id', 'site_id', 'meter',

                        'primary_use', 'wind_compass_direction',

                        'day_of_week', 'hour','is_weekend', 'season']
# tranform training and validation set into lgbm datasets

train_dataset = lgb.Dataset(X_train, label=y_train, 

                            categorical_feature=categorical_features, 

                            free_raw_data=False)

valid_dataset = lgb.Dataset(X_valid, label=y_valid, 

                            categorical_feature=categorical_features, 

                            free_raw_data=False)



# to record eval results for plotting

evals_result = {} 



# initial parameters of light gbm algorithm

initial_params = {"objective": "regression",

                  "boosting": "gbdt",

                  "num_leaves": 60,

                  "learning_rate": 0.05,

                  "feature_fraction": 0.85,

                  "reg_lambda": 2,

                  "metric": {'rmse'}

}
print("Building model with first 3 quarter pieces and evaluating the model on the last quarter:")

lgb_model = lgb.train(initial_params, 

                      train_set = train_dataset, 

                      num_boost_round = 1000, 

                      valid_sets=[train_dataset, valid_dataset],

                      verbose_eval = 100,

                      early_stopping_rounds = 500,

                      evals_result=evals_result)
print('Training and Validation Error of the Model')

ax = lgb.plot_metric(evals_result, metric='rmse')

plt.show()
del lgb_model

del train_dataset

del valid_dataset

del X_train

del X_valid

del y_train

del y_valid

gc.collect()
# add back to train and validation sets

# X = pd.concat([X_train, 

#              X_valid])



#y = pd.concat([y_train,

#               y_valid])
# cretae kfold object and empty model and evaluation lists

kf = KFold(n_splits=4, shuffle=False, random_state=42)

models = []

evaluations = []



# dynamically split X and y with the k-fold split indexes

for train_index,valid_index in kf.split(X):

    X_train_kf = X.loc[train_index]

    y_train_kf = y.loc[train_index]

    

    X_valid_kf = X.loc[valid_index]

    y_valid_kf = y.loc[valid_index]

    

    d_train = lgb.Dataset(X_train_kf, 

                          label=y_train_kf,

                          categorical_feature=categorical_features, 

                          free_raw_data=False)

    

    d_valid = lgb.Dataset(X_valid_kf, 

                          label=y_valid_kf,

                          categorical_feature=categorical_features, 

                          free_raw_data=False)

    evaluation_result = {}

    

    model = lgb.train(initial_params, 

                      train_set=d_train, 

                      num_boost_round=1000, 

                      valid_sets=[d_train, d_valid],

                      verbose_eval=100, 

                      early_stopping_rounds=500,

                      evals_result=evaluation_result)

    

    models.append(model)

    evaluations.append(evaluation_result)

    

    del X_train_kf, y_train_kf, X_valid_kf, y_valid_kf, d_train, d_valid

    gc.collect()
print('Training and Validation Error of the First Fold')

ax = lgb.plot_metric(evaluations[0], metric='rmse')

plt.show()
print('Training and Validation Error of the Second Fold')

ax = lgb.plot_metric(evaluations[1], metric='rmse')

plt.show()
print('Training and Validation Error of the Third Fold')

ax = lgb.plot_metric(evaluations[2], metric='rmse')

plt.show()
'''# objective function to optimize

def objective(params):

    # parameters to perform search

    params = {

        'num_leaves': int(params['num_leaves']),

        'colsample_bytree': '{:.2f}'.format(params['colsample_bytree']),

        'learning_rate': '{:.3f}'.format(params['learning_rate'])}

    

    # model and parameters to kept constant

    lgb_model = lgb.LGBMRegressor(

        reg_lambda= 2,

        **params,

        random_state=42)

    

    score = fit_evaluate_model(lgb_model, X_train, y_train, X_valid, y_valid)

    print("RMSE is {:.4f} with parameters {}".format(score, params))

    return score



# define search space

space = {

    'num_leaves': hp.choice('num_leaves', range(60, 130, 2)),

    'colsample_bytree': hp.uniform('colsample_bytree', 0.8, 1.0),

    'learning_rate': hp.uniform('learning_rate', 0.01, 0.1)

}



SEED = 42

# best model with the hyperopt

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=5,

            rstate= np.random.RandomState(SEED))'''
X.to_csv('X.csv', index=False)

y.to_csv('y.csv', index=False)