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
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

# Modified to support timestamp type, categorical type

# Modified to add option to use float16 or not. feather format does not support float16.

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
def load_data(dataset): 

    df = reduce_mem_usage(pd.read_csv("/kaggle/input/ashrae-energy-prediction/{}.csv".format(dataset), parse_dates=["timestamp"]))

    building_metadata = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv").groupby("building_id").first().fillna(-1)

    weather_data = reduce_mem_usage(pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_{}.csv".format(dataset), parse_dates=["timestamp"])\

        .groupby("site_id").apply(lambda x: x.sort_values("timestamp").ffill().bfill().groupby("timestamp").first().resample(rule="H").first()).drop(columns="site_id"))

    df = df.join(other=building_metadata, on="building_id")

    df = df.join(other=weather_data, on=["site_id", "timestamp"]).fillna(-1)

    df["hour_of_day"] = df.timestamp.dt.hour

    return pd.get_dummies(reduce_mem_usage(df))[['timestamp', 'meter', 'building_id', 'square_feet', 'year_built', 'floor_count', 'air_temperature',

       'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',

       'sea_level_pressure', 'wind_direction', 'wind_speed', 'hour_of_day',

       'primary_use_Education', 

       'primary_use_Lodging/residential', 'primary_use_Office'] + ([] if dataset == "test" else ["meter_reading"])]
df = load_data("train").sample(4000000)

df["meter_reading"] = np.log1p(df.meter_reading)

df
from seaborn import distplot

# distplot(df.meter_reading)
df["meter_id"] = df["building_id"].apply(str) + "_" + df["meter"].apply(str)
from seaborn import scatterplot

# scatterplot(x="square_feet", y="meter_reading", data=df.groupby("meter_id").max(), hue="meter")
# scatterplot(x="square_feet", y="meter_reading", data=df[df.meter == 0].groupby("meter_id").max())
# scatterplot(x="square_feet", y="meter_reading", data=df[df.meter == 1].groupby("meter_id").max())
# scatterplot(x="square_feet", y="meter_reading", data=df[df.meter == 2].groupby("meter_id").max())
# scatterplot(x="square_feet", y="meter_reading", data=df[df.meter == 3].groupby("meter_id").max())
df = df[~df.meter_id.isin(df[df.meter_reading > 11].meter_id)]

# scatterplot(x="square_feet", y="meter_reading", data=df.groupby("meter_id").max(), hue="meter").set_title("Scatterplot of meter readings after outlier removal")
# distplot(df.meter_reading).set_title("Distribution of meter readings after outlier removal")
from seaborn import lineplot

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# lineplot(data=df.groupby("timestamp")["meter_reading"].agg(lambda c: np.expm1(c).sum()).resample(rule="D").sum()).set_title("Total energy usage over time")
import gc

from lightgbm import LGBMRegressor

# from sklearn.ensemble import RandomForestRegressor



r = LGBMRegressor(n_estimators=447, learning_rate=0.1616975721056746, max_depth=13, num_leaves=127, min_child_samples=16, reg_alpha=0.01, reg_lambda=0.00001, verbose=2, n_jobs=-1)

attributes = ['meter', 'square_feet', 'year_built', 'floor_count', 'air_temperature',

       'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',

       'sea_level_pressure', 'wind_direction', 'wind_speed', 'hour_of_day',

       'primary_use_Education', 

       'primary_use_Lodging/residential', 'primary_use_Office']

train_columns = attributes + ["meter_reading"]

df = df[train_columns + ["timestamp"]]

sdf = df.sort_values("timestamp").drop("timestamp", axis=1)

train_X, train_y = sdf[attributes], sdf["meter_reading"]

del df

gc.collect()
from sklearn.metrics import mean_squared_log_error

from sklearn.metrics.scorer import make_scorer



def rmsle(y_true, y_pred, **kwargs):

    score = np.sqrt(mean_squared_log_error(np.expm1(y_true), np.clip(np.expm1(y_pred), 0, None)))

    print("RMSLE: {}".format(score))

    return score



rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
# from sklearn.metrics.scorer import make_scorer

# from sklearn.model_selection import TimeSeriesSplit

# from sklearn.feature_selection import RFECV



# rfecv = RFECV(estimator=r, step=1, cv=TimeSeriesSplit(n_splits=5), verbose=1, scoring=make_scorer(rmsle))

# rfecv.fit(train_X, train_y)



# print("Optimal number of features : %d" % rfecv.n_features_)
# import matplotlib.pyplot as plt

# plt.figure()

# plt.xlabel("Number of features selected")

# plt.ylabel("Cross validation score (nb of correct classifications)")

# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

# plt.show()
# sorted(list(zip(rfecv.ranking_, train_X.columns)), reverse=True)
# from scipy.stats import uniform, randint

# from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit



# # distributions = dict(n_estimators=randint(50, 500), max_depth=randint(2, 15), learning_rate=uniform(loc=0, scale=0.3), num_leaves=randint(2, 128), min_child_samples=randint(5, 200), 

# #                      reg_alpha=uniform(loc=0, scale=0.1), reg_lambda=uniform(loc=0, scale=0.1))

# distributions = dict(reg_alpha=[10**i for i in range(-5,1)], reg_lambda=[10**i for i in range(-5,1)])

# rscv = RandomizedSearchCV(r, distributions, scoring=make_scorer(rmsle, greater_is_better=False), cv=TimeSeriesSplit(n_splits=5), n_iter=10, verbose=2)

# search = rscv.fit(train_X, train_y)

# print("Best score: {}".format(search.best_score_))

# print(search.best_params_)
# from scipy.stats import sem, t

# from scipy import mean



# def get_mean_confidence_diff(confidence, values):

#     n = len(values)

#     m = mean(values)

#     std_err = sem(values)

#     diff = std_err * t.ppf((1 + confidence) / 2, n - 1)

#     return diff
# from sklearn.model_selection import cross_val_score, TimeSeriesSplit



# scores = cross_val_score(r, train_X, train_y, scoring=rmsle_scorer, cv=TimeSeriesSplit(n_splits=5))



# print("RMSLE (95% confidence): {:.2f} (+/- {:.2f})".format(-scores.mean(), get_mean_confidence_diff(0.95, scores)))
r.fit(train_X, train_y)
del train_X

del train_y

gc.collect()

test_df = load_data("test")

test_df = test_df[attributes]

test_df
predictions = r.predict(test_df)

predictions
sample_submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
sample_submission["meter_reading"] = np.expm1(predictions)
sample_submission
sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')