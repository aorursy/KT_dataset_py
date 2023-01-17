%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



plt.style.use('ggplot')

import matplotlib.cm as cm

import seaborn as sns



import plotly.express as px



import pandas as pd

import pandas_profiling

import numpy as np

from numpy import percentile

from scipy import stats

from scipy.stats import skew

from scipy.special import boxcox1p

import random



import os, sys

import re

from tabulate import tabulate

import missingno



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder



from sklearn.model_selection import train_test_split



from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



from sklearn.metrics import explained_variance_score

from sklearn.metrics import max_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import median_absolute_error

from sklearn.metrics import r2_score



from sklearn.dummy import DummyRegressor

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, RidgeCV

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



import xgboost as xgb

import lightgbm as lgb



from umap import UMAP



import warnings

warnings.filterwarnings('ignore')



def seed_everything(seed=0):

    random.seed(seed)

    np.random.seed(seed)

    

seed_everything()



plt.rc('font', size=18)        

plt.rc('axes', titlesize=22)      

plt.rc('axes', labelsize=18)      

plt.rc('xtick', labelsize=12)     

plt.rc('ytick', labelsize=12)     

plt.rc('legend', fontsize=12)   



plt.rcParams['font.sans-serif'] = ['Verdana']



pd.options.mode.chained_assignment = None

pd.options.display.max_seq_items = 500

pd.options.display.max_rows = 500

pd.set_option('display.float_format', lambda x: '%.5f' % x)



BASE_PATH = "../input/ashrae-energy-prediction/"
# reducing memory for now, we check later if this affects our models

def reduce_memory(df_):

    for col in df_.columns:

        if df_[col].dtype =='float64': df_[col] = df_[col].astype('float32')

        if df_[col].dtype =='int64': df_[col] = df_[col].astype('int32')

    return df_





# This functions is based on this cool script: 

# https://www.kaggle.com/bwilsonkg/column-statistics

def show_stats(data_frame):

    stats_column_names = ('column', 'dtype', 'nan_cts', 'nan_perc', 'val_cts',

                          'min', 'max', 'mean', 'median', 'stdev', 'skew', 'kurtosis')

    stats_array = []

    length_df = len(data_frame)

    for column_name in sorted(data_frame.columns):

        col = data_frame[column_name]

        if is_numeric_column(col):

            nan_perc = 100 / length_df * col.isna().sum()

            stats_array.append(

                [column_name, col.dtype, col.isna().sum(), nan_perc, len(col.value_counts()),

                 col.min(), col.max(), col.mean(), col.median(), col.std(), col.skew(),

                 col.kurtosis()])

        else:

            nan_perc = 100 / length_df * col.isna().sum()

            stats_array.append(

                [column_name, col.dtype, col.isna().sum(), nan_perc, len(col.value_counts()),

                 0, 0, 0, 0, 0, 0, 0])

    stats_df = pd.DataFrame(data=stats_array, columns=stats_column_names)

    with pd.option_context('display.float_format', lambda x: '%.1f' % x):

        display(stats_df)

        

def of_type(stats_data_frame, column_dtype):

    return stats_data_frame[stats_data_frame['dtype'] == column_dtype]



def sort(data_frame, column_name, ascending=False):

    return data_frame.sort_values(column_name, ascending=ascending)



def is_numeric_column(df_column):

    numeric_types = (np.int16, np.float16, np.int32, np.float32,

                     np.int64, np.float64)

    return df_column.dtype in numeric_types
bldg_df = reduce_memory(pd.read_csv(f"{BASE_PATH}building_metadata.csv"))



wth_train = reduce_memory(pd.read_csv(f"{BASE_PATH}weather_train.csv"))

wth_test  = reduce_memory(pd.read_csv(f"{BASE_PATH}weather_test.csv"))

wth_train.timestamp = pd.to_datetime(wth_train.timestamp)

wth_test.timestamp = pd.to_datetime(wth_test.timestamp)



weather_df = pd.concat([wth_train, wth_test])

weather_df.timestamp = pd.to_datetime(weather_df.timestamp)



train = reduce_memory(pd.read_csv(f"{BASE_PATH}train.csv"))

test  = reduce_memory(pd.read_csv(f"{BASE_PATH}test.csv"))

train.timestamp = pd.to_datetime(train.timestamp)

test.timestamp  = pd.to_datetime(test.timestamp)



# columns that can be set to dtype category

category_cols = ["site_id", "building_id", "primary_use", "year_built", "meter"]



# merge building with training data for EDA

#df = pd.merge(train, bldg_df, how="left")

#df.timestamp = pd.to_datetime(df.timestamp)

#df[category_cols] = df[category_cols].astype("category")



# merge train and test sets for modelling

train_df = pd.merge(pd.merge(train, bldg_df, how="left"), wth_train, how="left")

test_df  = pd.merge(pd.merge(test, bldg_df, how="left"), wth_test, how="left")



train_df[category_cols] = train_df[category_cols].astype("category")

test_df[category_cols]  = test_df[category_cols].astype("category")



train_df.timestamp = pd.to_datetime(train_df.timestamp)

test_df.timestamp  = pd.to_datetime(test_df.timestamp)



del wth_train, wth_test, train, test
for frame_name, frame in zip(["bldg_df    ", 

                              "weather_df ", 

                              "train_df   ", 

                              "test_df    ", 

                              "df         "], 

                             [bldg_df, weather_df, train_df, test_df]):

    print(f'{frame_name}: {sys.getsizeof(frame)/(1024.0**3) :.2f} GB')
show_stats(train_df)

display(train_df.sample(5).head())
show_stats(bldg_df)

display(bldg_df.sample(5).head())

missingno.matrix(bldg_df, figsize=(16,5), fontsize=12);
show_stats(weather_df)

display(weather_df.sample(5).head())

missingno.matrix(weather_df, figsize=(16,5), fontsize=12);
train_df.timestamp = pd.to_datetime(train_df.timestamp)

print(train_df.info(null_counts=True))
print(train_df.shape)

print(train_df.drop_duplicates().shape)
energy_types_dict = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}

energy_types      = ['electricity', 'chilledwater', 'steam', 'hotwater']
plt.figure(figsize=(16,5))

tmp_df = train_df.meter.value_counts()

tmp_df.index = energy_types

tmp_df.sort_values().plot(kind="barh")

plt.title(f"Most readings measure electricity")

plt.xlabel("Count of measurements")

plt.ylabel(f"Meter type")

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,5))

tmp_df = train_df.groupby("meter").meter_reading.sum()

tmp_df.index = energy_types

tmp_df.sort_values().plot(kind="barh")

plt.title(f"Generating steam consumes most energy")

plt.xlabel("Sum of consumed energy")

plt.ylabel(f"Type of energy")

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5))

sns.distplot(train_df.meter_reading, hist=False)

plt.title(f"Target variable meter_reading is highly skewed")

plt.ylabel("Count of readings")

plt.xlabel(f"Measured consumption")

plt.xlim(0, train_df.meter_reading.max() + 100_000)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,5))

sns.distplot(np.log1p(train_df.meter_reading))

plt.title(f"After log transform, meter readings look more workable but still skewed")

plt.ylabel("Count of readings")

plt.xlabel(f"Measured consumption")

plt.xlim(0, 12)

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,5))

for idx in range(0,4):

    sns.distplot(np.log1p(train_df[train_df.meter==idx].meter_reading), hist=False, label=energy_types[idx])

plt.title(f"After log transform, distributions of energy types look comparably skewed")

plt.ylabel("Count of readings")

plt.xlabel(f"Measured consumption")

plt.legend()

plt.xlim(0, 12)

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,7))

_ = stats.probplot(train_df['meter_reading'], plot=plt)

plt.title("Probability plot for meter_reading shows extreme skewness")

plt.show()
plt.figure(figsize=(16,7))

_ = stats.probplot(np.log(train_df['meter_reading']), plot=plt)

plt.title("Even log transformed meter_reading is highly skewed")

plt.show()
train_df.groupby("building_id").meter_reading.sum().sort_values(ascending=False)[:5]
for bldg_id in [1099, 778, 1197, 1168, 1159]:

    plt.figure(figsize=(16,5))

    tmp_df = train_df[train_df.building_id == bldg_id].copy()

    tmp_df.set_index("timestamp", inplace=True)

    tmp_df.resample("D").meter_reading.sum().plot()

    plt.title(f"Meter readings for building #{bldg_id} ")

    plt.xlabel("Sum of readings")

    plt.tight_layout()

    plt.show()
temp_df = train_df.groupby("primary_use").meter_reading.sum().sort_values()



outliers_index = train_df[train_df.building_id.isin([1099, 778])].index

temp_df_inliers = train_df.drop(outliers_index).groupby("primary_use").meter_reading.sum().sort_values()



plt.figure(figsize=(16,9))

temp_df.plot(kind="barh")

plt.title(f"Education buildings consume by far most of energy")

plt.xlabel("Sum of readings")

plt.ylabel(f"Primary use")

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,9))

temp_df_inliers.plot(kind="barh")

plt.title(f"Less so without outliers 1099, 778")

plt.xlabel("Sum of readings")

plt.ylabel(f"Primary use")

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,9))

temp_df[:-1].plot(kind="barh")

plt.title(f"Among other types, office buildings consume most energy")

plt.xlabel("Sum of readings")

plt.ylabel(f"Primary use w/o «Education»")

plt.tight_layout()

plt.show()
sq_binned = pd.cut(train_df.square_feet, bins=np.arange(0, 1_000_000, 100_000))

sq_binned = pd.DataFrame(sq_binned)

sq_binned.columns = ["sq_binned"]

tmp_df = pd.concat([train_df, sq_binned], axis=1).groupby("sq_binned").meter_reading.mean().sort_index()



plt.figure(figsize=(16,7))

tmp_df.plot(kind="barh")

plt.title(f"Buildings between 300-400k square feet consume by far most of energy")

plt.xlabel("Mean of consumption")

plt.ylabel(f"Square feet of building (binned)")

plt.tight_layout()

plt.show()
sq_binned = pd.cut(train_df.drop(outliers_index).square_feet, bins=np.arange(0, 1_000_000, 100_000))

sq_binned = pd.DataFrame(sq_binned)

sq_binned.columns = ["sq_binned"]

tmp_df = pd.concat([train_df.drop(outliers_index), sq_binned], axis=1).groupby("sq_binned").meter_reading.mean().sort_index()



plt.figure(figsize=(16,7))

tmp_df.plot(kind="barh")

plt.title(f"More balanced distribution if we remove outliers 1099, 778")

plt.xlabel("Mean of consumption")

plt.ylabel(f"Square feet of building (binned)")

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5))

train_df.groupby("building_id").meter.nunique().value_counts().sort_index().plot(kind="bar")

plt.title(f"Buildings have up to four types of meters, most have one (or three) types")

plt.ylabel("Count of buildings")

plt.xlabel(f"Number of different types of meters")

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5))

train_df.building_id.value_counts().plot(kind="bar")

plt.title("There are distinct ranges of meter readings per building")

plt.ylabel("Count of readings")

plt.xlabel("building_id")

plt.xticks([])

plt.tight_layout()

plt.show()



plt.figure(figsize=(16,5))

tmp_df = pd.cut(train_df.building_id.value_counts(), bins=np.arange(0, 45_000, 10_000))

tmp_df = pd.DataFrame(tmp_df)

tmp_df.building_id.value_counts().sort_index().plot(kind="bar")

plt.title("Distinct ranges of meter readings per building (binned)")

plt.ylabel("Count of readings")

plt.xlabel("building_id")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()



display(tmp_df.building_id.value_counts())
print(f"Timestamps in the training set range from {train_df.timestamp.min()} to {train_df.timestamp.max()}")
timeframes = {"month"   : train_df.timestamp.dt.month,

              "week"    : train_df.timestamp.dt.week, 

              "weekday" : train_df.timestamp.dt.weekday, 

              "hour"    : train_df.timestamp.dt.hour}



for timeframe_name, timeframe in timeframes.items():

    plt.figure(figsize=(16,5))

    train_df.groupby(timeframe).building_id.count().plot(kind="bar")

    plt.title(f"Quite even counts of meter readings per {timeframe_name}")

    plt.ylabel("Count of readings")

    plt.xlabel(f"{timeframe_name}")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()
for timeframe_name, timeframe in timeframes.items():

    plt.figure(figsize=(16,5))

    train_df.groupby(timeframe).meter_reading.median().plot(kind="bar")

    plt.title(f"Fairly stable median of energy consumption per {timeframe_name}")

    plt.ylabel("Median of energy consumption")

    plt.xlabel(f"{timeframe_name}")

    plt.xticks(rotation=45)

    

    if timeframe_name == "weekday":

        plt.title("Lower median consumption during weekend")

    if timeframe_name == "hour":

        plt.title("Higher median consumption during daytime")

        

    plt.tight_layout()

    plt.show()

    

    

    plt.figure(figsize=(16,5))

    train_df.groupby(timeframe).meter_reading.sum().plot(kind="bar")

    plt.title(f"Energy consumption peaks significantly in Spring (again due to outliers 1099, 778)")

    plt.ylabel("Total energy consumption")

    plt.xlabel(f"{timeframe_name}")

    plt.xticks(rotation=45)

    

    if timeframe_name == "weekday":

        plt.title("Lower total consumption during weekend")

    if timeframe_name == "hour":

        plt.title("Higher total consumption during daytime and evening")

        

    plt.tight_layout()

    plt.show()

    

    

    plt.figure(figsize=(16,7))

    sns.boxplot(x=timeframe, y="meter_reading", data=train_df, showfliers=False)

    plt.title(f"Noticable differences in distribution of meter readings per {timeframe_name}")

    plt.ylabel("meter readings")

    plt.xlabel(f"{timeframe_name}")

    plt.xticks(rotation=45)

    

    if timeframe_name == "weekday":

            plt.title(f"Fairly stable distribution of meter readings per {timeframe_name}")

            plt.xlabel(f"{timeframe_name} (0 == Monday)")

    if timeframe_name == "hour":

            plt.title(f"Fairly stable distribution of meter readings per {timeframe_name}")



    plt.tight_layout()

    plt.show()
for timeframe_name, timeframe in timeframes.items():

    plt.figure(figsize=(16,7))

    for idx in range(0,4):

        tmp_df = train_df[train_df.meter==idx].groupby(timeframe).meter_reading.sum()

        tmp_df.plot(kind="line", label=energy_types[idx], use_index=True)

    plt.xticks(rotation=45)

    plt.ylabel("Median of consumption")

    plt.xlabel(f"{timeframe_name}")

    plt.title(f"Steam energy consumption fairly stable through out the day")

    

    if timeframe_name in ["month", "week"]:

        plt.title(f"Steam energy consumption peaks extremely in Spring due to outliers")

    if timeframe_name in ["weekday"]:

        plt.title(f"Steam energy consumption stronger on Wednesdays, less on weekends")

    

    plt.legend(bbox_to_anchor=(1, 1), loc=2)

    plt.tight_layout()

    plt.show()
for timeframe_name, timeframe in timeframes.items():

    plt.figure(figsize=(16,7))

    for idx in range(0,4):

        tmp_df = train_df.drop(outliers_index)

        tmp_df = tmp_df[tmp_df.meter==idx].groupby(timeframe).meter_reading.sum()

        tmp_df.plot(kind="line", label=energy_types[idx], use_index=True)

    plt.xticks(rotation=45)

    plt.ylabel("Median of consumption")

    plt.xlabel(f"{timeframe_name}")

    plt.title(f"Energy consumption fairly stable through out the day")

    

    if timeframe_name in ["month", "week"]:

        plt.title(f"Steam energy consumption w/o outliers stronger in cold season")

    if timeframe_name in ["weekday"]:

        plt.title(f"Energy consumption a little less on weekends")

    

    plt.legend(bbox_to_anchor=(1, 1), loc=2)

    plt.tight_layout()

    plt.show()
for timeframe_name, timeframe in timeframes.items():

    plt.figure(figsize=(16,7))

    for idx in range(0,4):

        if idx == 2:

            continue

        tmp_df = train_df.drop(outliers_index)

        tmp_df = tmp_df[tmp_df.meter==idx].groupby(timeframe).meter_reading.sum()

        tmp_df.plot(kind="line", label=energy_types[idx], use_index=True)

    plt.xticks(rotation=45)

    plt.ylabel("Median of consumption")

    plt.xlabel(f"{timeframe_name}")

     

    if timeframe_name in ["month", "week"]:

        plt.title(f"Chilled water most consumed in late summer / September")

    if timeframe_name in ["weekday"]:

        plt.title(f"Consumption fairly stable during week, less on weekend")

    if timeframe_name in ["hour"]:

        plt.title(f"Consumption stronger during daytime, less at night\nMore hotwater consumption in mornings, chilledwater more on afternoons")

   

    plt.legend(bbox_to_anchor=(1, 1), loc=2)

    plt.tight_layout()

    plt.show()
# calculate normal and extreme upper and lower cut off



cut_off  = train_df["meter_reading"].std() * 3

lower    = train_df["meter_reading"].mean() - cut_off 

upper    = train_df["meter_reading"].mean() + cut_off

df_lower = train_df[train_df["meter_reading"] < lower]

df_upper = train_df[train_df["meter_reading"] > upper]

    

if df_lower.shape[0] != 0 or df_upper.shape[0] != 0:

    print(f"{'meter_reading'}")

    print(f"lower bound: {lower:.2f}\nupper bound: {upper:.2f}")

if df_lower.shape[0] != 0:

        display(train_df[train_df["meter_reading"] < lower].sort_values("meter_reading"))

if df_upper.shape[0] != 0:

        display(train_df[train_df["meter_reading"] > upper].sort_values("meter_reading"))



display(df_upper.building_id.value_counts())

display(df_upper.meter.value_counts())
print(bldg_df.info())
print(bldg_df.shape)

print(bldg_df.drop_duplicates().shape)
missing = [(c, bldg_df[c].isna().mean()*100) for c in bldg_df]

missing = pd.DataFrame(missing, columns=["feature", "percentage"])

missing["count"] = [bldg_df[c].isna().sum() for c in bldg_df]

missing = missing[missing.percentage > 0]

display(missing.sort_values("percentage", ascending=False))
print(f"We have {bldg_df.building_id.nunique()} unique buildings and {bldg_df.site_id.nunique()} sites of buildings.")

sites_df = pd.DataFrame(bldg_df.site_id.value_counts())

sites_df.sort_values("site_id", inplace=True, ascending=False)

sites_ordered_index = sites_df.index

print(f"The number of buildings per site range from {sites_df.site_id.min()} to {sites_df.site_id.max()}.")



plt.figure(figsize=(16,5))

sites_df.site_id.plot(kind="bar")

plt.title("Count of buildings per site_id vary between 274 and 5")

plt.ylabel("Count of buildings")

plt.xlabel("site id")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
show_stats(bldg_df[["square_feet", "year_built", "floor_count"]])
print("Smallest building")

display(bldg_df[bldg_df.square_feet == bldg_df.square_feet.min()])

print("Largest building")

display(bldg_df[bldg_df.square_feet == bldg_df.square_feet.max()])

print("Oldest buildings")

display(bldg_df[bldg_df.year_built == bldg_df.year_built.min()])

print("Most recent building")

display(bldg_df[bldg_df.year_built == bldg_df.year_built.max()])

print("Least tallest building")

display(bldg_df[bldg_df.floor_count == bldg_df.floor_count.min()][:5])

print("Tallest building")

display(bldg_df[bldg_df.floor_count == bldg_df.floor_count.max()])
display(bldg_df.primary_use.value_counts())
graph_df = bldg_df.groupby("site_id")["primary_use"].value_counts().unstack()

graph_df = graph_df.reindex(sites_ordered_index)

graph_df.plot(kind="bar", stacked=True, figsize=(16,7))

plt.title("Primary use mapped to site_id")

plt.ylabel("Count of buildings")

plt.xlabel("site_id")

plt.legend(bbox_to_anchor=(1, 1), loc=2)

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5))

sns.distplot(bldg_df["square_feet"])

plt.title(f"Distribution of square feet of buildings")

plt.xlabel(f"square feet")

plt.tight_layout()

plt.show()
size_df = bldg_df.groupby("primary_use")["square_feet"].mean().sort_values(ascending=False)

order = size_df.index

plt.figure(figsize=(16,12))

sns.boxplot(x="primary_use", y="square_feet", data=bldg_df, order=order)

plt.title("Parking facilities by far the biggest on average")

plt.ylabel("square feet")

plt.xticks(rotation=80)

plt.tight_layout()

plt.show()
plt.figure(figsize=(16,5))

bldg_df["year_built"].dropna().plot(kind="hist", bins=117, rwidth=0.9)

plt.title(f"A significant peak of 55 buildings from 1976")

plt.xlabel(f"Year built")

plt.tight_layout()

plt.show()



display(bldg_df['year_built'].dropna().value_counts().iloc[:1])
size_df = bldg_df.groupby("primary_use")["year_built"].median().sort_values(ascending=False)

order = size_df.index

plt.figure(figsize=(16,12))

sns.boxplot(x="primary_use", y="year_built", data=bldg_df, order=order)

plt.title("Food, Parking, Healthcare and Retail buildings are the youngest")

plt.ylabel("year_built")

plt.xticks(rotation=80)

plt.tight_layout()

plt.show()
size_df = bldg_df.groupby("primary_use")["floor_count"].median().sort_values(ascending=False)

order = size_df.index



plt.figure(figsize=(16,12))

sns.boxplot(x="primary_use", y="floor_count", data=bldg_df, order=order)

plt.title("Lodging/residential on average the tallest buildings")

plt.ylabel("floor count")

plt.xticks(rotation=80)

plt.tight_layout()

plt.show()
# calculate normal and extreme upper and lower cut off

for feature in bldg_df.select_dtypes("number").columns:



    cut_off = bldg_df[feature].std() * 3

    lower   = bldg_df[feature].mean() - cut_off 

    upper   = bldg_df[feature].mean() + cut_off

    df_lower = bldg_df[bldg_df[feature] < lower]

    df_upper = bldg_df[bldg_df[feature] > upper]

    

    if df_lower.shape[0] != 0 or df_upper.shape[0] != 0:

        print(f"{feature}")

        print(f"lower bound: {lower:.2f}\nupper bound: {upper:.2f}")

        if df_lower.shape[0] != 0:

            display(bldg_df[bldg_df[feature] < lower].sort_values(feature))

        if df_upper.shape[0] != 0:

            display(bldg_df[bldg_df[feature] > upper].sort_values(feature))

        print()
weather_df.set_index("timestamp", inplace=True)

print(weather_df.info())
print(weather_df.shape)

print(weather_df.drop_duplicates().shape)
missing = [(c, weather_df[c].isna().mean()*100) for c in weather_df]

missing = pd.DataFrame(missing, columns=["feature", "percentage"])

missing["count"] = [weather_df[c].isna().sum() for c in weather_df]

missing = missing[missing.percentage > 0]

display(missing.sort_values("percentage", ascending=False))
plt.figure(figsize=(16,5))

weather_df.groupby(pd.Grouper(freq="W"))['air_temperature'].sum().plot()

plt.title('air_temperature')

plt.ylabel('air_temperature')

plt.tight_layout()

plt.show()
features = ['air_temperature', 'dew_temperature', 

            'sea_level_pressure', 'wind_direction', 'wind_speed', 

            'cloud_coverage', 'precip_depth_1_hr',]

for feature in features:

    plt.figure(figsize=(16,5))

    for sid in weather_df.site_id.unique():

        weather_df[weather_df.site_id==sid].groupby(pd.Grouper(freq="W"))[feature].sum().plot()

    plt.title(f"{feature}")

    plt.ylabel(f"{feature}")

    plt.tight_layout()

    plt.show()
# these plots are inspired by this kernel: 

# https://www.kaggle.com/blue07/eda-insights-on-weather-buildings



for feature in features:

    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(16, 12))

    for sid in weather_df.site_id.unique():

        row = int(sid / 4)

        col = sid%4

        tmp_df = weather_df[weather_df.site_id==sid]

        missing = 100 / len(tmp_df) * tmp_df[feature].isnull().sum()

        tmp_df.groupby(pd.Grouper(freq="M"))[feature].sum().plot(ax=axes[row,col])

        if missing !=0:

            axes[row, col].set_title(f"site {sid}, null:{missing :.2f}%", fontsize=12, color="darkred")

        else:

            axes[row, col].set_title(f"site {sid}", fontsize=12, color="darkgreen")

        axes[row, col].set_xlabel("")

    fig.suptitle(f"{feature}", fontsize=18)

    fig.subplots_adjust(top=0.92)
fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, figsize=(16, 12))

for sid in weather_df.site_id.unique():

    row = int(sid / 4)

    col = sid%4

    tmp_df = weather_df[weather_df.site_id==sid]

    missing = 100 / len(tmp_df) * tmp_df['wind_direction'].isnull().sum()

    tmp_df.groupby(tmp_df.index.hour)['wind_direction'].median().plot(ax=axes[row,col])

    if missing !=0:

        axes[row, col].set_title(f"site {sid}, null:{missing :.2f}%", fontsize=12, color="darkred")

    else:

        axes[row, col].set_title(f"site {sid}", fontsize=12, color="darkgreen")

    axes[row, col].set_xlabel("")

fig.suptitle(f"wind_direction per hour of day", fontsize=18)

fig.subplots_adjust(top=0.92)
display(weather_df[weather_df.site_id ==1].cloud_coverage.value_counts())
lb = LabelEncoder()

train_df.primary_use_lb = lb.fit_transform(train_df.primary_use)

train_df.primary_use_lb = train_df.primary_use_lb.astype("int32")

# sample 100k to avoid crashing of kernel

corr_raw = train_df.sample(100_000).drop(["timestamp", "primary_use"], axis=1).astype(float)
# adding some temporary time related features

corr_raw["quarter"] = train_df.timestamp.dt.quarter

corr_raw["quarter_start"] = train_df.timestamp.dt.is_quarter_start

corr_raw["quarter_end"] = train_df.timestamp.dt.is_quarter_end

corr_raw["month"] = train_df.timestamp.dt.month

corr_raw["month_start"] = train_df.timestamp.dt.is_month_start

corr_raw["month_end"] = train_df.timestamp.dt.is_month_end

corr_raw["week"] = train_df.timestamp.dt.week

corr_raw["dayofweek"] = train_df.timestamp.dt.dayofweek

corr_raw["weekend"] = corr_raw.dayofweek.apply(lambda x: True if x in [5, 6] else False)

corr_raw["day"] = train_df.timestamp.dt.day

corr_raw["hour"] = train_df.timestamp.dt.hour
corr = corr_raw.corr()

plt.figure(figsize=(16,9));

corr["meter_reading"].sort_values(ascending=True)[:-1].plot(kind="barh")

plt.title("Correlation of features to meter_reading")

plt.xlabel("Correlation to meter_reading")

plt.tight_layout()

plt.show()
# get correlation among all features with pandas .corr() function

corr = corr_raw.corr()

# filter correlations less than 0.1

cut_off = 0.1

corr = corr[(corr > cut_off) | (corr < -cut_off)]



plt.subplots(figsize=(16,16));

sns.heatmap(corr, cmap="RdBu", square=True, annot=False, cbar_kws={"shrink": .6}, )

plt.title(f"Correlation of features greater than +/-{cut_off}")

plt.tight_layout()

plt.show()
## these heatmaps are inspired by this kernel: 

# https://www.kaggle.com/blue07/eda-insights-on-weather-buildings



rows = 8

cols = 2



fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(16, 64))

for sid in corr_raw.site_id.unique():

    row = int(sid/cols)

    col = int(sid%cols)

    tmp_df = corr_raw[corr_raw.site_id==sid]

    corr = tmp_df.corr()

    corr = corr[(corr > cut_off) | (corr < -cut_off)]

    sns.heatmap(corr, cmap="RdBu", square=True, cbar=False, ax=axes[row,col])

    axes[row, col].set_xlabel("")

    axes[row, col].set_title(f"site {int(sid)}", fontsize=12)

fig.suptitle(f"Correlation per site_id, greater than +/-{cut_off}", fontsize=18)

fig.subplots_adjust(top=0.965)