# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g.L pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# importing libraries



%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
# Reading the data



# Reading metadata

# specify column dtypes to save memory (by default pandas reads some columns as floats)

dtypes = {

    "site_id": np.int8,

    "building_id": np.int8,

    "primary_use": np.object,

    "square_feet": np.int8,

    "year_built": np.float16,

    "floor_count": np.float16,

}



# meta data information for metadata dataframe

skiprows = None

nrows = None

columns = ['site_id', 'building_id', 'primary_use', 'square_feet', 'year_built', 'floor_count']

path = "/kaggle/input/ashrae-energy-prediction/building_metadata.csv"



metadata_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)
# Reading training data



# specify column dtypes to save memory (by default pandas reads some columns as floats)

dtypes = {

    "building_id": np.int8,

    "meter": np.int8,

    "timestamp": np.object,

    "meter_reading": np.float16

}



# meta data information for metadata dataframe

skiprows = None

nrows = None

columns = ['building_id', 'meter', 'timestamp', 'meter_reading']

path = "/kaggle/input/ashrae-energy-prediction/train.csv"



# using chunck size to avoid memory related issues

train_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)
# Reading testing data



# specify column dtypes to save memory (by default pandas reads some columns as floats)

dtypes = {

    "row_id": np.int8,

    "building_id": np.int8,

    "meter": np.int8,

    "timestamp": np.object

}



# meta data information for metadata dataframe

skiprows = None

nrows = None

columns = ['row_id', 'building_id', 'meter', 'timestamp']

path = "/kaggle/input/ashrae-energy-prediction/test.csv"



# using chunck size to avoid memory related issues

test_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)
# Reading weather training data



# specify column dtypes to save memory (by default pandas reads some columns as floats)

dtypes = {

    "site_id": np.int8,

    "timestamp": np.object,

    "air_temperature": np.float16,

    "cloud_coverage": np.float16,

    "dew_temperature": np.float16,

    "precip_depth_1_hr": np.float16,

    "sea_level_pressure": np.float16,

    "wind_direction": np.float16,

    "wind_speed": np.float16

}



# meta data information for metadata dataframe

skiprows = None

nrows = None

columns = ['site_id', 'timestamp', 'air_temperature', 'cloud_coverage',

       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',

       'wind_direction', 'wind_speed']

path = "/kaggle/input/ashrae-energy-prediction/weather_train.csv"



weather_train_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)
# Reading weather testing data



# specify column dtypes to save memory (by default pandas reads some columns as floats)

dtypes = {

    "site_id": np.int8,

    "timestamp": np.object,

    "air_temperature": np.float16,

    "cloud_coverage": np.float16,

    "dew_temperature": np.float16,

    "precip_depth_1_hr": np.float16,

    "sea_level_pressure": np.float16,

    "wind_direction": np.float16,

    "wind_speed": np.float16

}



# meta data information for metadata dataframe

skiprows = None

nrows = None

columns = ['site_id', 'timestamp', 'air_temperature', 'cloud_coverage',

       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',

       'wind_direction', 'wind_speed']

path = "/kaggle/input/ashrae-energy-prediction/weather_test.csv"



weather_test_df = pd.read_csv(path, skiprows=skiprows, nrows=nrows, dtype=dtypes, usecols=columns)
# describing about the data



metadata_df.head()
metadata_df.describe()
metadata_df.info()
weather_test_df.head()
weather_train_df.describe()
weather_train_df.info()
weather_test_df.head()
weather_test_df.describe()
weather_test_df.info()
train_df.head()
train_df.describe()
train_df.info()
test_df.head()
test_df.describe()
test_df.info()
# checking the null values on metadata dataset



metadata_df.isnull().sum(axis=0)
weather_train_df.isnull().sum(axis=0)
weather_test_df.isnull().sum(axis=0)
train_df.isnull().sum(axis=0)
test_df.isnull().sum(axis=0)
# percentage of missing values in weather training dataset



round((weather_train_df.isnull().sum() * 100 / len(weather_train_df)), 2)
# percentage of missing values in weather testing dataset



round((weather_test_df.isnull().sum() * 100 / len(weather_test_df)), 2)
# Finding no.f unique values in weather training dataset



print("Finding no.f unique values in weather training dataset: {}".format(len(pd.unique(weather_train_df['site_id']))))
# Finding no.f unique values in weather testing dataset



print("Finding no.f unique values in weather testing dataset: {}".format(len(pd.unique(weather_test_df['site_id']))))
# Finding no.f unique values in training dataset 



print("Finding no.f unique values in training dataset: {}".format(len(pd.unique(train_df["building_id"]))))
# Finding no.f unique values in testing dataset



print("Finding no.f unique values in testing dataset: {}".format(len(pd.unique(test_df["building_id"]))))
# No.f records in weather training dataset



print("No.f records in weather training dataset: {}".format(len(weather_train_df)))
# No.f records in weather testing dataset



print("No.f records in weather testing dataset: {}".format(len(weather_test_df)))
# No.f records in trainging dataset



print("No.f records in training dataset: {}".format(len(train_df)))
# No.f records in testing dataset



print("No.f records in testing dataset: {}".format(len(test_df)))
weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])
weather_train_df.head()
weather_test_df.head()
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
train_df.head()
test_df.head()
# removing unnecessary data



import gc

gc.collect()