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
import numpy as np

import pandas as pd

import datetime 

import seaborn as sns

import plotly.offline as py

import matplotlib.pyplot as plt

import plotly.graph_objs as go
#Load the data



weather_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

train_data = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")

building_metadata = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")

test_data = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")

weather_test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")
weather_train.head()
train_data.head()
building_metadata.head()
train_data.dtypes
train_data.memory_usage()
#Reduce Memory Usuage 

def reduce_mm_usage(df, verbose=True):

    type_list = ['int16','int32','int64','float16','float32','float64']

    

    print("Inital memory usage in KB", df.memory_usage().sum()/1024)

    initial_mm = df.memory_usage().sum()/1024

    for col in df.columns:

        if df[col].dtypes in type_list:

            

            col_min = df[col].min()

            col_max = df[col].max()

            

            if str(df[col].dtypes)[:3]=="int":

                

                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

                    

            else:

                

                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else: #col_min > np.finfo(np.float64).min and col_max < np.finfo(np.float64).max:

                    df[col] = df[col].astype(np.float64)

    

    end_mm = df.memory_usage().sum()/1024

    print("after memory usuage in KB", df.memory_usage().sum()/1024)

    print("percetage reduction in memory", ( (initial_mm-end_mm)/initial_mm) )

    return df
train_data = reduce_mm_usage(train_data)
test_data = reduce_mm_usage(test_data)
building_metadata = reduce_mm_usage(building_metadata)
weather_train = reduce_mm_usage(weather_train)
weather_test = reduce_mm_usage(weather_test)
#null values 

train_data.isnull().sum()
train_data.timestamp = pd.to_datetime(train_data.timestamp, format='%Y-%m-%d %H:%M:%S')

train_data = train_data.set_index('timestamp')
#groupby buliding_id and meter_id, roll on monthly data over summation of meter reading 



train_data = train_data.groupby([pd.Grouper(freq='M'),'building_id','meter'], as_index=True)['meter_reading'].sum().reset_index()
train_data.head()
train_data.timestamp = train_data.timestamp.dt.strftime("%Y-%m-%d")
plt.figure(figsize=(18,12))

lm = sns.scatterplot(x='timestamp',y="meter_reading",data=train_data, hue='meter', palette='tab20b',s=100)
#Distribution of target variable that is meter_reading 

plt.figure(figsize=(18,12))

sns.distplot(train_data.meter_reading)
#Missing Data Analysis



def miss_data(df):

    x = ['column_name','missing_data', 'missing_in_percentage']

    missing_data = pd.DataFrame(columns=x)

    columns = df.columns

    for col in columns:

        icolumn_name = col

        imissing_data = df[col].isnull().sum()

        imissing_in_percentage = (df[col].isnull().sum()/df[col].shape[0])*100

        

        missing_data.loc[len(missing_data)] = [icolumn_name, imissing_data, imissing_in_percentage]

    print(missing_data)

        
miss_data(train_data)
miss_data(test_data)
miss_data(weather_train)
miss_data(weather_test)
miss_data(building_metadata)
train_data.corr()['meter_reading'].sort_values()
sns.heatmap(train_data.corr(), cmap="rainbow", vmin=-0.01, vmax=1)
def plot_dist(df, column):

    plt.figure(figsize=(18,12))

    ax = sns.distplot(df[column].dropna())

    ax.set_title(column+" Distribution", fontsize=16)

    plt.xlabel(column, fontsize=12)

    #plt.ylabel("distribution", fontsize=12)

    plt.show()
plot_dist(train_data, "meter_reading")
plot_dist(weather_train, "air_temperature")
weather_train.dtypes
plot_dist(weather_train, "dew_temperature")
plot_dist(weather_train, "sea_level_pressure")