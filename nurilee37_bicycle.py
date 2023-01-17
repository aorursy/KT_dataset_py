"""

# variable descriptions

datetime - hourly date + timestamp  

season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 

holiday - whether the day is considered a holiday

workingday - whether the day is neither a weekend nor holiday

weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy

2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist

3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds

4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 

temp - temperature in Celsius

atemp - "feels like" temperature in Celsius

humidity - relative humidity

windspeed - wind speed

casual - number of non-registered user rentals initiated

registered - number of registered user rentals initiated

count - number of total rentals



# submission

datetime,count

2011-01-20 00:00:00,0

"""
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
# load libraries

import numpy as np

import pandas as pd

import warnings

import seaborn as sn

import datetime

import matplotlib.pyplot as plt

import missingno as msno

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
# load data

train_df = pd.read_csv("../input/bike-sharing-demand/train.csv")

test_df = pd.read_csv("../input/bike-sharing-demand/test.csv")
# shape of data

train_df.shape
# get the head of data

train_df.head()
# get the summary of data types and count of null values

train_df.info()
# process data 

train_df['date'] = train_df.datetime.apply(lambda x : x.split()[0]) # date 2011-01-01

train_df['hour'] = train_df.datetime.apply(lambda x : x.split()[1].split(":")[0]) # hour 00~

# get the weekday from (2010, 52, 6) >> the 2nd one

train_df['weekday'] = train_df['date'].apply(lambda x : datetime.date([int(x) for x in x.split('-')][0],[int(x) for x in x.split('-')][1],[int(x) for x in x.split('-')][2]).isocalendar()[1])

train_df['month'] = train_df['date'].apply(lambda x : x.split('-')[1]) # month 01~12

train_df['season'] = train_df.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })

train_df["weather"] = train_df.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })

# get head from processed data

train_df.head(2)
# category list

categoryVariableList = ["season","holiday","workingday","weather","hour","weekday","month"]

# set data type

for var in categoryVariableList:

    train_df[var] = train_df[var].astype("category")
# axis 1 >> drop the column

train_df  = train_df.drop(["datetime"],axis=1) 
# check data type count

pd.DataFrame(train_df.dtypes.value_counts())
train_df.head(1)
# reset index

pd.DataFrame(train_df.dtypes.value_counts()).reset_index()
# rename columns

dataTypeDf = pd.DataFrame(train_df.dtypes.astype(str).value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})
dataTypeDf
# Q1. to include multiple graphs >> use subplots ? & ax?? and draw barplot

fig,ax = plt.subplots()

fig.set_size_inches(12,5)

sn.barplot(data=dataTypeDf,x="variableType",y="count",ax=ax)

ax.set(xlabel='variableType', ylabel='Count',title="Variables DataType Count")
# visualize missing values

msno.matrix(train_df,figsize=(12,5)) # train_df.shape = (10886, 12)