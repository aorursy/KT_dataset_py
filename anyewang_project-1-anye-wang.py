# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import datetime
df = pd.read_csv("../input/cee-498-project1-london-bike-sharing/train.csv")
df.head()
df.dtypes
df.info()
#Season Consideration

df_season = df

df_season['season'] = df['season'].map({0:'spring', 1:'summer', 2:'autumn', 3:'winter'})

df_season = df_season.groupby('season').mean()



plt.figure()

plt.bar(df_season.index, df_season['cnt'])

plt.xlabel("Season")

plt.ylabel("cnt")

plt.suptitle ("Bike Numbers in Different Seasons")

plt.xticks (rotation = 90)

plt.show()
#Weather Consideration

df_weather = df

df_weather['weather'] = df['weather_code'].map({1:'Clear', 2:'Few Clouds', 3:'Broken Clouds', 4:'Cloudy', 7:'Rain', 10:'Stormy', 26: 'Snow', 94:'Freezing Fog'})

df_weather = df_weather.drop(['weather_code'], axis =1)



df_weather = df_weather.groupby('weather').mean()



plt.figure()

plt.bar(df_weather.index, df_weather['cnt'])

plt.xlabel("Weathers")

plt.ylabel("cnt")

plt.suptitle ("Bike Numbers in Different Weathers")

plt.xticks (rotation = 90)

plt.show()
#Time Consideration

df_time = df

df_time['timestamp'] = pd.to_datetime(df['timestamp'])



df_time['hour'] = df['timestamp'].dt.hour

df_time['month'] = df['timestamp'].dt.month

df_time['day'] = df['timestamp'].dt.day

df_time['weekday'] = df['timestamp'].dt.dayofweek



df_time = df_time.set_index('timestamp')
#Each Day

plt.figure()

sns.lineplot (data = df_time, x = df_time.index, y = df_time.cnt)

plt.suptitle ("Bike Numbers with Date")

plt.xticks(rotation = 90)

plt.show()
#By hour

df_by_hour = df_time.groupby('hour').mean()



plt.figure()

sns.lineplot(data = df_by_hour, x = df_by_hour.index, y = df_by_hour.cnt)

plt.xticks(rotation = 90)

plt.show()
#By month

df_by_month = df_time.groupby('month').sum()



plt.figure()

sns.lineplot(data = df_by_month, x = df_by_month.index, y = df_by_month.cnt)

plt.xticks(rotation = 90)

plt.show()
#By day

df_by_day = df_time.groupby('day').mean()



plt.figure()

sns.lineplot(data = df_by_day, x = df_by_day.index, y = df_by_day.cnt)

plt.xticks(rotation = 90)

plt.show()
#By week

df_by_week = df_time.groupby('weekday').mean()



plt.figure()

sns.lineplot(data = df_by_week, x = df_by_week.index, y = df_by_week.cnt)

plt.xticks(rotation = 90)

plt.show()
# Holiday and Nonholiday

df_holiday = df

df_holiday['holiday'] = df['is_holiday'].map({0:'non holiday', 1:'holiday'})

df_holiday = df_holiday.groupby('holiday').mean()



plt.figure()

plt.bar(df_holiday.index, df_holiday['cnt'])

plt.xlabel("Holiday")

plt.ylabel("cnt")

plt.suptitle ("Bike Numbers in holiday and non holiday")

plt.xticks (rotation = 90)

plt.show()
#Weekend and Nonweekend

df_weekend = df

df_weekend['weekend'] = df['is_weekend'].map({0:'non weekend', 1:'weekend'})

df_weekend = df_weekend.groupby('weekend').sum()



plt.figure()

plt.bar(df_weekend.index, df_weekend['cnt'])

plt.xlabel("Weekend")

plt.ylabel("cnt")

plt.suptitle ("Bike Numbers in weekend and non weekend")

plt.xticks (rotation = 90)

plt.show()
df.corr()
sns.heatmap(df.corr())