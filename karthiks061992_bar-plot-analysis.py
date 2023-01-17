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
dataset=pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')

dataset.head()
import seaborn as sn

import matplotlib.pyplot as plt

dataset.Region.value_counts()
dataset.isnull().sum()

dataset.drop(columns=['State'],axis=1,inplace=True)
dataset.head()
dataset.isnull().sum()#NO null values 

plt.figure(figsize=(20,10))

dataset.groupby('Region')['AvgTemperature'].mean().sort_values(ascending=False).plot.bar(color='red')

#Highest average temperature depending on regions lead by middle asia 
dataset.groupby(['Country'])['AvgTemperature'].count().sort_values(ascending=False).head(7).plot.bar(color='orange')

#US stands on highest number of measurements in temperatures

plt.figure(figsize=(14,6))

dataset.groupby('Year')['AvgTemperature'].mean().sort_values(ascending=False).head(5).plot.bar(color="pink")

#Highest temperature recorded according to years

plt.figure(figsize=(14,6))

dataset.groupby(['Year','Month'])['AvgTemperature'].mean().sort_values(ascending=False).head(5).plot.bar(color="black")

#Highest recorded average temperature in the previous time.
dataset.head()
dataframe1=dataset.loc[:,['Year','AvgTemperature']]

dataframe1.head()

plt.figure(figsize=(14,6))

import seaborn as sn

z=dataframe1.groupby('Year')['AvgTemperature'].mean().sort_values(ascending=False).head().plot.line()
dataset.head()

plt.figure(figsize=(14,6))

#Analysis based on city the highest temperature wise city

dataset.groupby('City')['AvgTemperature'].mean().sort_values(ascending=False).head(4).plot.bar(color="brown")
#usually the month with the highest temperature

plt.figure(figsize=(14,6))

dataset.groupby('Month')['AvgTemperature'].mean().sort_values(ascending=False).head(4).plot.bar()

#July,August,June and September