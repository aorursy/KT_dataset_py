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
import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option("display.max_columns", 80)
bike = pd.read_csv('/kaggle/input/bike-sharing-dataset/day.csv')
bike.head()
bike.shape
bike.isnull().sum()
bike.rename(columns={'holiday':'is_holiday',

                        'workingday':'is_workingday',

                        'weathersit':'weather_condition',

                        'hum':'humidity',

                        'mnth':'month',

                        'cnt':'count',

                        'yr':'year'},inplace=True)
bike.info()
bike['month'] = bike['month'].map({1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 

                                   9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"})



bike['weather_condition'] = bike['weather_condition'].map({1:"Clear", 2:"Cloudy", 3:"Light Rain", 4:"Heavy Rain"})



bike['season'] = bike['season'].map({1:"spring", 2:"summer", 3:"fall", 4:"winter"})



dmap = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',0:'Sun'}

bike['weekday'].astype('object')

bike['weekday'] = bike['weekday'].map(dmap)



figure, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4)

figure.set_size_inches(20, 10)



sns.boxplot(data=bike, y='count', ax=ax1)

sns.boxplot(data=bike, x='year', y='count', ax=ax2)

sns.boxplot(data=bike, x='month', y='count', ax=ax3)

sns.boxplot(data=bike, x='weekday', y='count', ax=ax4)

sns.boxplot(data=bike, x='season', y='count', ax=ax5)

sns.boxplot(data=bike, x='is_holiday', y='count', ax=ax6)

sns.boxplot(data=bike, x='is_workingday', y='count', ax=ax7)

sns.boxplot(data=bike, x='weather_condition', y='count', ax=ax8)

categorical_cols = ['year', 'month', 'weekday', 'season', 'is_holiday', 'is_workingday', 'weather_condition']

plt.figure(figsize = (20, 10))



for i in enumerate(categorical_cols):

    plt.subplot(2, 4, i[0]+1)

    plt.tight_layout(pad=0.5)

    sns.boxplot(x = i[1], y= 'count', data = bike)
bike.drop(columns=['instant', 'dteday', 'temp', 'casual', 'registered'], inplace=True)


plt.figure(figsize = (15, 5))



continuous_cols = ['atemp', 'humidity', 'windspeed']

for i in enumerate(continuous_cols):

    plt.title('Correlation between continuous variable & count')

    plt.subplot(1, 3, i[0]+1)

    plt.tight_layout(pad=0.5)

    sns.regplot(x=i[1], y='count', data=bike)
import seaborn as sns



sns.pairplot(data=bike, x_vars=['atemp', 'humidity', 'windspeed'], y_vars=['count'])
corr = bike[['atemp', 'humidity', 'windspeed', 'count']].corr()

sns.heatmap(data=corr,

           vmax=1.0,

           vmin=-1.0,

           annot=True)