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
#loading packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import tree

from scipy.stats import binned_statistic

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

from numpy import inf

import seaborn as sns

%matplotlib inline

#reading in train and test data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()
test.head()
print(train.shape)

print(test.shape)
train.dtypes
#combining both train and test data for further understanding

test['registered'] = 0

test['casual'] = 0

test['count'] = 0

data = pd.concat([train, test], sort = False)

data.info()
#checking for missing values

combined_data.isnull().values.ravel().sum()
num_var = data.select_dtypes(include = [np.number]).columns.tolist()

num_var
#Histogram for count

sns.set_style('darkgrid')

sns.distplot(train['count'], bins = 50, color = 'red')

plt.show()
#Q-Q Plot

from scipy import stats

plt = stats.probplot(train['count'], plot=sns.mpl.pyplot)
#Boxplot for count

import matplotlib.pyplot as plt

sns.boxplot(x = 'count', data = train, color = 'gold')

plt.show()
Q1 = train['count'].quantile(0.25)

Q3 = train['count'].quantile(0.75)

IQR = Q3 - Q1

outliers = train[(train['count'] < (Q1 - 1.5 * IQR)) | (train['count'] > (Q3 + 1.5 * IQR))]

print((len(outliers)/len(data))*100)
#Data without the outliers in count

data = data[~data.isin(outliers)]

data = data[data['datetime'].notnull()]
sns.barplot(x = 'season', y = 'count', data = train, estimator = np.average, palette='coolwarm')

plt.ylabel('Average Count')

plt.show()
sns.barplot(x = 'holiday', y = 'count', data = train, estimator = np.average, palette='deep')

plt.ylabel('Average Count')

plt.show()
sns.barplot(x = 'workingday', y = 'count', data = train, estimator = np.average, palette='colorblind')

plt.ylabel('Average Count')

plt.show()
sns.barplot(x = 'weather', y = 'count', data = train, estimator = np.average, palette='deep')

plt.ylabel('Average Count')

plt.show() 
plt.figure(figsize = (10,7))

tc = train.corr()

sns.heatmap(tc, annot = True, cmap = 'coolwarm', linecolor = 'white', linewidths=0.1)
#Convert to integer variables

columns=['season', 'holiday', 'workingday', 'weather']

for i in columns:

    data[i] = data[i].apply(lambda x : int(x))
#Convert string to datatime and create Hour, Month and Day of week

data['datetime'] = pd.to_datetime(data['datetime'])

data['Hour'] = data['datetime'].apply(lambda x:x.hour)

data['Month'] = data['datetime'].apply(lambda x:x.month)

data['Day of Week'] = data['datetime'].apply(lambda x:x.dayofweek)
plt.figure(figsize = (8,4))

sns.lineplot(x = 'Month', y = 'count', data = data, estimator = np.average, hue = 'weather', palette = 'coolwarm')

plt.ylabel('Average Count')

plt.show()
data[data['weather'] == 4]
fig, axes = plt.subplots(ncols = 2, figsize = (15,5), sharey = True)

sns.pointplot(x = 'Hour', y = 'count', data = data, estimator = np.average, hue = 'workingday', ax = axes[0], palette = 'muted')

sns.pointplot(x = 'Hour', y = 'count', data = data, estimator = np.average, hue = 'holiday', ax = axes[1], palette = 'muted')

ax = [0,1]

for i in ax:

    axes[i].set(ylabel='Average Count')
plt.figure(figsize = (10,4))

sns.pointplot(x = 'Hour', y = 'count', data = data, estimator=np.average, hue = 'Day of Week', palette='coolwarm')
sns.jointplot(x = 'atemp', y = 'count', data = data, kind = 'kde', cmap = 'plasma')

plt.show()

plt.figure(figsize = (8,4))

sns.pointplot(x = 'Hour', y = 'casual', data = data, estimator = np.average, color = 'blue')

sns.pointplot(x = 'Hour', y = 'registered', data = data, estimator = np.average, color = 'red')

plt.ylabel('Registered')

plt.show()
#Histogram for Windspeed

sns.set_style('darkgrid')

sns.distplot(data['windspeed'], bins = 50, color = 'green') #Windspeed cannot be 0.

plt.show()
#Replacing 0s in windspeed with the mean value grouped by season

data['windspeed'] = data['windspeed'].replace(0, np.nan)

data['windspeed'] = data['windspeed'].fillna(data.groupby('weather')['season'].transform('mean'))

sns.distplot(data['windspeed'], bins = 50, color = 'red')

plt.show()


#Encoding cyclical features

data['Month_sin'] = data['Month'].apply(lambda x: np.sin((2*np.pi*x)/12))

data['Month_cos'] = data['Month'].apply(lambda x: np.cos((2*np.pi*x)/12))

data['Hour_sin'] = data['Hour'].apply(lambda x: np.sin((2*np.pi*(x+1))/24))

data['Hour_cos'] = data['Hour'].apply(lambda x: np.cos((2*np.pi*(x+1))/24))

data['DayOfWeek_sin'] = data['Day of Week'].apply(lambda x: np.sin((2*np.pi*(x+1))/7))

data['DayOfWeek_cos'] = data['Day of Week'].apply(lambda x: np.cos((2*np.pi*(x+1))/7))
#trainsforming target variable using log transformation

data['count'] = np.log(data['count'])
#Converting Categorical to numerical - Removing Co-Linearity

data_ = pd.get_dummies(data=data, columns=['season', 'holiday', 'workingday', 'weather'])

train_ = data_[pd.notnull(data_['count'])].sort_values(by=["datetime"])

test_ = data_[~pd.notnull(data_['count'])].sort_values(by=["datetime"])
#Standardizing numerical variables

from sklearn.preprocessing import StandardScaler

cols = ['temp','atemp','humidity', 'windspeed', 'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin','DayOfWeek_cos']

features = data[cols]
#Standard Scaler

scaler = StandardScaler().fit(features.values)

data[cols] = scaler.transform(features.values)
#Predictor columns names

cols = ['temp','atemp','humidity', 'windspeed', 'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin','DayOfWeek_cos', 'season_1','season_2', 'season_3',

        'holiday_0', 'workingday_0', 'weather_1', 'weather_2', 'weather_3']
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
#train test split

X = train_[cols]

y = train_['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)