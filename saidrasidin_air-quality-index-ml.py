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
df_train = pd.read_csv('/kaggle/input/air-quality-index/train.csv')

df_test = pd.read_csv('/kaggle/input/air-quality-index/test.csv')

print('Training data info',df_train.info())

print(10*'--')

print('Testing data info:',df_test.info())
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import OrdinalEncoder



%matplotlib inline
#looking the data

df_train.head(10)
# change date column from object to date-time format

df_train['date_time'] = pd.to_datetime(df_train['date_time'])



#drop holiday column because most of the value is none (normal day) therefore has no significant impact

df_train.drop('is_holiday', axis=1, inplace=True)



#change wather_type column to representative number using Ordinal encoder

encoder = OrdinalEncoder()

weather = df_train[['weather_type']]

weather_enc = encoder.fit_transform(weather)

df_train['weather_type'] = weather_enc
df_train.head(2)
encoder.categories_
df_train.describe()
plt.figure(figsize=(10,10))

plt.scatter(df_train['date_time'], df_train['air_pollution_index'], marker='o', alpha=0.5);
date = df_train['date_time']

print(date.min(),'|||||' ,date.max())
#avarging value to daily instead of hourly

df_train.set_index('date_time', inplace=True)

df_train_D = df_train.resample('D').mean()

df_train_D.head(2)

#weather_type column now to be avarge value too, i keep this value to see how it impact the model later
df_train_D.tail(3)
df_train_D.dropna(axis=0, inplace=True)

df_train_D.isna().sum()
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.scatter(df_train_D.index, df_train_D['air_pollution_index'], alpha=0.5, c='g', )

plt.title('AQI'); plt.ylabel('aqi')



plt.subplot(1,2,2)

plt.scatter(df_train_D.index, df_train_D['temperature'])

plt.title('Temperature'); plt.ylabel('temp');
sns.distplot(df_train_D['air_pollution_index'],);
sns.pairplot(df_train_D)
plt.figure(figsize=(10,10))

sns.heatmap(df_train_D.corr(), annot=True, fmt='0.3f', square=True)
#since weather type is ordinal number that has been avaraging and has higher corellation with clouds_all so i just drop it

df_train_D.drop('weather_type', axis=1, inplace=True)

#splitting data and its label

X_train, y_train = df_train_D.drop(['air_pollution_index'], axis=1), df_train_D['air_pollution_index']
from sklearn.preprocessing import StandardScaler, MinMaxScaler



S_scaler = StandardScaler()

MX_Scaler = MinMaxScaler ()

X_train_S_Scal = S_scaler.fit_transform(X_train)

X_train_MX_Scal = MX_Scaler.fit_transform(X_train)
X_train_S_Scal.shape
models = {'lrg':LinearRegression(), 'SVR': SVR(), 'KNN':KNeighborsRegressor(), 'RFR':RandomForestRegressor(), 'GBR': GradientBoostingRegressor(),

         'Tree': DecisionTreeRegressor()}



def model_evaluate(model, X_train, y_train):

    score = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)

    return np.sqrt(-score)



def scoring_mean_std(val):

    mse = {}

    for key, value in val.items():

        mse[key] = (value.mean(), value.std())

    return mse
def modelc_cv_score(X_train, y_train):

    val= {}

    for key, value in models.items():

        val[key] = model_evaluate(value, X_train, y_train)

    return val

val_MinMax_Scaller = modelc_cv_score(X_train_MX_Scal, y_train=y_train)

val_SS_Scaller = modelc_cv_score(X_train_S_Scal, y_train)
scoring_mean_std(val_MinMax_Scaller)
scoring_mean_std(val_SS_Scaller)
#SVR Has minimum score, it's tuning the parameters now

from sklearn.model_selection import GridSearchCV



svr = SVR()

param = [{"C": [0.01, 0.1, 0.5, 1, 10, 100], 'degree':[3,4,5,6,7]}]

grid_search = GridSearchCV(svr, param_grid=param, scoring='neg_mean_squared_error' ,return_train_score=True, cv=10)

grid_search.fit(X_train_MX_Scal, y_train)

grid_search.best_estimator_
svr = SVR(C=0.01, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',

          kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

svr.fit(X_train_MX_Scal, y_train)
# change date column from object to date-time format

df_test['date_time'] = pd.to_datetime(df_test['date_time'])



#drop holiday column because most of the value is none (normal day) therefore has no significant impact

df_test.drop(['is_holiday', 'weather_type'], axis=1, inplace=True)



df_test.set_index('date_time', inplace=True)

df_test_D = df_test.resample('D').mean()

df_test_D.head(2)
df_test_D.isnull().sum()
df_test_scaled = MX_Scaler.transform(df_test_D)



df_test_D['AQI'] = svr.predict(df_test_scaled)

df_test_D
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.distplot(df_train_D['air_pollution_index'], norm_hist=True)

plt.title('Training Data')

plt.subplot(1,2,2)

sns.distplot(df_test_D['AQI'], norm_hist=True);

plt.title('Testing Data (Pediction)');
df_test_D['AQI'].median()
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)

plt.scatter(df_test_D.index, df_test_D['AQI'], alpha=0.5, c='g', )

plt.title('AQI'); plt.ylabel('aqi')



plt.subplot(1,3,2)

plt.scatter(df_test_D.index, df_test_D['temperature'])

plt.title('Temperature'); plt.ylabel('temp');



plt.subplot(1,3,3)

plt.scatter(df_test_D.index, df_test_D['traffic_volume'])

plt.title('Trafficc'); plt.ylabel('Traffic Volume');