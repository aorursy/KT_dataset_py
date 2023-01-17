# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# importing libraries

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)



import calendar



from datetime import datetime



import matplotlib

from matplotlib import pyplot as plt



import seaborn as sns



%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_sampleSubmission = pd.read_csv('../input/sampleSubmission.csv')
print('Training data shape',df_train.shape)

print('Test data shape',df_test.shape)

print('Training data set header')

df_train.head()
print('Training data set data types')

df_train.dtypes
# to apply a function on each value in a series, use apply and lambda as below. [0] in the end is index of 

# first element in split.

df_train['date']=df_train['datetime'].apply(lambda x: x.split()[0])



df_train['Hour']=df_train['datetime'].apply(lambda x:x.split()[1].split(':')[0])
df_train['weekday']=df_train['date'].apply(lambda x: calendar.day_name[datetime.strptime(x,'%Y-%m-%d').weekday()])

# In datetime library, strptime is to create a datetime object a date into year, month, date, hour and minute. 

# Weekday returns the weekday of that date. with 0 as Monday

datetime.strptime(df_train['datetime'][2],'%Y-%m-%d %H:%M:%S').weekday()
df_train['Month'] = df_train['date'].apply(lambda x: calendar.month_name[datetime.strptime(x,'%Y-%m-%d').month])

df_train['season'] = df_train['season'].map({1:'Spring',2:'Summer',3:'Fall',4:'Winter'})

df_train['weather'] = df_train['weather'].map({1:'Clear + Few clouds + Partly cloudy + Partly cloudy',

                                               2:'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',

                                               3:'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',

                                               4:'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog'})
categorical_columns=['season','weather','Hour','weekday','Month','holiday','workingday']



for col in categorical_columns:

    df_train[col] = df_train[col].astype('category') 



df_train.dtypes
df_train = df_train.drop(['datetime'],axis=1)
print('Step 1: Find missing values')

print('Is there any missing value?',df_train.isnull().values.any())

print('----Step2: Outlier analysis----')



print('Using box plot we can find the outliers')



#returns a figure with 4 subplots (2 on each row)

fig,ax= plt.subplots(nrows=2,ncols=2)

#set figure size

fig.set_size_inches(12,10)

sns.boxplot(data=df_train,y='count',orient='v',ax=ax[0][0])

sns.boxplot(data=df_train,y='count', x='season', orient='v',ax=ax[0][1])

sns.boxplot(data=df_train,y='count', x='Hour',orient='v',ax=ax[1][0])

sns.boxplot(data=df_train,y='count', x='workingday',orient='v',ax=ax[1][1])



ax[0][0].set(ylabel='Count', title = 'Plot on Count')

ax[0][1].set(ylabel='Count', xlabel = 'Season', title = 'Plot on Season vs Count')

ax[1][0].set(ylabel='Count', xlabel = 'Hour of day', title = 'Plot on Hour of day vs Count')

ax[1][1].set(ylabel='Count', xlabel = 'Working day', title = 'Plot on Working day(or not) vs Count')
df_train_corr = df_train.corr()

df_train_corr
mask = np.array(df_train_corr)

#this will make half of the array as 0

mask[np.tril_indices_from(mask)]=False

fig,ax = plt.subplots()

fig.set_size_inches(12,10)

#cmap colors - Yellow, Green, blue

#mask is an array. where value is missing in mask array, nothing will be drawn.

#square is True. each box is a square

# annot is true. each box has a value.

sns.heatmap(df_train_corr,mask=mask,vmax=0.8, square=True,annot=True, cmap='YlGnBu' ,ax=ax)
fig,ax = plt.subplots(ncols=3)

fig.set_size_inches(20,8)

sns.regplot(data = df_train,x='temp',y='count',ax=ax[0])

sns.regplot(data = df_train,x='humidity', y='count',ax=ax[1])

sns.regplot(data = df_train,x='windspeed',y='count',ax=ax[2])
from sklearn.ensemble import RandomForestRegressor
df_train.dtypes

columns_for_model = ['season','holiday','workingday','weather','temp','humidity','Hour','weekday','Month']
fig,ax = plt.subplots()

fig.set_size_inches(12,8)

df_groupingSeasonHour = pd.DataFrame(df_train.groupby(by=['Hour','season'],sort=True)['count'].mean()).reset_index()

sns.pointplot(data=df_groupingSeasonHour,x=df_groupingSeasonHour['Hour'],y=df_groupingSeasonHour['count'], 

              hue=df_groupingSeasonHour['season'],join=True,ax=ax)

fig,ax = plt.subplots()

fig.set_size_inches(12,8)

df_groupingSeasonHour = pd.DataFrame(df_train.groupby(by=['Hour','holiday'],sort=True)['count'].mean()).reset_index()

sns.pointplot(data=df_groupingSeasonHour,x=df_groupingSeasonHour['Hour'],y=df_groupingSeasonHour['count'], 

              hue=df_groupingSeasonHour['holiday'],join=True,ax=ax)

fig,ax = plt.subplots()

fig.set_size_inches(12,8)

df_groupingSeasonHour = pd.DataFrame(df_train.groupby(by=['Hour','workingday'],sort=True)['count'].mean()).reset_index()

sns.pointplot(data=df_groupingSeasonHour,x=df_groupingSeasonHour['Hour'],y=df_groupingSeasonHour['count'], 

              hue=df_groupingSeasonHour['workingday'] ,join=True,ax=ax)

df_train.iloc[np.where(df_train['weekday']=='Monday')]
df_train.head()
#Training set

training_data = pd.read_csv('../input/train.csv')



training_data['date']= training_data['datetime'].apply(lambda x: x.split()[0])

training_data['Hour']=training_data['datetime'].apply(lambda x:x.split()[1].split(':')[0])

training_data['weekday']=training_data['date'].apply(lambda x: calendar.day_name[datetime.strptime(x,'%Y-%m-%d').weekday()])

training_data['Month'] = training_data['date'].apply(lambda x: calendar.month_name[datetime.strptime(x,'%Y-%m-%d').month])



training_data.head()
#dropping columns that are not required

training_data.drop(columns=['datetime', 'atemp','windspeed','casual','registered','date'],inplace=True)

training_data.head()
training_data.dtypes
training_data.weekday=training_data.weekday.astype('category')

training_data.Month = training_data.Month.astype('category')

training_data.season = training_data.season.astype('category')

training_data.holiday = training_data.holiday.astype('category')

training_data.workingday = training_data.workingday.astype('category')

training_data.weather = training_data.weather.astype('category')
weekday_dummies = pd.get_dummies(training_data['weekday'])

month_dummies = pd.get_dummies(training_data.Month)

season_dummies = pd.get_dummies(training_data.season)

season_dummies.columns = ['Spring','Summer','Fall','Winter']

weather_dummies = pd.get_dummies(training_data.weather)

weather_dummies.columns = ['Clear','Mist','Snow','Rain']



training_data = pd.concat([training_data,weekday_dummies,month_dummies,season_dummies,weather_dummies],axis=1)
training_data.drop(columns=['season','weather','weekday','Month'],inplace=True)



training_data.columns

randomForest = RandomForestRegressor(n_estimators=100)



x_trainingSet = training_data[['holiday', 'workingday', 'temp', 'humidity', 'Hour', 'Friday',

       'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',

       'April', 'August', 'December', 'February', 'January', 'July', 'June',

       'March', 'May', 'November', 'October', 'September', 'Spring', 'Summer',

       'Fall', 'Winter', 'Clear', 'Mist', 'Snow', 'Rain']]

y_trainingSet = training_data['count']



randomForest.fit(x_trainingSet,y_trainingSet)



y_hat_training = randomForest.predict(X=x_trainingSet)



from sklearn.metrics import mean_squared_error

print('Root Mean square error: ',np.sqrt(mean_squared_error(y_trainingSet,y_hat_training)))
#Testing set

testing_data = pd.read_csv('../input/test.csv')



testing_data['date']= testing_data['datetime'].apply(lambda x: x.split()[0])

testing_data['Hour']=testing_data['datetime'].apply(lambda x:x.split()[1].split(':')[0])

testing_data['weekday']=testing_data['date'].apply(lambda x: calendar.day_name[datetime.strptime(x,'%Y-%m-%d').weekday()])

testing_data['Month'] = testing_data['date'].apply(lambda x: calendar.month_name[datetime.strptime(x,'%Y-%m-%d').month])



testing_data.drop(columns=['datetime', 'atemp','windspeed','date'],inplace=True)



testing_data.weekday=testing_data.weekday.astype('category')

testing_data.Month = testing_data.Month.astype('category')

testing_data.season = testing_data.season.astype('category')

testing_data.holiday = testing_data.holiday.astype('category')

testing_data.workingday = testing_data.workingday.astype('category')

testing_data.weather = testing_data.weather.astype('category')



weekday_dummies = pd.get_dummies(testing_data['weekday'])

month_dummies = pd.get_dummies(testing_data.Month)

season_dummies = pd.get_dummies(testing_data.season)

season_dummies.columns = ['Spring','Summer','Fall','Winter']

weather_dummies = pd.get_dummies(testing_data.weather)

weather_dummies.columns = ['Clear','Mist','Snow','Rain']



testing_data = pd.concat([testing_data,weekday_dummies,month_dummies,season_dummies,weather_dummies],axis=1)



testing_data.drop(columns=['season','weather','weekday','Month'],inplace=True)



x_testSet = testing_data[['holiday', 'workingday', 'temp', 'humidity', 'Hour', 'Friday',

       'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',

       'April', 'August', 'December', 'February', 'January', 'July', 'June',

       'March', 'May', 'November', 'October', 'September', 'Spring', 'Summer',

       'Fall', 'Winter', 'Clear', 'Mist', 'Snow', 'Rain']]



y_hat_testing = randomForest.predict(X=x_testSet)

y_hat_testing.shape
testing_data.head()
testing_col = pd.read_csv('../input/test.csv')

testing_col.head()


final_dataframe = pd.DataFrame({"datetime": testing_col.datetime,"count":[int(x) for  x in y_hat_testing]})
final_dataframe.to_csv('bike_prediction.csv',index=False)