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
#Import relevant libraries

import pandas as pd

import glob

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt

%matplotlib inline



#Timeseries modeling

from fbprophet import Prophet



#Evaluation Params

from sklearn.metrics import mean_squared_error, mean_absolute_error
#Import the data and get df ready for energy consumption. Prepare a combined dataframe.

path = '/kaggle/input/ontario-energy-prices/'

all_files = glob.glob(path + "/*.csv")



li = [] #empty list to collect the data.



for filename in all_files:

    df_full = pd.read_csv(filename, index_col=None, header=0, skiprows=3, parse_dates=['Date'], encoding='latin')

    li.append(df_full)



df_full = pd.concat(li, axis=0, ignore_index=True)
#Subet Dataframe to only relevant features.

df = df_full[['Date','Hour','HOEP']]

del df_full
#Sorting and resetting indexing as hygiene.

df.sort_values('Date', inplace=True)

df.reset_index(drop=True, inplace=True)
#Let's have a look.

df.head()
#2.1 - Replace , in thousands using string object and convert to float type.

df['HOEP']= df['HOEP'].astype(str).str.replace(',', '').astype(float)



#2.2 - Change 24 to 0 since the date does not change with every month's 24 hour time.

#df['Hour'] = df['Hour'].map(lambda x:0 if x==24 else x)



#2.3 - Concatenate the hours and date to get a datetime object.

df['date'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')



#2.4 - Sort Values by Date

df.sort_values('date', ascending=True, inplace=True)



#2.5 - Set index as dates.

df.set_index('date', drop=True,inplace=True)
#Dropping old 'Date' and 'Hour' features.

df.drop(['Date','Hour'], axis=1,inplace=True)
#Flooring the data as it has neg values.

df['HOEP'] = df['HOEP'].map(lambda x:0 if x<0 else x)

df['HOEP'] = df['HOEP'].map(lambda x:.0001 if x==0 else x)
#Extracting all the features using TimeSeries object. The output is a combined_df with all the features.



def extract_features(df, label=None):

    df = df.copy()

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    if label:

        y = df[label]

        return X, y

    return X

X, y = extract_features(df, label='HOEP')

combined_df = pd.concat([X, y], axis=1)



#Shout out to https://www.kaggle.com/robikscube/time-series-forecasting-with-prophet#Data.
#Plotting a pairplot to see the relationship bw features.

sb.pairplot(combined_df.dropna(), hue='hour', x_vars=['hour','dayofweek','year','weekofyear'], y_vars='HOEP',

             height=6, plot_kws={'alpha':0.8, 'linewidth':0},palette= "husl")

#Plot title

plt.suptitle('Ontario Power Use by Hour, Day of Week, Year and Week of Year')

plt.show()
#The data is for 17+ years. Great.

df.shape[0]/24/365
print ("Lowest Date in Dataset: ", min(df.index))

print ("Highest Date in Dataset: ", max(df.index))
#Splitting the data on 1st Jan 2017.

split_date = '01-Jan-2017'

df_train = df.loc[df.index <= split_date]

df_test = df.loc[df.index > split_date]
#Plotting the split of the dataset.

temp = df_test.rename(columns={'HOEP': 'Test Data'})

temp2 = temp.join(df_train.rename(columns={'HOEP': 'Train Data'}), how='outer')

temp2.plot(figsize=(15,5), title='Test and Train Split', style='.', color=['grey','blue'])

plt.show()

del temp,temp2

#Preprocessing for model specifications.



#Prophet also imposes the strict condition that the input columns be named ds (the time column) and 

#y (the metric column) so let’s rename the columns in our DataFrame:

df_train = df_train.reset_index().rename(columns={'date':'ds', 'HOEP':'y'})

df_train.head()
# Setup and Train model and fit



#Can set the uncertainty interval to 95% (the Prophet default is 80%)

model = Prophet() 

model.fit(df_train.reset_index().rename(columns={'date':'ds','HOEP':'y'}))
#Predict on test set.

HOEP_predictions = model.predict(df=df_test.reset_index().rename(columns={'date':'ds'}))

#Note that the predictions start from 01-01-2017 (the split date) test set values untill last data point 06-02-2020.



#Check Predictions.

HOEP_predictions
# Plot the components of the prophet model.

fig = model.plot_components(HOEP_predictions)

#Observe how the prices are lower during summer months - this could be due to people being out/requiring lesser hvac.

#The coldest months in Ontario have the steepest energy prices. 
# Plot the Forecast and Actuals



#Plot params

f, ax = plt.subplots(1)

f.set_figheight(10)

f.set_figwidth(45)



#Plot scatter plot of actual vs forecasts.

ax.scatter(df_test.index, df_test['HOEP'], color='g') #Red color is forecasted values.

fig = model.plot(HOEP_predictions, ax=ax)
#Since the split_date = '01-Jan-2017', let's check the forecasts for the first month i.e 1st Jan'17 to 1st Feb'17



# Plot the forecast with the actuals

f, ax = plt.subplots(1)

f.set_figheight(5)

f.set_figwidth(25)

ax.scatter(df_test.index, df_test['HOEP'], color='y')

fig = model.plot(HOEP_predictions, ax=ax)



import datetime



#Set monthly limit.

ax.set_xlim([datetime.date(2017, 1, 1), datetime.date(2017, 2, 1)])

ax.set_ylim([-100,250 ])

plot = plt.suptitle('January 2017 Forecast vs Actuals')



#The model seems to be doing okay with the current tweaks. Let's evaluate it using other metrics.
#Since the split_date = '01-Jan-2017', let's check the forecasts for the first WEEK.



# Plot the forecast with the actuals

f, ax = plt.subplots(1)

f.set_figheight(5)

f.set_figwidth(25)

ax.scatter(df_test.index, df_test['HOEP'], color='b')

fig = model.plot(HOEP_predictions, ax=ax)



import datetime



#Set monthly limit.

ax.set_xlim([datetime.date(2017, 1, 1), datetime.date(2017, 1, 7)])

ax.set_ylim([-100,250 ])

plot = plt.suptitle('Weekly Prediction for Jan17')



#The model seems to be doing okay with the current tweaks. Let's evaluate it using other metrics.
#MSE



#For each point, it calculates square difference between the predictions and the target and then average 

#those values.

mse_prophet = mean_squared_error(y_true=df_test['HOEP'],

                   y_pred=HOEP_predictions['yhat'])

mse_prophet
#Mean Absolute Error - Avg of absolute distance. Not as sensitive to outliers as MSE.

mae_prophet = mean_absolute_error(y_true=df_test['HOEP'],

                   y_pred=HOEP_predictions['yhat'])

mae_prophet
#Next Steps - Try another model (ARIMA) and Deep Learning.



#Happy Learning!