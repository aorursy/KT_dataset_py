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
########loading required packages#############

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as scist

import time



############Hypothesis Generation##########

"""

1. House's Price decreased in year time period

2. House's Price increased in year time period

3. House's price effect on monthly basis

4. House's price effect on Daily basis

"""



########load dataset#########

data_original = pd.read_csv("/kaggle/input/median-home-sale-price/Sale_Prices_City.csv")

data=data_original.copy() 



######features of data#######

data.columns #keys()
########Data types###########

data.dtypes
########Shape of data########

data.shape
########data sample##########

data.head()
#############count of null values of data########

data.isnull().sum()
#################Overview of dataset#################

data.describe().transpose()
###################Handle missing values#####################

data = data.groupby(['StateName'],sort=False).apply(lambda x: x.fillna(x.mean()))

data = data.fillna(0)

#############count of null values of data########

data.isnull().sum()
#Reshaping the data to move columns for each month into rows

data = data.melt(id_vars=['RegionID', 'RegionName', 'StateName', 'SizeRank'], var_name="Date", value_name="Price")
########data sample##########

data.head()


############Feature Extraction##############

"""

We will convert data type of Date is object to datetime format and extract the time and date from the Datetime

"""

data['Date'] = pd.to_datetime(data.Date,format='%Y-%m-%d') 



#Splitting Date column into individual Year, Month and Day columns

data['Year'] = data.Date.dt.year 

data['Month'] = data.Date.dt.month 

data['Day'] = data.Date.dt.day 
#########Exploratory Analysis############

##########Plot Data on yearly House's Price###########

"""

House's Price decreased from 2009 to 2011 year time period

House's Price increased from 2013 to 2019 year time period

"""

data.groupby('Year')['Price'].mean().plot.bar()

##########Plot Data on monthly House's Price###########

"""

House's price does not effect on monthly basis

"""

data.groupby('Month')['Price'].mean().plot.bar()
##########Plot Data on daily House's Price###########

"""

House's price does not effect on daily basis

"""

data.groupby('Day')['Price'].mean().plot.bar()
##########Plot Data on Date House's Price###########

###########line chart############

temp=data.groupby(['Year', 'Month'])['Price'].mean() 

temp.plot(figsize=(15,5), title= 'Price(Monthwise)', fontsize=14)
####### build models for Time Series Forecasting########



"""

######Splitting the data into training and validation part#########

we will need a dataset(validation) to check the performance and generalisation ability of our model.

The dataset should have the true values of the dependent variable against

which the predictions can be checked. Therefore, test dataset cannot be used for the purpose.



The model should not be trained on the validation dataset.

Hence, we cannot train the model on the train dataset and validate on it as well.



So, for the above two reasons, we generally divide the train dataset into two parts.

One part is used to train the model and the other part is used as the validation/test dataset.



To divide the data into training and validation set,

we will take last 36 months as the validation data and rest for training data. 

"""



Train=data.ix[:'2017-12-01 00:00:00']

valid=data.ix['2017-12-01 00:00:00':]



Train.Price.plot(figsize=(15,8), title= "House's Price", fontsize=14, label='train')

valid.Price.plot(figsize=(15,8), title= "House's Price", fontsize=14, label='valid')

plt.xlabel("Date")

plt.ylabel("House's Price")

plt.legend(loc='best')

plt.show()

########## Naive Approach###########

"""

In this forecasting technique, we assume that the next expected point is equal to the last observed point.

So we can expect a straight horizontal line as the prediction. 

"""



dd= np.asarray(Train.Price) 

y_hat = valid.copy() 

y_hat['naive'] = dd[len(dd)-1] 

plt.figure(figsize=(12,8)) 

plt.plot(Train.index, Train['Price'], label='Train') 

plt.plot(valid.index,valid['Price'], label='Valid') 

plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 

plt.legend(loc='best') 

plt.title("Naive Forecast") 

plt.show()
#########calculate accuracy of predictions are using rmse(Root Mean Square Error)###############

"""

rmse is the standard deviation of the residuals.

Residuals are a measure of how far from the regression line data points are.

"""

from sklearn.metrics import mean_squared_error 

from math import sqrt 

rms = sqrt(mean_squared_error(valid.Price, y_hat.naive)) 

print(rms)

###########Moving Average###########

"""

we will take the average of the passenger counts for last few time periods only.

the predictions are made on the basis of the average of last few points instead of

taking all the previously known values

"""



y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Price'].rolling(10).mean().iloc[-1] # average of last 10 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Price'], label='Train') 

plt.plot(valid['Price'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations') 

plt.legend(loc='best') 

plt.show() 

y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Price'].rolling(20).mean().iloc[-1] # average of last 20 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Price'], label='Train') 

plt.plot(valid['Price'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations') 

plt.legend(loc='best') 

plt.show() 
y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Price'].rolling(50).mean().iloc[-1] # average of last 50 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Price'], label='Train') 

plt.plot(valid['Price'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(valid.Price, y_hat_avg.moving_avg_forecast)) 

print(rms)

###########Simple Exponential Smoothing###########

"""

In this technique, we assign larger weights to more recent observations

than to observations from the distant past.

The weights decrease exponentially as observations come from further in the past,

the smallest weights are associated with the oldest observations.

"""



from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

y_hat_avg = valid.copy() 

fit2 = SimpleExpSmoothing(np.asarray(Train['Price'])).fit(smoothing_level=0.6,optimized=False)

y_hat_avg['SES'] = fit2.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['Price'], label='Train') 

plt.plot(valid['Price'], label='Valid') 

plt.plot(y_hat_avg['SES'], label='SES') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(valid.Price, y_hat_avg.SES)) 

print(rms)