# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import scipy

import matplotlib.pyplot as plt

import sklearn

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Loading the csv file into data

data = pd.read_csv("../input/Call_Setup_Success_Rate_1month_hourly.csv")



#Renaming the column names

data.columns=['KPI','Unit','KPI-Id','Date','Hour','NE-Id', 'Value']



#Checking for Null Values

print("Null Values\n")

print(data.isnull().sum())



#To prevent truncation of columns

pd.set_option('display.max_columns', 30)



data.head()
#Deleting the KPI, Unit and KPI-Id Columns 



del data['KPI']

del data['Unit']

del data['KPI-Id']



#Displaying the first and last 5 rows of the dataset

print("\nThe frist 5 values are\n")

print(data.head(5))

print("\nThe last 5 values are\n")

print(data.tail(5))
print("\nDimensions of the Dataframe:\n")

print(data.shape)



print("\nData Type of each Column:\n")

print(data.dtypes)



print("\nStatistical Summary:\n")

print(data.describe())

data.head(5)
data.insert(1,"Time",pd.to_datetime(data.Date) + pd.to_timedelta(data.Hour, unit='h'))

data.head(5)
#Deleting Date and Hour as they are no longer required

del data['Date']

del data['Hour']



print("\nData Type of each Column:\n")

print(data.dtypes)



print("\nThe frist 5 values are\n")

print(data.head(5))
#Rearranging the coluns of the dataset



data = data [['NE-Id',"Time","Value"]]



print("\n\nThe frist 5 values are\n")

print(data.head(5))



#Grouping the Network Elements by class and finding the size of each subset

print("\nNetwork Elements and the corresponding size\n")

print(data.groupby('NE-Id').size()) 



data.head(5)
#NEX contains the columns numbers in which the particular Network Element exists 

NE1=data['NE-Id'].str.contains("MEXGDLMSS1")

NE2=data['NE-Id'].str.contains("MEXMTYMSS1")

NE3=data['NE-Id'].str.contains("MEXMTYMSS2")

NE4=data['NE-Id'].str.contains("MEXTIJMSS1")

NE5=data['NE-Id'].str.contains("MEXTLAMSS1")



NEX=[NE1,NE2,NE3,NE4,NE5]

j = 1

for i in NEX:

            print("The length of the subset containing Network element NE",j," is:",len(data[i]))

            j = j + 1

#Printing the First five elements of the subset for the Network Element:  MEXMTYMSS2

print(data[NE3].head(5))
# Loading subset NE-Id MEXGDLMSS1 into a new Dataframe df1

df1=pd.DataFrame(data[NE1])



# Loading subset NE-Id MEXMTYMSS1 into a new Dataframe df2

df2=pd.DataFrame(data[NE2])



# Loading subset NE-Id MEXMTYMSS2 into a new Dataframe df3

df3=pd.DataFrame(data[NE3])



# Loading subset NE-Id MEXTIJMSS1 into a new Dataframe df4

df4=pd.DataFrame(data[NE4])



# Loading subset NE-Id MEXTLAMSS1 into a new Dataframe df5

df5=pd.DataFrame(data[NE5])

df1.head(5)

df1.set_index('Time', inplace=True)

df1.index



def plot_vis(timeseries, timeseriesr):

    #Plotting df1 Values along the Time Series

    timeseries.plot(figsize =(15,6))

    plt.title("Time(H) vs Values")

    plt.ylabel("Values")

    plt.xlabel("Time(H)")

    plt.show()

    

    #Resample the data by taking frequency between every 6 Hours for a more Smoother Plot

    timeseriesr.plot(figsize =(15,6))

    plt.title("Time(6H) vs Values")

    plt.ylabel("Values")

    plt.xlabel("Time(6H)")

    plt.show()

    

    #Decomposing Time Series into the following components:Observed, Trend, Seasonality and Noise

    from pylab import rcParams

    import statsmodels.api as sm

    rcParams['figure.figsize'] = 16, 8

    decomposition = sm.tsa.seasonal_decompose(timeseries, model='additive')

    fig = decomposition.plot()

    plt.show()



ts=df1['Value']

tsr=df1['Value'].rolling('6H').mean()

plot_vis(ts, tsr)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing Rolling Statistics

    rolmean = timeseries.rolling(12).mean()

    rolstd = timeseries.rolling(12).std()

    

    #Plot Rolling Mean:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    plt.legend(loc='best')

    plt.title('Rolling Mean')

    plt.show(block=False)

    

    #Plot Rolling Mean & Standard Deviation:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

  

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

    

test_stationarity(ts)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from matplotlib.pyplot import plot



## Original Plot

plot_acf(df1.Value)

plot_acf(df1.Value,lags=50)

plt.show()



# Plot after 1st order Differencing

df1f = pd.DataFrame( df1.Value.diff(periods=1))

df1f = df1f[1:]

plot_acf(df1f)

plot_acf(df1f, lags=50)

plt.show()



plot_pacf(df1.Value, lags=100)

plt.show()



plot_pacf(df1f, lags=100)

plt.show()
print("The total size of the  dataset is",len(df1['Value']))

# Length of df1['Value'] is 792. We're looking to split our Test and train dataset 80:20

# 0.8 of 792 = 633

train=pd.DataFrame(df1.head(633))

print("\nTrain Dataset:\n",)

print(train.Value.head(5))



test=pd.DataFrame(df1.tail(len(df1)-len(train)))

print("\n\nTest Dataset:\n",)

print(test.Value.head(5))

predictions = []

from statsmodels.tsa.arima_model import ARIMA



# 6,0,0 ARIMA Model

model = ARIMA(train.Value, order=(5,0,2))

model_fit = model.fit(disp=0)

#Printing out ARIMA Model Results

print(model_fit.summary())

#Printing AIC number related to that ARIMA Model's p,d,q

print(model_fit.aic)
import itertools

p=d=q=range(0,5)

pdq= list(itertools.product(p,d,q))



import warnings

warnings.filterwarnings('ignore')

for para in pdq:

    try:

        model = ARIMA(train.Value, order=para)

        model_fit = model.fit(disp=0)

        print(para, model_fit.aic)

    except:

        continue

        

model = ARIMA(train.Value, order=(4,0,2))

model_fit = model.fit(disp=0)

#Printing out ARIMA Model Results

print(model_fit.summary())

#Printing AIC number related to that ARIMA Model's p,d,q

print(model_fit.aic)


# 4,0,1 pdq has the lowest aic value so suubstitute it in the ARIMA Model

model = ARIMA(df1.Value, order=(4,0,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())

print(model_fit.aic)
#Predicted Values

predictions = model_fit.forecast(steps=159)[0]

print(predictions)
#Comparing the actual df1.value and the fitted data values

model_fit.plot_predict(dynamic=False)

plt.show()
#Predicting data for the next two days - 2x24= 48, so steps=48

#Forecasted Model starts at 1 and ends at 792+48=840



model_fit.plot_predict(1,840)

X=model_fit.forecast(steps=48)



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error

from math import sqrt

def acc_met(timeseries):

    mae = mean_absolute_error(timeseries, predictions)

    print('Mean Absolute Error(MAE): %f\t' % mae)

    mse = mean_squared_error(test.Value, predictions)

    print('Mean Squared Error(MSE): %f\t' % mse)

    rmse = sqrt(mse)

    print('Root Mean Squared Error(RMSE): %f\t' % rmse)

    

acc_met(test.Value)
df2.head(5)
df2.set_index('Time', inplace=True)

df2.index

ts=df2['Value']

tsr=df2['Value'].rolling('6H').mean()

plot_vis(ts, tsr)
test_stationarity(ts)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from matplotlib.pyplot import plot



## Original Plot

plot_acf(df2.Value)

plot_acf(df2.Value,lags=50)

plt.show()



# Plot after 1stOrder Differencing

df2f = pd.DataFrame( df2.Value.diff(periods=1))

df2f = df2f[1:]

plot_acf(df2f)

plot_acf(df2f, lags=50)

plt.show()



plot_pacf(df2.Value, lags=100)

plt.show()



plot_pacf(df2f, lags=100)

plt.show()

print("The total size of the  dataset is",len(df2['Value']))

# Length of df1['Value'] is 792. We're looking to split our Test and train dataset 80:20

# 0.8 of 792 = 633

train=pd.DataFrame(df2.head(633))

print("\nTrain Dataset:\n",)

print(train.Value.head(5))



test=pd.DataFrame(df2.tail(len(df2)-len(train)))

print("\n\nTest Dataset:\n",)

print(test.Value.head(5))
predictions = []

from statsmodels.tsa.arima_model import ARIMA



model = ARIMA(train.Value, order=(2,0,2))

model_fit = model.fit(disp=0)

#Printing out ARIMA Model Results

print(model_fit.summary())

#Printing AIC number related to that ARIMA Model's p,d,q

print(model_fit.aic)
model = ARIMA(df2.Value, order=(2,0,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())

print(model_fit.aic)
#Predicted Values

predictions = model_fit.forecast(steps=159)[0]

print(predictions)
#Comparing the actual df1.value and the fitted data values

model_fit.plot_predict(dynamic=False)

plt.show()

#Predicting data for the next two days - 2x24= 48, so steps=48

#Forecasted Model starts at 1 and ends at 792+48=840



model_fit.plot_predict(1,840)

X=model_fit.forecast(steps=48)



acc_met(test.Value)
df3.head(5)
df3.set_index('Time', inplace=True)

df3.index
ts=df3['Value']

tsr=df3['Value'].rolling('6H').mean()

plot_vis(ts, tsr)
test_stationarity(ts)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from matplotlib.pyplot import plot



## Original Plot

plot_acf(df3.Value)

plot_acf(df3.Value,lags=50)

plt.show()



#Plot after 1st Order Differencing

df3f = pd.DataFrame( df3.Value.diff(periods=1))

df3f = df3f[1:]

plot_acf(df3f)

plot_acf(df3f, lags=50)

plt.show()



plot_pacf(df3.Value, lags=100)

plt.show()



plot_pacf(df3f, lags=100)

plt.show()
print("The total size of the  dataset is",len(df3['Value']))

# Length of df1['Value'] is 792. We're looking to split our Test and train dataset 80:20

# 0.8 of 792 = 633

train=pd.DataFrame(df3.head(633))

print("\nTrain Dataset:\n",)

print(train.Value.head(5))



test=pd.DataFrame(df3.tail(len(df3)-len(train)))

print("\n\nTest Dataset:\n",)

print(test.Value.head(5))

predictions = []

from statsmodels.tsa.arima_model import ARIMA



# 6,0,0 ARIMA Model

model = ARIMA(train.Value, order=(4,0,2))

model_fit = model.fit(disp=0)

#Printing out ARIMA Model Results

print(model_fit.summary())

#Printing AIC number related to that ARIMA Model's p,d,q

print(model_fit.aic)
# 4,0,1 pdq has the lowest aic value so suubstitute it in the ARIMA Model

model = ARIMA(df3.Value, order=(4,0,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())

print(model_fit.aic)
#Predicted Values

predictions = model_fit.forecast(steps=159)[0]

print(predictions)

#Comparing the actual df1.value and the fitted data values

model_fit.plot_predict(dynamic=False)

plt.show()

#Predicting data for the next two days - 2x24= 48, so steps=48

#Forecasted Model starts at 1 and ends at 792+48=840



model_fit.plot_predict(1,840)

X=model_fit.forecast(steps=48)

acc_met(test.Value)
df4.head(10)
df4.set_index('Time', inplace=True)

df4.index

ts=df4['Value']

tsr=df4['Value'].rolling('6H').mean()

plot_vis(ts, tsr)

test_stationarity(ts)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from matplotlib.pyplot import plot



## Original plot

plot_acf(df4.Value)

plot_acf(df4.Value,lags=50)

plt.show()



# Plot after 1st Order Differencing

df4f = pd.DataFrame( df4.Value.diff(periods=1))

df4f = df4f[1:]

plot_acf(df4f)

plot_acf(df4f, lags=50)

plt.show()



plot_pacf(df4.Value, lags=100)

plt.show()



plot_pacf(df4f, lags=100)

plt.show()

print("The total size of the  dataset is",len(df4['Value']))

# Length of df1['Value'] is 792. We're looking to split our Test and train dataset 80:20

# 0.8 of 792 = 633

train=pd.DataFrame(df4.head(633))

print("\nTrain Dataset:\n",)

print(train.Value.head(5))



test=pd.DataFrame(df4.tail(len(df4)-len(train)))

print("\n\nTest Dataset:\n",)

print(test.Value.head(5))
predictions = []



model = ARIMA(train.Value, order=(3,0,2))

model_fit = model.fit(disp=0)

#Printing out ARIMA Model Results

print(model_fit.summary())

#Printing AIC number related to that ARIMA Model's p,d,q

print(model_fit.aic)
model = ARIMA(df4.Value, order=(3,0,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())

print(model_fit.aic)
#Predicted Values

predictions = model_fit.forecast(steps=159)[0]

print(predictions)

#Comparing the actual df1.value and the fitted data values

model_fit.plot_predict(dynamic=False)

plt.show()
#Predicting data for the next two days - 2x24= 48, so steps=48

#Forecasted Model starts at 1 and ends at 792+48=840



model_fit.plot_predict(1,840)

X=model_fit.forecast(steps=48)
acc_met(test.Value)
df5.head(5)
df5.set_index('Time', inplace=True)

df5.index


ts=df5['Value']

tsr=df5['Value'].rolling('6H').mean()

plot_vis(ts, tsr)

test_stationarity(ts)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from matplotlib.pyplot import plot



## Original Series

plot_acf(df5.Value)

plot_acf(df5.Value,lags=50)

plt.show()



# 1st Differencing

df5f = pd.DataFrame( df5.Value.diff(periods=1))

df5f = df5f[1:]

plot_acf(df5f)

plot_acf(df5f, lags=50)

plt.show()



plot_pacf(df5.Value, lags=100)

plt.show()



plot_pacf(df5f, lags=100)

plt.show()
print("The total size of the  dataset is",len(df5['Value']))

# Length of df1['Value'] is 792. We're looking to split our Test and train dataset 80:20

# 0.8 of 792 = 633

train=pd.DataFrame(df5.head(633))

print("\nTrain Dataset:\n",)

print(train.Value.head(5))



test=pd.DataFrame(df5.tail(len(df5)-len(train)))

print("\n\nTest Dataset:\n",)

print(test.Value.head(5))

predictions = []



model = ARIMA(train.Value, order=(1,0,1))

model_fit = model.fit(disp=0)

#Printing out ARIMA Model Results

print(model_fit.summary())

#Printing AIC number related to that ARIMA Model's p,d,q

print(model_fit.aic)
# 4,0,1 pdq has the lowest aic value so suubstitute it in the ARIMA Model

model = ARIMA(df5.Value, order=(1,0,1))

model_fit = model.fit(disp=0)

print(model_fit.summary())

print(model_fit.aic)

#Predicted Values

predictions = model_fit.forecast(steps=159)[0]

print(predictions)
#Comparing the actual df1.value and the fitted data values

model_fit.plot_predict(dynamic=False)

plt.show()
#Predicting data for the next two days - 2x24= 48, so steps=48

#Forecasted Model starts at 1 and ends at 792+48=840



model_fit.plot_predict(1,840)

X=model_fit.forecast(steps=48)
acc_met(test.Value)