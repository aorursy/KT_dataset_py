import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/productdemandforecasting/Historical Product Demand.csv",parse_dates=['Date'])

df.head()
# Check Shape & data Types



print(df.shape)

print(df.dtypes)
# to check columns if they have any missing values, it will return number of Nan in the given columns



df.isnull().sum()



# Mssing values are in dates

# calculating % of data missing

print("% of Data missing =", df.isnull().sum().sum()/len(df)*100)
# Data missing is approxiately 1% of actual data, so we can remove it using Dropna



df.dropna(axis=0, inplace=True) #remove all rows with Nan



#setting date as index columns

df.reset_index(drop = True)

df.isnull().sum()



# Now there is no missing data
df.sort_values('Date')[10:20]

# Some value in Order_Demand column has (), therefore need tro remove before converting them into integer
df['Order_Demand'] = df['Order_Demand'].str.replace('(',"")

df['Order_Demand'] = df['Order_Demand'].str.replace(')',"")

df.sort_values('Date')[10:20]

# Now () are removed from the column
#Making the data type into integer

df['Order_Demand'] = df['Order_Demand'].astype('int64')
sns.set(rc={'figure.figsize':(16,5)})

sns.distplot(df['Order_Demand'], bins = 100);

# it can seen that most of our demand lies between 0 to 500000, which is highly skewed
df.groupby('Warehouse')['Order_Demand'].sum().sort_values(ascending=False)

# It can be seen Warehouse J has maximum demand
df1 = pd.DataFrame(df.groupby('Product_Category')['Order_Demand'].sum().sort_values(ascending=False))

df1["% Contribution"] = df1['Order_Demand']/df1['Order_Demand'].sum()*100

df1

# It can be seen starting top 4 products category contribute more than 90% of the demand
df2 = pd.pivot_table(df,index=["Date"],values=["Order_Demand"],columns=["Product_Category"],aggfunc=np.sum)

df2.columns = df2.columns.droplevel(0)

df2["Category_019"].dropna()

# Creating Pivot table with date as index, Product category as columns & and values as sum
y = df2.resample('M').sum() # Resampling the data on monthly basis 

y.index.freq = "M" # Setting datetime frequency to Month

y.head(20)

# In Year 2011 so much data is missing, so we will exlude it 
df_019 = pd.DataFrame(y["Category_019"].iloc[12:-1]) # Including data from 2012 to 2016 end except last value

df_019.head()
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from statsmodels.tsa.holtwinters import ExponentialSmoothing



span = 4

alpha = 2/(span+1)

df_019['EWMA4'] = df_019["Category_019"].ewm(alpha=alpha,adjust=False).mean() # Simple Weighted Moving Average

# Simple Exponentional Smoothing

df_019['SES4']=SimpleExpSmoothing(df_019["Category_019"]).fit(smoothing_level=alpha,optimized=False).fittedvalues.shift(-1)



#Double Exponentional  Smothening

df_019['DESadd4'] = ExponentialSmoothing(df_019["Category_019"], trend='add').fit().fittedvalues.shift(-1)
df_019[["Category_019",'SES4','DESadd4']].plot(figsize = (20,6)) # Plot for Weighted Moving average & Double Exponentional, 

#It can be seen data has some seasonailty, therefore will use ARIMA, ARMA
# Will Ignore harmless warnings

import warnings

warnings.filterwarnings("ignore")



! pip install pmdarima 

from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders

from pmdarima import auto_arima # for determining ARIMA orders

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
# to Check series is stationary or not

from statsmodels.tsa.stattools import adfuller



def adf_test(series,title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print(f'Augmented Dickey-Fuller Test: {title}')

    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out[f'critical value ({key})']=val

        

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    

    if result[1] <= 0.05:

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")
adf_test(df_019["Category_019"])

# Series is non stationary
fit = auto_arima(df_019["Category_019"], start_p=1, start_q=1,

                          max_p=5, max_q=5, m=12,

                          start_P=0, seasonal=True,

                          d=None, D=1, trace=True,

                          error_action='ignore',   # we don't want to know if an order does not work

                          suppress_warnings=True,  # we don't want convergence warnings

                          stepwise=True)           # set to stepwise



fit.summary()

# After running Auto Arima, Best order is SARIMAX(3, 1, 3)x(0, 1, [1, 2], 12) for which AIC is minimum
len(df_019["Category_019"])

# Train & Test Data

train = df_019["Category_019"].iloc[:48]

test = df_019["Category_019"].iloc[48:]
model = SARIMAX(train,order=(3,1,3),seasonal_order=(0,1,1,12))

results = model.fit()

results.summary()
# Obtain predicted values

start=len(train)

end=len(train)+len(test)-1

predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA (3,1,3),(0,1,1,12) Predictions')
ax = test.plot(legend=True,figsize=(12,6))

predictions.plot(legend=True)

ax.autoscale(axis='x',tight=True)

#plotting Test data & predicted demand
from sklearn.metrics import mean_squared_error



error = np.sqrt(mean_squared_error(test, predictions))

print(f'SARIMA(0,1,3)(1,0,1,12) RMSE Error: {error:11.10}')

print('Std of Test data:                  ', df_019["Category_019"].std())

# Comparison of RMSE & Std of data, as Std if very high compared to RMSE
# Retrain the model on the full data, and forecasting for next 4 months

model = SARIMAX(df_019["Category_019"],order=(3,1,3),seasonal_order=(0,1,1,12))

results = model.fit()

fcast = results.predict(len(df_019["Category_019"]),len(df_019["Category_019"])+4,typ='levels').rename('SARIMA(3,1,3)(0,1,1,12) Forecast')
ax = df_019["Category_019"].plot(legend=True,figsize=(12,6))

fcast.plot(legend=True)

ax.autoscale(axis='x',tight=True)

#plotting actual data & 4 month forecasted demand
# Same steps need to be done for each category, considering the missing data, stationarity. Finding the best order to be fit into the model and using the train test split to validate the model, Finally forecasting for the required Months.