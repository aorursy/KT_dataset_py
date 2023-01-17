# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pmdarima

#Make sure you have enabled internet while running this inside Kaggle Kernel
##Importing the most frequent libraries used

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np  # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
#setting figure size

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20,10
#for normalizing data

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
#We will use one dataset from the above list to perform our analysis (Maruti) 

data = pd.read_csv("/kaggle/input/nifty50-stock-market-data/MARUTI.csv")

data.head()
#We will creat a empty dataframe to store all our prediction results

plot_df = pd.DataFrame()
data.shape

#4098 rows and 15 columns
# This will show us the what are the data columns and its data type available for analysis

data.info()
#Not much, but still we seem to have null values as shown below

data.isnull().sum()
#We will drop this column as we are not going to use this and it has considerable amount of null values 

#We will also drop the null values

data.drop(['Trades'], axis=1,inplace = True)

data.dropna(inplace=True)
#We will set the Index to the date column availabe as it will be best suited in this secnario

data.set_index("Date", drop=False, inplace=True)

data.head()
data.shape
data.VWAP.plot()

#Shows as increasing trend over the time

plt.yticks(np.arange(100, 10000, 1000))
data[['Open','Close','VWAP','High','Low']].plot()

plt.yticks(np.arange(100, 10000, 1000))
#Lets visualize the correlation among the data

corr = data.corr()

sns.heatmap(corr)
data.Date = pd.to_datetime(data.Date, format="%Y-%m-%d")

data["month"] = data.Date.dt.month

data["week"] = data.Date.dt.week

data["day"] = data.Date.dt.day

data["day_of_week"] = data.Date.dt.dayofweek

data.head()
#Split is not random, as we are dependent on time for the analysis

data_train = data[data.Date < "2019"]

data_valid = data[data.Date >= "2019"]
from pmdarima import auto_arima



model_ARIMA = auto_arima(data_train.VWAP,trace=True, start_p=1, start_q=1,max_p=3, max_q=3, 

                   m=12,start_P=0, seasonal=True,d=1, D=1,error_action='ignore',suppress_warnings=True)

model_ARIMA.fit(data_train.VWAP)



forecast_ARIMA = model_ARIMA.predict(n_periods=len(data_valid))

plot_df['VWAP'] = data_valid['VWAP']

plot_df['Forecast_ARIMAX'] = forecast_ARIMA

plot_df[["VWAP", "Forecast_ARIMAX"]].plot()

plt.yticks(np.arange(100, 10000, 1000))
#importing libraries

from sklearn import neighbors

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
x_train = data_train.drop(['VWAP','Date','Symbol','Series'], axis=1)

y_train = data_train['VWAP']

x_valid = data_valid.drop(['VWAP','Date','Symbol','Series'], axis=1)

y_valid = data_valid['VWAP']
#scaling data

x_train_scaled = scaler.fit_transform(x_train)

x_train = pd.DataFrame(x_train_scaled)

x_valid_scaled = scaler.fit_transform(x_valid)

x_valid = pd.DataFrame(x_valid_scaled)



#using gridsearch to find the best parameter

params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

knn = neighbors.KNeighborsRegressor()

model_knn = GridSearchCV(knn, params, cv=5)



#fit the model and make predictions

model_knn.fit(x_train,y_train)

forecast_knn = model_knn.predict(x_valid)
plot_df["Forecast_KNN"] = forecast_knn

plot_df[["VWAP", "Forecast_KNN"]].plot()

plt.yticks(np.arange(100, 10000, 1000))
#implement linear regression

from sklearn.linear_model import LinearRegression

model_lin = LinearRegression()

model_lin.fit(x_train,y_train)
preds_lin = model_lin.predict(x_valid)
plot_df["Forecast_lin"] = preds_lin

plot_df[["VWAP", "Forecast_lin"]].plot()
from fbprophet import Prophet
#We will use one dataset from the above list to perform our analysis (Maruti) 

data_prophet = pd.read_csv("/kaggle/input/nifty50-stock-market-data/MARUTI.csv")

data_prophet.head()
data_train_p = data_prophet[data_prophet.Date < "2019"]

data_valid_p = data_prophet[data_prophet.Date >= "2019"]
#fit the model

model_fbp = Prophet()

model_fbp.fit(data_train_p[["Date", "VWAP"]].rename(columns={"Date": "ds", "VWAP": "y"}))
forecast_prophet = model_fbp.predict(data_valid_p[["Date", "VWAP"]].rename(columns={"Date": "ds"}))

preds_prophet = forecast_prophet.yhat.values
plot_df["Forecast_prophet"] = preds_prophet

plot_df[["VWAP", "Forecast_prophet"]].plot()

plt.yticks(np.arange(100, 10000, 1000))
new_data = pd.read_csv("/kaggle/input/nifty50-stock-market-data/MARUTI.csv")

new_data.head()
new_data.reset_index(drop=True, inplace=True)

lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]

window1 = 3

window2 = 7

window3 = 30



new_data_rolled_3d = new_data[lag_features].rolling(window=window1, min_periods=0)

new_data_rolled_7d = new_data[lag_features].rolling(window=window2, min_periods=0)

new_data_rolled_30d = new_data[lag_features].rolling(window=window3, min_periods=0)



new_data_mean_3d = new_data_rolled_3d.mean().shift(1).reset_index().astype(np.float32)

new_data_mean_7d = new_data_rolled_7d.mean().shift(1).reset_index().astype(np.float32)

new_data_mean_30d = new_data_rolled_30d.mean().shift(1).reset_index().astype(np.float32)



new_data_std_3d = new_data_rolled_3d.std().shift(1).reset_index().astype(np.float32)

new_data_std_7d = new_data_rolled_7d.std().shift(1).reset_index().astype(np.float32)

new_data_std_30d = new_data_rolled_30d.std().shift(1).reset_index().astype(np.float32)



for feature in lag_features:

    new_data[f"{feature}_mean_lag{window1}"] = new_data_mean_3d[feature]

    new_data[f"{feature}_mean_lag{window2}"] = new_data_mean_7d[feature]

    new_data[f"{feature}_mean_lag{window3}"] = new_data_mean_30d[feature]

    

    new_data[f"{feature}_std_lag{window1}"] = new_data_std_3d[feature]

    new_data[f"{feature}_std_lag{window2}"] = new_data_std_7d[feature]

    new_data[f"{feature}_std_lag{window3}"] = new_data_std_30d[feature]



new_data.fillna(new_data.mean(), inplace=True)



new_data.set_index("Date", drop=False, inplace=True)

new_data.head()
new_data.Date = pd.to_datetime(new_data.Date, format="%Y-%m-%d")

new_data["month"] = new_data.Date.dt.month

new_data["week"] = new_data.Date.dt.week

new_data["day"] = new_data.Date.dt.day

new_data["day_of_week"] = new_data.Date.dt.dayofweek

new_data.head()
new_data_train = new_data[new_data.Date < "2019"]

new_data_valid = new_data[new_data.Date >= "2019"]



exogenous_features = ["High_mean_lag3", "High_std_lag3", "Low_mean_lag3", "Low_std_lag3",

                      "Volume_mean_lag3", "Volume_std_lag3", "Turnover_mean_lag3",

                      "Turnover_std_lag3", "Trades_mean_lag3", "Trades_std_lag3",

                      "High_mean_lag7", "High_std_lag7", "Low_mean_lag7", "Low_std_lag7",

                      "Volume_mean_lag7", "Volume_std_lag7", "Turnover_mean_lag7",

                      "Turnover_std_lag7", "Trades_mean_lag7", "Trades_std_lag7",

                      "High_mean_lag30", "High_std_lag30", "Low_mean_lag30", "Low_std_lag30",

                      "Volume_mean_lag30", "Volume_std_lag30", "Turnover_mean_lag30",

                      "Turnover_std_lag30", "Trades_mean_lag30", "Trades_std_lag30",

                      "month", "week", "day", "day_of_week"]
model_fbp_features = Prophet()

for feature in exogenous_features:

    model_fbp_features.add_regressor(feature)



model_fbp_features.fit(new_data_train[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds", "VWAP": "y"}))



forecast_prophet_features = model_fbp_features.predict(new_data_valid[["Date", "VWAP"] + exogenous_features].rename(columns={"Date": "ds"}))

plot_df["Forecast_Prophet_features"] = forecast_prophet_features.yhat.values
plot_df[["VWAP","Forecast_Prophet_features"]].plot()

plt.yticks(np.arange(100, 10000, 1000))
#Overall Comparision of various timeseries models

plot_df[["VWAP","Forecast_ARIMAX" , "Forecast_KNN" , "Forecast_lin" , "Forecast_prophet" , "Forecast_Prophet_features"]].plot()

plt.yticks(np.arange(100, 10000, 1000))