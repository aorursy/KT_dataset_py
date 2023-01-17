%matplotlib inline

import matplotlib.pyplot as plt

import plotly.express as px

import pandas as pd

import json

import requests





import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn.model_selection import train_test_split

from sklearn import metrics

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (12, 8)

plt.rcParams['font.size'] = 14



import datetime 

from pandas.plotting import register_matplotlib_converters



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn.model_selection import train_test_split

from sklearn import metrics

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

%matplotlib inline

pd.options.display.float_format = '{:.2f}'.format

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats, integrate

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:.2f}'.format

plt.rcParams['figure.figsize'] = (12, 8)

plt.rcParams['font.size'] = 14



import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

from matplotlib.pyplot import figure



import datetime as dt

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from mpl_toolkits.basemap import Basemap

from sklearn.model_selection import TimeSeriesSplit

plt.style.use('ggplot')

%config InlineBackend.figure_format = 'retina'

import warnings

warnings.filterwarnings('ignore')



# retrieve json file for Live Traffic

#url = "https://api.midway.tomtom.com/ranking/liveHourly/POL_wroclaw"

live = "https://api.midway.tomtom.com/ranking/live/POL_wroclaw"

wro_live = requests.get(live)

wro_json = wro_live.json()





# retrieve json file for Weather

weather = "https://api.weather.midway.tomtom.com/weather/live/POL_wroclaw"



weather_req = requests.get(weather)

weather_json = weather_req.json()



pd.set_option("display.max_rows", False)

#wro_json
#wro_json["data"][0]["TrafficIndexLive"]
# create empty lists of append data for Live Traffic

jams_delay = []

traffic_index_live = []

update_time = []

jams_length = []

jams_count = []

traffic_index_week = []

time = []





count = len(wro_json["data"])-1



# append each item in the json file to the empty lists

i=0

while i<=count:

    jams_delay.append(wro_json["data"][i]["JamsDelay"],)

    traffic_index_live.append(wro_json["data"][i]["TrafficIndexLive"],)

    update_time.append(wro_json["data"][i]["UpdateTime"])

    jams_length.append(wro_json["data"][i]["JamsLength"],)

    jams_count.append(wro_json["data"][i]["JamsCount"],)

    #traffic_index_week.append(wro_json["data"][i]["TrafficIndexWeekAgo"],)

    #traffic_index_week.append(wro_json["data"][i]["UpdateTimeWeekAgo"],)

    time.append(wro_json["data"][i]["UpdateTime"])

    i+=1

    

# create dataframe with the traffic data 

#df = pd.DataFrame({"Live Traffic":live_traffic,"Jams Delay":jams_delay,"Jams Length":jams_length,"Jams Count":jams_count,"Traffic Index Week Ago":traffic_index_week_ago,"Update Time Week Ago":update_time_week_ago}, index=time)

df = pd.DataFrame({"Delay":jams_delay,"Traffic":traffic_index_live  ,"Date":update_time,"Length":jams_length,"Count":jams_count}, index=time)

#df.update_time = pd.to_datetime(df.update_time, unit="ms")

df.index = pd.to_datetime(df.index, unit="ms")

df['Date'] = pd.to_datetime(df['Date'], unit="ms")

df.index.name = "Time"

#df.head()
from sklearn.model_selection import train_test_split



train_df, test_df = train_test_split(df, test_size=0.2)
#train_df
#test_df
corr =  df.corr()

plt.subplots(figsize=(20,9))

sns.heatmap(corr)
from pandas.plotting import register_matplotlib_converters



df.index = df['Date'] # indexing the Datetime to get the time period on the x-axis. 

#df=df.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis. 

ts = df['Count'] 

plt.figure(figsize=(16,8)) 

plt.plot(ts, label='Traffic Jams in Wroclaw') 

plt.title('Jams Count') 

plt.xlabel("Time(year-month)") 

plt.ylabel("Number of Jams") 

plt.legend(loc='best')
df.index = df['Date'] # indexing the Datetime to get the time period on the x-axis. 

#df=df.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis. 

ts = df['Count'] 



plt.figure(figsize=(25,8))

sns.barplot(x=df.index.date, y=df["Count"])

plt.title("Traffic Jams in Wroclaw")

plt.xticks(rotation=90)

train = train_df

test = test_df
# Making copy of dataset



train_original=train.copy() 

test_original=test.copy()
#print(train_original.head())

#print (test_original.head())
#train.info(), test.info()
#df = df.rename(columns={'Jams Delay': 'Jams_Delay', 'Live Traffic': 'Live_Traffic','Update Time': 'Update_Time', 'Jams Length': 'Jams_Length', 'Jams Count': 'Jams_Count'})

#df.head()
import datetime 



train['Date'] = pd.to_datetime(train.Date,format='%d-%m-%Y %H:%M',infer_datetime_format=True) 

test['Date'] = pd.to_datetime(test.Date,format='%d-%m-%Y %H:%M', infer_datetime_format=True) 

test_original['Date'] = pd.to_datetime(test_original.Date,format='%d-%m-%Y %H:%M', infer_datetime_format=True) 

train_original['Date'] = pd.to_datetime(train_original.Date,format='%d %m %Y %H:%M',  infer_datetime_format=True)
#train_original.head()
for i in (train, test, test_original, train_original):

    i['year']=i.Date.dt.year 

    i['month']=i.Date.dt.month 

    i['day']=i.Date.dt.day

    i['Hour']=i.Date.dt.hour 
train['dow']=train['Date'].dt.dayofweek 

temp = train['Date']
def applyer(row):

    if row.dayofweek == 5 or row.dayofweek == 6:

        return 1

    else:

        return 0 

temp2 = train['Date'].apply(applyer) 

train['weekend']=temp2
from pandas.plotting import register_matplotlib_converters



train.index = train['Date'] # indexing the Datetime to get the time period on the x-axis. 

#df=train.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis. 

ts = df['Count'] 

plt.figure(figsize=(16,8)) 

plt.plot(ts, label='Jams Count') 

plt.title('Time Series') 

plt.xlabel("Time(year-month)") 

plt.ylabel("Passenger count") 

plt.legend(loc='best')
train.groupby('year')['Count'].mean().plot.bar(fontsize=14,figsize=(10,7),title='Yearly Jams Count')
train.groupby('month')['Count'].mean().plot.bar(fontsize=14,figsize=(10,7), title='Monthly Jams Count')
#=train.groupby(['year', 'month'])['Count'].mean() 

#temp.plot(figsize=(15,5), title= 'Jams Count(Year& Month)', fontsize=14)
train.groupby('day')['Count'].mean().plot.bar(fontsize=14,figsize=(10,7),title='Daily Jams in Wroclaw')
train.groupby('Hour')['Count'].mean().plot.bar(color='m', figsize=(10,7),fontsize=14,title='Hourly Traffic Jams in Wroclaw')
train.groupby('weekend')['Count'].mean().plot.bar(fontsize=14,figsize=(10,7),title='Weekend Traffic Jams Analysis')
train.groupby('dow')['Count'].mean().plot.bar(fontsize=14,figsize=(10,7), title='Traffic Jams based on the day of week')
train["WeekOfYear"]=train.index.weekofyear



week_num=[]

count_index = []

length_index = []

delay_index = []



w=1

for i in list(train["WeekOfYear"].unique()):

    count_index.append(train[train["WeekOfYear"]==i]["Count"].iloc[-1])

    length_index.append(train[train["WeekOfYear"]==i]["Length"].iloc[-1])

    

    week_num.append(w)

    w=w+1



plt.figure(figsize=(8,5))

plt.plot(week_num,count_index,linewidth=3)

plt.plot(week_num,length_index,linewidth=3)



plt.ylabel("Number of Traffic Jams")

plt.xlabel("Jams Length")

plt.title("Correlation of Traffic Jams and Lengths along the week.")

plt.xlabel
week_num=[]

count_index = []

length_index = []

delay_index = []



w=1

for i in list(train["WeekOfYear"].unique()):

    count_index.append(train[train["WeekOfYear"]==i]["Count"].iloc[-1])

    length_index.append(train[train["WeekOfYear"]==i]["Length"].iloc[-1])

    

    week_num.append(w)

    w=w+1









plt.figure(figsize=(25,8))

plt.plot(df["Count"],marker="o",label="Number of Jams")

plt.plot(df["Length"],marker="*",label="Length")

plt.plot(df["Delay"],marker="^",label="Delays")

plt.ylabel("Number of Traffic Jams")

plt.xlabel("Dates")

plt.xticks(rotation=90)

plt.title("Progress of traffic jams over time")

plt.legend()

plt.savefig('004br.png')
#train=train.drop('ID',1)
train.Timestamp = pd.to_datetime(train.Date,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp 

# Hourly time series 

hourly = train.resample('H').mean() 

# Converting to daily mean 

daily = train.resample('D').mean() 

# Converting to weekly mean 

weekly = train.resample('W').mean() 

# Converting to monthly mean 

monthly = train.resample('M').mean()
fig, axs = plt.subplots(4,1) 

hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0]) 

daily.Count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1])

weekly.Count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2]) 

monthly.Count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3]) 
train.shape, test.shape
test.Timestamp = pd.to_datetime(test.Date,format='%d-%m-%Y %H:%M') 

test.index = test.Timestamp  

# Converting to daily mean 

test = test.resample('D').mean() 
train.Timestamp = pd.to_datetime(train.Date,format='%d-%m-%Y %H:%M') 

train.index = train.Timestamp 

# Converting to daily mean 

train = train.resample('D').mean()
#Train=train.loc['2012-08-25':'2014-06-24'] 

#valid=train.loc['2014-06-25':'2014-09-25']
Train=train.loc['2020-05-01':'2020-12-31'] 

valid=train.loc['2020-05-09':'2020-05-15']
Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train') 

valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid') 

plt.xlabel("Datetime") 

plt.ylabel("Passenger count") 

plt.legend(loc='best') 

plt.show()
dd= np.asarray(Train.Count) 

y_hat = valid.copy() 

y_hat['naive'] = dd[len(dd)-1] 

plt.figure(figsize=(12,8)) 

plt.plot(Train.index, Train['Count'], label='Train') 

plt.plot(valid.index,valid['Count'], label='Valid') 

plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 

plt.legend(loc='best') 

plt.title("Naive Forecast") 

plt.show()
# calculating RMSE to check the accuracy of our model on validation data set.



from sklearn.metrics import mean_squared_error 

from math import sqrt 

rms = sqrt(mean_squared_error(valid.Count, y_hat.naive)) 

print(rms)
# Considering rolling mean for last 10, 20, 50 days and visualize the results.



y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(10).mean().iloc[-1] # average of last 10 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations') 

plt.legend(loc='best') 

plt.show() 

y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(20).mean().iloc[-1] # average of last 20 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations') 

plt.legend(loc='best') 

plt.show() 

y_hat_avg = valid.copy() 

y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(50).mean().iloc[-1] # average of last 50 observations. 

plt.figure(figsize=(15,5)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations') 

plt.legend(loc='best') 

plt.show()
# RMSE value for Moving Average 



#rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.moving_avg_forecast)) 

#print(rms)
#Here the predictions are made by assigning larger weight to the recent values and lesser weight to the old values.



from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 



y_hat_avg = valid.copy() 

fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.6,optimized=False) 

y_hat_avg['SES'] = fit2.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['SES'], label='SES') 

plt.legend(loc='best') 

plt.show()
rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SES)) 

print(rms)
import statsmodels.api as sm 

sm.tsa.seasonal_decompose(Train.Count).plot() 

result = sm.tsa.stattools.adfuller(train.Count) 

plt.show()
y_hat_avg = valid.copy() 

fit1 = Holt(np.asarray(Train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1) 

y_hat_avg['Holt_linear'] = fit1.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot(Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear') 

plt.legend(loc='best') 

plt.show()
# Calculating the RMSE of the model



rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_linear)) 

print(rms)
y_hat_avg.Holt_linear.head()
valid.Count.shape, y_hat_avg.Holt_linear.shape
y_hat_avg = valid.copy() 

fit1 = ExponentialSmoothing(np.asarray(Train['Count']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit() 

y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid)) 

plt.figure(figsize=(16,8)) 

plt.plot( Train['Count'], label='Train') 

plt.plot(valid['Count'], label='Valid') 

plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter') 

plt.legend(loc='best') 

plt.show()

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_Winter)) 

print(rms)
predict=fit1.forecast(len(test))
train_log_moving_avg_diff = Train_log - moving_avg
train_log_moving_avg_diff.dropna(inplace = True) 

#test_stationarity(train_log_moving_avg_diff)
train_log_diff = Train_log - Train_log.shift(1) 

test_stationarity(train_log_diff.dropna())
from statsmodels.tsa.seasonal import seasonal_decompose 

decomposition = seasonal_decompose(pd.DataFrame(Train_log).Count.values, freq = 7) 



trend = decomposition.trend 

seasonal = decomposition.seasonal 

residual = decomposition.resid 



plt.subplot(411) 

plt.plot(Train_log, label='Original') 

plt.legend(loc='best') 

plt.subplot(412) 

plt.plot(trend, label='Trend') 

plt.legend(loc='best') 

plt.subplot(413) 

plt.plot(seasonal,label='Seasonality') 

plt.legend(loc='best') 

plt.subplot(414) 

plt.plot(residual, label='Residuals') 

plt.legend(loc='best') 

plt.tight_layout() 

plt.show()
train_log_decompose = pd.DataFrame(residual) 

train_log_decompose['date'] = Train_log.index 

train_log_decompose.set_index('date', inplace = True) 

train_log_decompose.dropna(inplace=True) 

test_stationarity(train_log_decompose[0])