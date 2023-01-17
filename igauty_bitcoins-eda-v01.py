# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import datetime, pytz

import time

import warnings

warnings.filterwarnings('ignore')





from scipy.stats import norm

from scipy import stats

from sklearn import preprocessing



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

sns.set_style('whitegrid')

%matplotlib inline
#define a conversion function for the native timestamps in the csv file

def dateparse (time_in_secs):    

    return datetime.datetime.fromtimestamp(float(time_in_secs))
data = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv', parse_dates=[0], date_parser=dateparse)
data.head()
def missing_check(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 

    #print("Missing check:",missing_data )

    return missing_data
missing_check(data)
def missing_value_exp(df):

    missing_values = df.isnull().sum(axis=0).reset_index()

    missing_values.columns = ['column_name', 'Total']

    missing_values = missing_values.loc[missing_values['Total']>0.5]

    missing_values = missing_values.sort_values(by='Total')



    ind = np.arange(missing_values.shape[0])

    width = 0.1

    fig, ax = plt.subplots(figsize=(16,4))

    rects = ax.barh(ind, missing_values.Total.values, color='r')

    ax.set_yticks(ind)

    ax.set_yticklabels(missing_values.column_name.values, rotation='horizontal')

    ax.set_xlabel("Count of Missing Observations")

    ax.set_title("Missing Categorical Observations in Dataset")

    plt.show()
missing_value_exp(data)
def treat_missing(df, choice):

    if(choice==1):

        df = df.dropna()         

    elif(choice==2):

        df = df.fillna(df.mean())        

    elif(choice==3):

        df = df.fillna(df.mode())           

    elif(choice==4):

        df = df.fillna(df.median())            

    elif(choice==5):

        df = df.fillna(method='ffill')               

    elif(choice==6):

        df = df.fillna(method='bfill')

    else:

        df = df.fillna(0)

    return df
data['Volume_(BTC)'] =treat_missing(data['Volume_(BTC)'], 7)

data['Volume_(Currency)'] =treat_missing(data['Volume_(Currency)'], 7)

data['Weighted_Price'] =treat_missing(data['Weighted_Price'], 7)

data['Open'] =treat_missing(data['Open'], 5)

data['High'] =treat_missing(data['High'], 5)

data['Low'] =treat_missing(data['Low'], 5)

data['Close'] =treat_missing(data['Close'], 5)
missing_check(data)
# Line Plot usage

data.Open.plot(kind='line', color='g', label='Open', figsize=(20, 10))
data.Close.plot(color='r', label='Close', figsize=(20, 10))
data.High.plot(color='g', label='High', figsize=(20, 10))
data.Low.plot(color='r', label='Low', figsize=(20, 10))
#df = pd.DataFrame()

data['date_time'] = pd.to_datetime(data['Timestamp'])

data['year'] = data['date_time'].dt.year

data['Month'] = data['date_time'].dt.month

data['Week'] = data['date_time'].dt.week

data['day'] = data['date_time'].dt.day



data['Time Decimal'] = data['Timestamp'].dt.hour + data['Timestamp'].dt.minute/60



day_Week={0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}

data['Day_of_Week'] = data['Timestamp'].dt.dayofweek.map(day_Week)
data.columns
data[['date_time', 'year', 'Month','Week', 'day']]
# create valid date range

start = datetime.datetime(2015, 1, 1, 0, 0, 0, 0, pytz.UTC)

end = datetime.datetime(2017, 10, 17, 20, 0, 0, 0, pytz.UTC)



start
end
# find rows between start and end time and find the first row (00:00 monday morning)

weekly_rows = data[(data['Timestamp'] >= '2015-01-01 00:00:00') & (data['Timestamp'] <= '2017-10-17 20:00:00')].groupby([pd.Grouper(key='Timestamp', freq='W-MON')]).first().reset_index()

weekly_rows.tail()
weekly_rows['Investment'] = (weekly_rows.index+1)*10

weekly_rows['Account_v'] = ((10.0 / weekly_rows['Close'].astype(float)).cumsum()) * weekly_rows['Close'].astype(float)

weekly_rows.Investment.plot(color='r', label='Investment', figsize=(20, 10))

weekly_rows.Account_v.plot(color='g', label='Account_value', figsize=(20, 10))

weekly_rows.Close.plot(color='b', label='Bitcoin Price', figsize=(20, 10))

plt.legend(loc='upper left')
# find indices with min value of that week

idx = data.groupby([pd.Grouper(key='Timestamp', freq='W-MON')])['Close'].transform(min) == data['Close']



# remove duplicate day rows

weekly_lows = data[idx].groupby([pd.Grouper(key='Timestamp', freq='D')]).first().reset_index()
sns.countplot(x='Day_of_Week',data=weekly_lows, order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
# find indices with min value of that day

daily_lows = data[data.groupby([pd.Grouper(key='Timestamp', freq='D')])['Close'].transform(min) == data['Close']]
sns.boxplot(x="Day_of_Week", y="Time Decimal", data=daily_lows, palette='rainbow')
