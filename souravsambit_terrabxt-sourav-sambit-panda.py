import pandas as pd

import numpy as np

from scipy import stats

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from math import sqrt

from datetime import datetime

from pandas import Series

import seaborn as sns

%matplotlib inline


df_train = pd.read_csv(r"../input/into-the-future/train.csv")

print(df_train.head(3))

df_test = pd.read_csv(r'../input/into-the-future/test.csv')

print(df_test.head(3))
df_train.shape
df_train["time"]= pd.to_datetime(df_train["time"])
df_train["year"]=df_train["time"].dt.year
df_train['month'] =df_train["time"].dt.month

df_train['day'] = df_train["time"].dt.day

df_train["hour"]=df_train["time"].dt.hour

df_train["second"]=df_train["time"].dt.second
df_train["minute"]=df_train["time"].dt.minute
df_train.head(2)
df_train.tail(2)
print('No. of years data : {}'.format(df_train["time"].dt.year.nunique()))

print('No. of months data : {}'.format(df_train["time"].dt.month.nunique()))

print('all days are : {}'.format(df_train["time"].dt.day.unique()))
print(sorted(df_train['hour'].unique()))
#Variable : day_name

df_train['day_name'] = df_train["time"].apply(lambda x : x.day_name())
df_train.head(2)
### here we can see counts of all the columns 

df_train.hist(figsize=(25,12))

plt.title('All Data Show Histogram System')

plt.show()

# Distribution of feature_1	 in month



sns.distplot(df_train['feature_1'], bins=10, kde=True)
# Distribution of feature_2 in month



sns.distplot(df_train['feature_2'], bins=10, kde=True)
#co-relation between columns

corrmat = df_train.corr() 

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
df_train.drop('id',axis=1,inplace=True)
df_train
#distribution of Feature_1 as per hour

import plotly.express as px

fig = px.scatter(df_train,df_train["hour"],df_train["feature_1"],color=df_train["second"])

fig.show()
#distribution of Feature_1 as per second

import plotly.express as px

fig = px.scatter(df_train,df_train["second"],df_train["feature_1"],color=df_train["hour"])

fig.show()
# average Feature on each second of month 

f=df_train.pivot_table('feature_1',index=['second'],columns=['month'],aggfunc=np.mean)

print(f)

f.plot()


df = pd.read_csv(r"../input/into-the-future/train.csv")

print(df.head(3))
df.dtypes
#we need to convert time to datetime 

df['time'] = pd.to_datetime(df['time'])

df.drop('id',axis=1,inplace=True)

df.set_index('time',inplace=True)
df.tail()
print(df.shape)


plt.plot(df['feature_1'])

plt.plot(df['feature_2'])
from fbprophet import Prophet
data = df.reset_index()

data.tail(n=3)
data2 = data[['time','feature_2']].reset_index()

data2.drop('index',axis=1,inplace=True)

data2.columns = ['ds', 'y']
#train test

prediction_size = 60

train_df2 = data2[:-60]

train_df2.tail()
m = Prophet()

m.fit(train_df2)
future = m.make_future_dataframe(periods=435, freq='10S')

future.tail(n=3)
forecast = m.predict(future)

forecast.tail(n=3)
m.plot_components(forecast)
fcast = forecast[504:563]['yhat']

fcast.head()
def score(df, fcast):

    

    df = pd.DataFrame()

    

    df['error'] = data2[504:563]['y'] - fcast

    df['relative_error'] = 100*df['error']/data2[504:563]['y']

    

    

    error_mean = lambda error_name: np.mean(np.abs(df[error_name]))

    

    

    return {'MAPE': error_mean('relative_error'), 'MAE': error_mean('error')}
for err_name, err_value in score(data2, fcast).items():

    print(err_name, err_value)
test = pd.read_csv('../input/into-the-future/test.csv')
d = forecast[564:]['yhat']
final = pd.DataFrame()

final['id'] = test['id']

final['feature_2'] = list(d)
final.head()
final.to_csv("/kaggle/working/solutionTERRABXT.csv", index=False)