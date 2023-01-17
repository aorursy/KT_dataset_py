import numpy as np

import pandas as pd

import os

path = '/kaggle/input/ntt-data-global-ai-challenge-06-2020/'

from fbprophet import Prophet

import plotly.graph_objects as go
# Covid 19 and Price data

covid_df = pd.read_csv(path+"COVID-19_and_Price_dataset.csv")

covid_df['Date'] = pd.to_datetime(covid_df['Date']) # convert Date column into Datetime type

# see first 5 and last 5 rows 

sliced_df = covid_df.head(5) # first 5 rows

sliced_df = sliced_df.append(covid_df.tail(5)) # last 5 rows

print("We have covid-19 data captured from {} to {}".format(str(sliced_df.Date[0]).split()[0],str(sliced_df.Date.iloc[-1]).split()[0]))

sliced_df.style.set_properties(**{'background-color': 'white',                                                   

                                    'color': 'red',                       

                                    'border-color': 'black'})
# Crude oil data

oil_df = pd.read_csv(path+"Crude_oil_trend.csv")

oil_df['Date'] = pd.to_datetime(oil_df['Date'])

# first 5 and last 5 rows

sliced_df = oil_df.head(5)

sliced_df = sliced_df.append(oil_df.tail(5))

print("We have 75 day moving average Crude oil price data from {} to {}".format(str(sliced_df.Date[0]).split()[0],str(sliced_df.Date.iloc[-1]).split()[0]))

sliced_df.style.set_properties(**{'background-color': 'black',                                                   

                                    'color': 'lawngreen',                       

                                    'border-color': 'red'})
del sliced_df # its a good practice to delete useless objects # save memory 



cols = ['SaudiArabia','UnitedArabEmirates','UnitedStates','Russia','Italy','China'] 

cols = ['Date'] + [i+'_new_cases' for i in cols]



covid_df_sliced = covid_df[cols]



# We will use the 75 day moving average crude oil price

df = pd.merge(covid_df_sliced, oil_df, on='Date')



cor_df = df.corr()



cor_df.style.background_gradient(cmap='magma_r')
# > 0.80 Correlation 

cols_ = ['Date'] + [i for i in covid_df.columns if i.endswith('new_cases')]

df_new_cases = covid_df[cols_]

df_ = pd.merge(df_new_cases, oil_df, on='Date')

cor_df_ = df_.corr()

i = abs(cor_df_['Price']) > 0.80

cor_df_ = cor_df_.loc[i, i]

cor_df_.style.background_gradient(cmap='magma_r')
del cor_df,cor_df_ # save memory even if its of less size

# removing china and Italy

df.drop(['Italy_new_cases','China_new_cases'],axis = 1, inplace = True)

# set date as index

df.set_index('Date',inplace = True)

# train

train = df[:'2020-06-05']

# test

test = df['2020-06-06':]



print("Train Test Split Done!")



train.tail()
test.head()
train_df = pd.DataFrame()

train_df['ds'] = train.index

train_df['y'] = train.Price.values

train_df['saudi'] = train.SaudiArabia_new_cases.values

train_df['uae'] = train.UnitedArabEmirates_new_cases.values

train_df['us'] = train.UnitedStates_new_cases.values

train_df['russia'] = train.Russia_new_cases.values

train_df.tail()
test_df = pd.DataFrame()

test_df['ds'] = test.index

test_df['y'] = test.Price.values

test_df['saudi'] = test.SaudiArabia_new_cases.values

test_df['uae'] = test.UnitedArabEmirates_new_cases.values

test_df['us'] = test.UnitedStates_new_cases.values

test_df['russia'] = test.Russia_new_cases.values

test_df.tail()
model = Prophet()

model.add_regressor('saudi')

model.add_regressor('uae')

model.add_regressor('us')

model.add_regressor('russia')

model.fit(train_df)
y_pred = model.predict(test_df.drop(columns='y'))

y_pred
fig = go.Figure()

fig.add_trace(go.Scatter(x=y_pred['ds'], y=y_pred['yhat'],

                    mode='lines',

                    name='Predicted Oil Prices'))

fig.add_trace(go.Scatter(x=test.index, y=test['Price'],

                    mode='lines',

                    name='Actual Oil Prices'))



# Edit the layout

fig.update_layout(title='Actual vs Predicted Price: Test data',

                   xaxis_title='Date',

                   yaxis_title='Price')

fig.show()
from sklearn.metrics import mean_squared_error

print("Test RMSE: ",np.sqrt(mean_squared_error(test['Price'],y_pred['yhat'])))
# Load Sample Submission

submission_file = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/sampleSubmission0710_updated.csv')

test_index = np.where(submission_file.Date == '2020-07-07')[0][0]

test_file = submission_file[test_index:].reset_index(drop = True)

test_file['Date'] = pd.to_datetime(test_file['Date'])



# Prophet time series

from fbprophet import Prophet

d={'ds':df.index,'y':df['Price']}

df_pred=pd.DataFrame(data=d)

model = Prophet(daily_seasonality=False)

model.fit(df_pred)



import matplotlib.pyplot as plt

future = model.make_future_dataframe(periods=len(test_file))

forecast = model.predict(future)





# Submission file

test_file['Price'] = forecast['yhat'][-len(test_file):].reset_index(drop = True)

pretest_file = df['2020-04-29':'2020-07-07']['Price'].reset_index()

test_price = pd.concat([pretest_file,test_file],ignore_index = True)

test_price['Date'] = pd.to_datetime(test_price['Date'])

submission_file['Date'] = pd.to_datetime(submission_file['Date'])

submission_file = submission_file.drop('Price',axis=1)

submission = pd.merge(submission_file, test_price, on='Date', how = 'left')



# July 3 is missing - replace with avg

a = submission[submission['Date'] == '2020-07-02']['Price'].reset_index(drop=True)[0]

b = submission[submission['Date'] == '2020-07-06']['Price'].reset_index(drop=True)[0]

submission = submission.fillna((a+b)/2)

submission.to_csv('submission.csv', index=False)