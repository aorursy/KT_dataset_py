# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/,but they won't be saved outside of the current session
df=pd.read_excel('/kaggle/input/integrateddata/integrated data.xlsx')
df.head()
#df.set_index('index',inplace=True)
df.rename(columns = {'Date':'y','index':'ds'}, inplace = True)
train,validate,test=np.split(df, [int(.6 * len(df)), int(.8 * len(df))])
train.head()
from fbprophet import Prophet
my_model = Prophet(interval_width=0.95,daily_seasonality=True)
my_model.fit(train)

future = my_model.make_future_dataframe(periods=30,freq='D')

prophet_pred = my_model.predict(future)
print(', '.join(prophet_pred.columns))
def make_comparison_dataframe(historical, forecast):

  

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
cmp_df = make_comparison_dataframe(validate,prophet_pred)

cmp_df.tail(n=4)
import plotly.offline as py

import plotly.graph_objs as go
py.iplot([

    go.Scatter(x=validate['ds'], y=validate['y'], name='y'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat'], name='yhat'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['trend'], name='Trend')

])
test_1=pd.concat([train, validate])
my_model = Prophet(interval_width=0.95,daily_seasonality=True)
my_model.fit(test_1)

future = my_model.make_future_dataframe(periods=30,freq='D')

prophet_pred = my_model.predict(future)
print(', '.join(prophet_pred.columns))
def make_comparison_dataframe(historical, forecast):

  

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
cmp_df = make_comparison_dataframe(test,prophet_pred)

cmp_df.tail(n=4)
py.iplot([

    go.Scatter(x=test['ds'], y=test['y'], name='y'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat'], name='yhat'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['trend'], name='Trend')

])
df.set_index('ds',inplace=True)
df.head()
train,validate,test=np.split(df, [int(.6 * len(df)), int(.8 * len(df))])
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train,  

                order = (1, 0, 1),  

                seasonal_order =(0, 1, 1, 12)) 

  

result = model.fit() 

result.summary()
predictions = result.predict(start='2020-05-19', end='2020-06-15', 

                             typ = 'levels').rename("Predictions")
predictions=predictions.to_frame()
validate.tail()
predictions.head()
predictions['Actual']=validate['y']
predictions
test_1.set_index('ds',inplace=True)
model = SARIMAX(test_1,  

                order = (1, 0, 1),  

                seasonal_order =(0, 1, 1, 12)) 

  

result = model.fit() 

result.summary()
test.tail()
predictions = result.predict(start='2020-06-16', end='2020-07-13', 

                             typ = 'levels').rename("Predictions")
predictions=predictions.to_frame()
predictions['Actual']=test['y']
predictions.head()
model = SARIMAX(test_1,  

                order = (1, 0, 1),  

                seasonal_order =(0, 1, 1, 12)) 

  

result = model.fit() 

result.summary()
predictions = result.predict(start='2020-06-16', end='2020-07-13', 

                             typ = 'levels').rename("Predictions")
predictions=predictions.to_frame()
predictions['Actual']=test['y']
predictions.head()