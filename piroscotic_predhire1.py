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

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

df=pd.read_excel('/kaggle/input/indeed-new/dfn.xlsx')
df.head()
df=df.replace(to_replace ="Date", value ="2020-04-17")
c=df[df['Title']=='Business Development Manager']['Date']
c=c.to_frame()
c.head()
c=c.sort_values(by='Date')
tsc=pd.DataFrame(c['Date'].value_counts())
tsc.head()
tsc.reset_index(inplace=True)
tsc=tsc.sort_values(by='index')
tsc.head()
tsc=tsc.drop(['level_0'], axis = 1)
tsc.set_index("index", inplace = True)
tsc.head()
tsc.reset_index(inplace=True)
tsc.head()
tsc.rename(columns = {'Date':'y','index':'ds'}, inplace = True) 
train_size = int(len(tsc) * 0.80)

train, test = tsc[0:train_size], tsc[train_size:len(tsc)]
from fbprophet import Prophet
my_model = Prophet(interval_width=0.95,daily_seasonality=True)
my_model.fit(train)

future = my_model.make_future_dataframe(periods=14,freq='D')

prophet_pred = my_model.predict(future)
future.tail()
my_model.plot(prophet_pred)
my_model.plot_components(prophet_pred)
print(', '.join(prophet_pred.columns))
def make_comparison_dataframe(historical, forecast):

  

    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
cmp_df = make_comparison_dataframe(tsc,prophet_pred)

cmp_df.tail(n=4)
import plotly.offline as py

import plotly.graph_objs as go
py.iplot([

    go.Scatter(x=tsc['ds'], y=tsc['y'], name='y'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat'], name='yhat'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=prophet_pred['ds'], y=prophet_pred['trend'], name='Trend')

])
print('RMSE: %f' % np.sqrt(np.mean((cmp_df['yhat']-cmp_df['y'])**2)))