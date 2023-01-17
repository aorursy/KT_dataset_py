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
from fbprophet import Prophet 

from fbprophet.plot import plot_plotly

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("../input/avocado-prices/avocado.csv")
data.head()
data.info()
le = LabelEncoder()

data.iloc[:,11] = le.fit_transform(data.iloc[:,11])

data.head()
X= data[['Date','Total Volume','4046','4225','Total Bags','Small Bags','XLarge Bags','type']]

y= data['AveragePrice']
train = pd.DataFrame()

train['ds'] = pd.to_datetime(X["Date"])



train['y'] = data['AveragePrice']
prophet_basic = Prophet()

prophet_basic.fit(train)
future = prophet_basic.make_future_dataframe(periods=300)

future.head()
forcast = prophet_basic.predict(future)
fig1 = prophet_basic.plot(forcast)
fig2 =  prophet_basic.plot_components(forcast)
from fbprophet.plot import add_changepoints_to_plot

fig = prophet_basic.plot(forcast)

a = add_changepoints_to_plot(fig.gca(),prophet_basic,forcast)
prophet_basic.changepoints
prophet_pro = Prophet(changepoint_range=0.9)

prophet_pro.fit(train)

new_forcast = prophet_pro.predict(future)

fig4 = prophet_pro.plot(new_forcast)

b = add_changepoints_to_plot(fig4.gca(),prophet_pro,new_forcast)
pro_change = Prophet(changepoint_prior_scale = 0.08,n_changepoints = 20,yearly_seasonality=True)

pro_change.fit(train)

more_flexible_forcast = pro_change.predict(future)

fig5 = pro_change.plot(more_flexible_forcast)

b = add_changepoints_to_plot(fig5.gca(),pro_change,more_flexible_forcast)
pro_change = Prophet(changepoint_prior_scale = 0.01,n_changepoints = 20,yearly_seasonality=True)

pro_change.fit(train)

less_flexible_forcast = pro_change.predict(future)

fig5 = pro_change.plot(less_flexible_forcast)

b = add_changepoints_to_plot(fig5.gca(),pro_change,less_flexible_forcast)
avocado_season = pd.DataFrame({

  'holiday': 'avocado season',

  'ds': pd.to_datetime(['2014-07-31', '2014-09-16', 

                        '2015-07-31', '2015-09-16',

                        '2016-07-31', '2016-09-16',

                        '2017-07-31', '2017-09-16',

                       '2018-07-31', '2018-09-16',

                        '2019-07-31', '2019-09-16']),

  'lower_window': -1,

  'upper_window': 0,

})

avocado_season.head()
pro_holiday = Prophet(holidays=avocado_season)

pro_holiday.fit(train)

future_data = pro_holiday.make_future_dataframe(periods=12,freq = 'm')

##############

forcast_data = pro_holiday.predict(future_data)

pro_holiday.plot(forcast_data)
X.head()
train['Total Volume'] = X['Total Volume']

train['4046'] = X['4046']

train['4225'] = X['4225']

train['Total Bags'] = X['Total Bags']

train['Small Bags'] = X['Small Bags']

train['type'] = X['type']
train_X= train[:18000]

test_X= train[18000:]
pro_regressor = Prophet()

pro_regressor.add_regressor('Total Volume')

pro_regressor.add_regressor('4046')

pro_regressor.add_regressor('4225')

pro_regressor.add_regressor('Total Bags')

pro_regressor.add_regressor('Small Bags')

pro_regressor.add_regressor('type')
pro_regressor.fit(train_X)

forcast_data = pro_regressor.predict(test_X)

pro_regressor.plot(forcast_data)