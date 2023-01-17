# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/avocado.csv')
data.head()
#drop the unnamed column since it doesnot contribute in the analysis
data=data.drop('Unnamed: 0',axis=1)
data.dtypes
#convert the data column from object datatype to datetype
data['Date']=pd.to_datetime(data['Date'])
#we have two types of data in the type we will see those 
data.groupby('type').groups
#lets predict it for both organic as well as conventional 
data=data[data.type=='conventional']
#lets see the each group has how many observations
regions=data.groupby(data.region) 
new_list = [{name:len(group)} for name,group in regions]
new_list
 #we can see that each of these regions have 169 observations
#lets Predict the forecast for Indianapolis region
#Indianapolis_data=data[data.region=='Indianapolis']
PREDICTING_FOR = "Indianapolis"
date_price = regions.get_group(PREDICTING_FOR)[['Date', 'AveragePrice']].reset_index(drop=True)
date_price.plot(x='Date', y='AveragePrice', kind="line")
from fbprophet import Prophet
#lets convert the date price values to ds and y since it requires ds,y form
date_price=date_price.rename(columns={'Date':'ds', 'AveragePrice':'y'})
m = Prophet()
m.fit(date_price)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

m.plot(forecast)
#plot the individual components of forecast: trend,weekly/yearly,seasonality
m.plot_components(forecast)
#adding holidays to the dataset
Holidays_list = pd.DataFrame({
  'holiday': 'Holidays_list',
  'ds': pd.to_datetime(['2018-02-04','2018-02-11','2018-03-18','2018-04-01','2018-05-06','2018-05-27',
                     '2017-12-31','2017-12-25','2017-11-26','2017-10-29','2017-06-18','2017-07-02',
                     '2017-09-03','2017-05-28','2017-05-07','2017-04-16','2017-03-19','2017-02-05',
                     '2017-02-12','2017-01-01','2016-12-25','2016-11-27','2016-10-30','2016-09-04',
                     '2016-07-03','2016-06-19','2016-05-29','2016-05-08','2016-03-27','2016-03-13',
                     '2016-02-14','2016-02-07','2016-01-03','2015-12-31','2015-12-25','2015-11-24',
                     '2015-10-31','2015-09-05','2015-07-04','2015-06-19','2016-05-29','2015-05-08',
                     '2015-03-27','2015-03-13','2015-02-14','2015-02-07','2015-01-03']),
  'lower_window': 0,
  'upper_window': 1,
})
m = Prophet(holidays=Holidays_list)
forecast = m.fit(date_price).predict(future)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
fig = m.plot_components(forecast)