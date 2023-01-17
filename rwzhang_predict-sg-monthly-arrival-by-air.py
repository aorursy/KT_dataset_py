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
import matplotlib.pyplot as plt
import fbprophet
import matplotlib.axes as ax
data=pd.read_csv('../input/SG_Monthly_Visitor_Arrival_by_Air-2017.csv',header=0)
data['Month_Year'] =  pd.to_datetime(data['Month_Year'], format='%m/%Y')
df = data.set_index('Month_Year')
country=list(df)
plt.figure(num=None, figsize=(40, 40), dpi=80, facecolor='w', edgecolor='k')
for i in range(1,38):
   plt.subplot(7,6,i)
   plt.plot(df.index,  df.iloc[:,i])
   plt.title(country[i-1],fontsize=18)
   plt.grid(True)
plt.show()
data1=data.drop([128,129,130])
import fbprophet 
df_1 = data1.rename(columns={'Month_Year': 'ds', 'China': 'y'}).loc[:,['ds','y']]
model_China =fbprophet.Prophet()
model_China.fit(df_1)
future_data_China =model_China.make_future_dataframe(periods=24, freq = 'MS')
forecast_data_China  = model_China.predict(future_data_China)
forecast_data_China_1=forecast_data_China[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
f1=model_China.plot(forecast_data_China_1)
c1=model_China.plot_components(forecast_data_China)
f1
c1
df_1 = data1.rename(columns={'Month_Year': 'ds', 'Indonesia': 'y'}).loc[:,['ds','y']]
model_Indonesia =fbprophet.Prophet()
model_Indonesia.fit(df_1)
future_data_Indonesia = model_Indonesia.make_future_dataframe(periods=24, freq = 'MS')
forecast_data_Indonesia  = model_Indonesia.predict(future_data_Indonesia )
forecast_data_Indonesia_1=forecast_data_Indonesia[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
f2=model_Indonesia.plot(forecast_data_Indonesia_1)
c2=model_Indonesia.plot_components(forecast_data_Indonesia)
f2
c2
df_1 = data1.rename(columns={'Month_Year': 'ds', 'Australia': 'y'}).loc[:,['ds','y']]
model_Australia =fbprophet.Prophet()
model_Australia.fit(df_1)
future_data_Australia = model_Australia.make_future_dataframe(periods=24, freq = 'MS')
forecast_data_Australia  = model_Australia.predict(future_data_Australia)
forecast_data_Australia_1=forecast_data_Australia[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
f3=model_Australia.plot(forecast_data_Australia_1)
c3=model_Australia.plot_components(forecast_data_Australia)
f3
c3
df_1 = data1.rename(columns={'Month_Year': 'ds', 'Malaysia': 'y'}).loc[:,['ds','y']]
model_Malaysia =fbprophet.Prophet()
model_Malaysia.fit(df_1)
future_data_Malaysia = model_Malaysia.make_future_dataframe(periods=24, freq = 'MS')
forecast_data_Malaysia  = model_Malaysia.predict(future_data_Malaysia)
forecast_data_Malaysia_1=forecast_data_Malaysia[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
f4=model_Malaysia.plot(forecast_data_Malaysia_1)
c4=model_Malaysia.plot_components(forecast_data_Malaysia)
f4
c4
df_1 = data1.rename(columns={'Month_Year': 'ds', 'India': 'y'}).loc[:,['ds','y']]
model_India =fbprophet.Prophet()
model_India.fit(df_1)
future_data_India = model_India.make_future_dataframe(periods=24, freq = 'MS')
forecast_data_India  = model_India.predict(future_data_India)
forecast_data_India_1=forecast_data_India[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
f5=model_India.plot(forecast_data_India_1)
c5=model_India.plot_components(forecast_data_India)
f5
c5