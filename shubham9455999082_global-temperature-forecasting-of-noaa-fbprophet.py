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
data=pd.read_csv('/kaggle/input/forecasting-dataset-of-noaa/2010.csv')
data.head()
data2=data[['lat','lon','Date','tem']]
data2.head()
x2=data2['lat'].astype('str')
x1=data2['lon'].astype('str')
data2['lat_lon']=x2+'_'+x1
data2.head()
from fbprophet import Prophet
data2 = data2.rename(columns={'Date': 'ds', 'tem':'y'})
data2.head(1)
grouped = data2.groupby('lat_lon')
final = pd.DataFrame()
for g in grouped.groups:
    group = grouped.get_group(g)
    m = Prophet()
    m.fit(group)
    future = m.make_future_dataframe(periods=700)
    forecast = m.predict(future)    
    forecast = forecast.rename(columns={'yhat': 'yhat_'+str(g)})
    final = pd.merge(final,  forecast.set_index('ds'), how='outer', left_index=True,right_index=True)

final = final[['yhat_' + str(g) for g in grouped.groups.keys()]]
final
final.iloc[0:, 0:5]['2020-06-17':'2020-06-30']

final