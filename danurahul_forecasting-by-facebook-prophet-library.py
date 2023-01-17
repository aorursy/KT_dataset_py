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
df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Feature-Engineering-Live-sessions/master/monthly-milk-production-pounds.csv')
df.head()
df.columns=['ds','y']
df
df.drop(168,axis=0,inplace=True)
df.plot()
df['ds']=pd.to_datetime(df['ds'])
df.head()
df['y'].plot()

df['y'].head()
df['y'].plot()
from fbprophet import Prophet
import matplotlib.pyplot as plt
%matplotlib inline
dir(Prophet)
#intialize the model
model=Prophet()
model.fit(df)
model
model.seasonalities
model.component_modes
#future dates of 365 days
future_dates=model.make_future_dataframe(periods=365)
df.tail()
future_dates
#prediction
prediction=model.predict(future_dates)
prediction.head()
prediction[['ds','yhat','yhat_lower','yhat_upper',]].tail()
prediction[['ds','yhat','yhat_lower','yhat_upper',]].head()
###plot the predicted prediction
model.plot(prediction)
###visualize each componeent trends,weekly
model.plot_components(prediction)
df.shape
from fbprophet.diagnostics import cross_validation
df_cv=cross_validation(model,horizon="365 days",period='180 days',initial='1095 days')
df.head()
df_cv.head()
from fbprophet.diagnostics import performance_metrics
df_performance=performance_metrics(df_cv,)
df_performance.head()
df_performance
from fbprophet.plot import plot_cross_validation_metric
fig=plot_cross_validation_metric(df_cv,metric='rmse')
