# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import pandas as pd

from fbprophet import Prophet

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv')

df.head()
df['Month'] = pd.DatetimeIndex(df['Month'])

df.info()
df = df.rename(columns={'Month': 'ds',

                        '#Passengers': 'y'})



df.head(5)
ax = df.set_index('ds')



plt.figure(figsize=(15,8))

sns.lineplot(legend = 'full' , data=ax)
model=Prophet()

model.fit(df)
future = model.make_future_dataframe(periods=365)

future.tail()
forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
model.plot_components(forecast)