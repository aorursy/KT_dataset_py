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


import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import statistics

import statsmodels

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



df = pd.read_csv('../input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')

df=df.drop('Unnamed: 0',axis=1)

df.describe(include='all')

df.info(verbose=True)

col_list=list(df)

#excluding the "0" the column, date column, as it will be converted to date-time object.

df[col_list[1:]] = df[col_list[1:]].apply(pd.to_numeric, errors='coerce')

df['Time Serie']=pd.to_datetime(df['Time Serie'])

df.info(verbose=True)

df_melted=df.melt(id_vars=["Time Serie"], 

        var_name="Currency type", 

        value_name="Value")



df_melted_india=df_melted.loc[df_melted['Currency type'] == "INDIA - INDIAN RUPEE/US$"]



india_data=df_melted_india.drop('Currency type',axis=1).rename(columns={'Time Serie':'ds','Value':'y'})



india_data.fillna(india_data.mean(), inplace=True)

india_data.ds= pd.to_datetime(india_data.ds)

india_data=india_data.set_index(['ds'])


# SARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX





# fit model

model = SARIMAX(india_data, order=(1, 1, 1), seasonal_order=(2, 2, 2, 2))

model_fit = model.fit(disp=False)
yhat3 = model_fit.predict()

y3=yhat3.to_frame()


from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

y1=yhat3.to_frame()

print(mean_absolute_error(india_data.y, y1))

print(mean_squared_error(india_data.y, y1))

print(np.sqrt(mean_squared_error(india_data.y, y1)))