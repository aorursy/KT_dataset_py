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
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.tsa.statespace.sarimax as sarima

%matplotlib inline
df = pd.read_csv('../input/forecast/GC_ARIMA.csv', index_col='Date', parse_dates=True)

df=df.iloc[:73]

df
train = df['02-06-20':'17-06-20']

test = df['17-06-20':]
D1train= sarima.SARIMAX(train, order=(3,0,1), trend='c').fit(disp=1)

forecast= D1train.forecast(steps=len(test))


fig1, ax = plt.subplots()

ax.plot(train, label='train')

ax.plot(test, label='test')

ax.plot(forecast, label='forecast')

plt.legend(loc='upper left')

plt.title("Differentiated First Order AR ARIMA (3,0,1)C")

plt.ylabel('Cases')

plt.xlabel('Date')

plt.show()