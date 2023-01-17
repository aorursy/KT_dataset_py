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
#Libraries



import numpy as np

import pandas as pd

import seaborn as sns



import matplotlib.pyplot as plt

from datetime import datetime

df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv", usecols = ["is_canceled","arrival_date_year","arrival_date_month","arrival_date_day_of_month","adults","children","babies"])
df.head()
df = df[df.is_canceled != 1]
df.columns
df.shape
len(df)
df["total_person"]= df.iloc[:, -3:-1].sum(axis=1)
cols = ["arrival_date_year","arrival_date_month","arrival_date_day_of_month"]

df['date'] = df[cols].apply(lambda row: '/'.join(row.values.astype(str)), axis=1)

df= df.drop(["adults","children","babies","arrival_date_year","arrival_date_month","arrival_date_day_of_month","is_canceled"], axis = 1)

df.head()
df.head()
df['date'] =  pd.to_datetime(df['date'],

                              format='%Y/%B/%d')
df["date"].value_counts()
main_df=df.groupby(df['date'].dt.date).sum()
main_df.head()
main_df.describe()
main_df.hist()

plt.show()
import seaborn as sns

# Use seaborn style defaults and set the default figure size

sns.set(rc={'figure.figsize':(11, 4)})

main_df['total_person'].plot(linewidth=0.5);
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(12).mean()

    rolstd = timeseries.rolling(12).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
test_stationarity(main_df)