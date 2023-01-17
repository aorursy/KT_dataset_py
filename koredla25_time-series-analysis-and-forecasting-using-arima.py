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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/portland-oregon-avg-rider-monthly-data/portland-oregon-average-monthly-.csv')
df.shape
df.columns
df.head()
df.info()
df.describe()
df.columns=["month","average_monthly_ridership"]
df.head()
df.dtypes
df['average_monthly_ridership'].unique()
df=df.drop(df.index[df['average_monthly_ridership']==' n=114'])
df.shape
df.head(3)
df.dtypes
df['average_monthly_ridership'] = df['average_monthly_ridership'].astype(np.int32)
df.dtypes
df['month'] = pd.to_datetime(df['month'])
df.dtypes
# Normal line plot so we can see data variation 
df.plot.line(x='month',y='average_monthly_ridership')
df1=df
#only store month
mon = df1['month']
print(mon.shape)
print(mon.head(2))
temp=pd.DatetimeIndex(mon)
temp.dtype
temp.month
temp.year
month = pd.Series(temp.month)
print(month.dtype)
print(month.head(2))
df1.columns
df1=df1.drop(['month'],axis=1)
df1.head(5)
df1=df1.join(month)
df1.head(5)
sns.barplot(x='month',y='average_monthly_ridership',data=df1)
plt.show()
df1.plot.scatter(x='month',y='average_monthly_ridership')
plt.show()
df2=df[['average_monthly_ridership']]
df2.head(2)
df2.dtypes
df2.rolling(6).mean().plot(figsize=(20,10),linewidth=5,fontsize=20)
df2.diff(periods =4).plot(figsize=(20,10), linewidth=5,fontsize=20)
plt.show()
pd.plotting.autocorrelation_plot(df['average_monthly_ridership'])
plt.show()
pd.plotting.lag_plot(df['average_monthly_ridership'])
plt.show()
df = df.set_index('month')
# Applying Seasonal ARIMA model to forcast the data 
import statsmodels.api as sm
mod = sm.tsa.SARIMAX(df['average_monthly_ridership'], trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
results = mod.fit()
print(results.summary())
df['forecast'] = results.predict(start = 102, end= 120, dynamic= True)  
df[['average_monthly_ridership', 'forecast']].plot(figsize=(12, 8))
plt.show()
def forcasting_future_months(df, no_of_months):
    df_perdict = df.reset_index()
    mon = df_perdict['month']
    mon = mon + pd.DateOffset(months = no_of_months)
    future_dates = mon[-no_of_months -1:]
    df_perdict = df_perdict.set_index('month')
    future = pd.DataFrame(index=future_dates, columns= df_perdict.columns)
    df_perdict = pd.concat([df_perdict, future])
    df_perdict['forecast'] = results.predict(start = 114, end = 125, dynamic= True)  
    df_perdict[['average_monthly_ridership', 'forecast']].iloc[-no_of_months - 12:].plot(figsize=(12, 8))
    plt.show()
    return df_perdict[-no_of_months:]
predicted = forcasting_future_months(df,10)
df.tail()
