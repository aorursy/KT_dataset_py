# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading the data
df = pd.read_csv("../input/portland-oregon-average-monthly-.csv")
# A glance on the data 
df.head()
# getting some information about dataset
df.info()
# further Analysis 
df.describe()
df.columns = ["month", "average_monthly_ridership"]
df.head()
df.dtypes
df['average_monthly_ridership'].unique()
df = df.drop(df.index[df['average_monthly_ridership'] == ' n=114'])
df['average_monthly_ridership'].unique()
df['average_monthly_ridership'] = df['average_monthly_ridership'].astype(np.int32)
df['month'] = pd.to_datetime(df['month'], format = '%Y-%m')
df.dtypes
# Normal line plot so that we can see data variation
# We can observe that average number of riders is increasing most of the time
# We'll later see decomposed analysis of that curve
df.plot.line(x = 'month', y = 'average_monthly_ridership')
plt.show()
to_plot_monthly_variation = df
# only storing month for each index 
mon = df['month']
# decompose yyyy-mm data-type 
temp= pd.DatetimeIndex(mon)
# assign month part of that data to ```month``` variable
month = pd.Series(temp.month)
# dropping month from to_plot_monthly_variation
to_plot_monthly_variation = to_plot_monthly_variation.drop(['month'], axis = 1)
# join months so we can get month to average monthly rider mapping
to_plot_monthly_variation = to_plot_monthly_variation.join(month)
# A quick glance
to_plot_monthly_variation.head()
# Plotting bar plot for each month
sns.barplot(x = 'month', y = 'average_monthly_ridership', data = to_plot_monthly_variation)
plt.show()
to_plot_monthly_variation.plot.scatter(x = 'month', y = 'average_monthly_ridership')
plt.show()
rider = df[['average_monthly_ridership']]
rider.rolling(6).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()
rider.diff(periods=4).plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()
pd.plotting.autocorrelation_plot(df['average_monthly_ridership'])
plt.show()
pd.plotting.lag_plot(df['average_monthly_ridership'])
plt.show()
df = df.set_index('month')
# Applying Seasonal ARIMA model to forcast the data 
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
