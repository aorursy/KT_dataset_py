import numpy as np 
import pandas as pd 
import os
import datetime
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")
df.head()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.describe().T
df.info()
dates = [datetime.datetime.strptime(ts, "%Y-%m-%d") for ts in df['Date']]
dates.sort()
sorteddates = [datetime.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
df['Date'] = pd.DataFrame({'Date':sorteddates})
df['Year'], df['Month'],  df['Day'] = df['Date'].str.split('-').str
df.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.figure(figsize=(12,5))
plt.title("Price Distribution Graph")
ax = sns.distplot(df["AveragePrice"], color = 'y')
df.tail()
dategroup=df.groupby('Date').mean()
plt.figure(figsize=(12,5))
dategroup['AveragePrice'].plot(x=df.Date)
plt.title('Average Price')
dategroup=df.groupby('Month').mean()
fig, ax = plt.subplots(figsize=(12,5))
ax.xaxis.set(ticks=range(0,13))
dategroup['AveragePrice'].plot(x=df.Month)
plt.title('Average Price by Month')
dategroup=df.groupby('Day').mean()
fig, ax = plt.subplots(figsize=(12,5))
ax.xaxis.set(ticks=range(0,31))
dategroup['AveragePrice'].plot(x=df.Day)
plt.title('Average Price by Day')
plt.figure(figsize=(20,20))
sns.set_style('whitegrid')
sns.pointplot(x='AveragePrice',y='region',data=df, hue='Year',join=False)
plt.xticks(np.linspace(1,2,5))
plt.xlabel('Region',{'fontsize' : 'large'})
plt.ylabel('Average Price',{'fontsize':'large'})
plt.title("Yearly Average Price in Each Region",{'fontsize':20})
print(df['type'].value_counts())
plt.figure(figsize=(12,5))
sns.countplot(df['type'])
plt.show()
from fbprophet import Prophet

dff = df.loc[:, ["Date","AveragePrice"]]
dff['Date'] = pd.DatetimeIndex(dff['Date'])
dff.dtypes
dff = dff.rename(columns={'Date': 'ds',
                        'AveragePrice': 'y'})

ax = dff.set_index('ds').plot(figsize=(20, 12))
ax.set_ylabel('Monthly Average Price of Avocado')
ax.set_xlabel('Date')
plt.show()
my_model = Prophet()
my_model.fit(dff)

future_dates = my_model.make_future_dataframe(periods=900)
forecast = my_model.predict(future_dates)
fig2 = my_model.plot_components(forecast)
