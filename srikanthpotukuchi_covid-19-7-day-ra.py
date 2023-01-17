import pycountry

import plotly.express as px

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt
df1 = pd.read_csv("../input/covid19-countrywise/covid-19.csv")
df1 = df1.sort_values(by=['Country','Date'])
df1['Confirmed_diff'] = df1['Confirmed'].diff(periods=7)

df1['Deaths_diff'] = df1['Deaths'].diff(periods=7)
print(df1.head(3))  # Get first 3 entries in the dataframe
print(df1.tail(3))  # Get last 3 entries in the dataframe
df1 = df1.fillna(0)
list_countries = ['US','Brazil','India','Russia','Peru']

# colors for the 5 horizontal bars

list_colors = ['blue','green','red','black','purple']

df2 = df1[df1['Country'].isin(list_countries)]
df2 = df2[df2['Confirmed_diff']>=0]
df2.groupby(['Date','Country']).sum()['Confirmed_diff']
df2.groupby(['Date','Country']).sum()['Deaths_diff']
fig, ax = plt.subplots(figsize=(16,8))

ax.set_ylabel('Confirmed New Cases')

# use unstack()

df2.groupby(['Date','Country']).sum()['Confirmed_diff'].unstack().plot(ax=ax)
ax.clear()

fig, ax = plt.subplots(figsize=(15,7))

# use unstack()

df2.groupby(['Date','Country']).sum()['Deaths_diff'].unstack().plot(ax=ax)