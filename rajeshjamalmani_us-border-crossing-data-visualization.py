import pandas as pd

import numpy as np

import seaborn as sns

import os
border=pd.read_csv("../input/us-border-crossing-entry-data/Border_Crossing_Entry_Data.csv",parse_dates=['Date'])

border['Year']=pd.to_datetime(border['Date']).dt.year

border['Month']=pd.to_datetime(border['Date']).dt.month

border['Day']=pd.to_datetime(border['Date']).dt.day
border.head()
border.columns
border.dtypes
border.columns.values
sns.barplot('Border', 'Value',   estimator = np.sum, data = border)
ax=sns.barplot('State', 'Value',   estimator = np.sum, data = border)

var=ax.set_xticklabels(ax.get_xticklabels(),rotation = 80)
ax=sns.barplot('Measure', 'Value',   estimator = np.sum, data = border)

var=ax.set_xticklabels(ax.get_xticklabels(),rotation = 80)
sns.set(rc={'figure.figsize':(21.7,8.27)})
bardt=pd.DataFrame(border.groupby('Port Name')['Value'].sum().sort_values(ascending=False)).reset_index()

ax=sns.barplot(x='Port Name', y='Value',   data = bardt)

var=ax.set_xticklabels(ax.get_xticklabels(),rotation = 80)
bardt=pd.DataFrame(border.groupby('Year')['Value'].sum().sort_values(ascending=False)).reset_index()

ax=sns.barplot(x='Year', y='Value',   data = bardt)

var=ax.set_xticklabels(ax.get_xticklabels(),rotation = 80)
sns.catplot('Border','Year', data = border, kind = 'box')
sns.boxplot('Border','Month', data = border)
sns.catplot('Year','State', data = border, kind = 'box')
sns.catplot('Year','Measure', data = border, kind = 'box')
sns.boxplot('Year','Measure', data = border)
sns.scatterplot(x='Value',y='State',data=border,hue="State")
sns.pairplot(data=border, hue="State")
border.dtypes
heatdt=border.drop (['Port Name','State', 'Border','Measure','Location','Date'], axis=1)
heatdt.dtypes
import matplotlib.pyplot as plt

fig, ax=plt.subplots(figsize=(12,7))

title="US Border crossing - Heat Map"

plt.title(title,fontsize=18)

ttl=ax.title

ttl.set_position([0.5,1.05])

heatdt1=heatdt.pivot_table(index="Month", columns="Year", values="Value", aggfunc=np.sum)

ax=sns.heatmap(heatdt1)

#sns.heatmap(heatdt1,fmt="",cmap="RdYlGn",linewidth=0.3,ax=ax)

#plt.show()
ax=sns.heatmap(heatdt1,fmt="",cmap="YlGnBu",linewidth=0.3)

plt.show()