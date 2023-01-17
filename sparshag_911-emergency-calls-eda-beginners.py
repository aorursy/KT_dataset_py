import pandas as pd
import matplotlib.pyplot as plt
import plotly
plotly.offline.init_notebook_mode()
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import seaborn as sns
import calendar
%matplotlib inline

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/911.csv')
df.head()
df.info()
df.describe()
#todo1: convert zip dtype to object
df['zip'] = df['zip'].astype('object')

#todo2: drop column 'e'
df = df.drop('e', axis=1)
plt.subplots(figsize=(20,5))
df['twp'].value_counts().plot(kind='bar')
# No. of Reasons
np.unique(df['title']).size
# top 20 main reasons
plt.subplots(figsize=(8,6))
df['title'].value_counts().sort_values(ascending=False).head(20).plot(kind='barh')
plt.gca().invert_yaxis()
# Let's break title columns into 'type: EMS, Traffic, Fire etc..' and 'subtype: 'Vehicle accident, fire alarm etc...'
df['type'], df['subtype'] = df['title'].str.split(': ', 1).str
df = df.drop('title', axis=1) #drop 'title' columns
# Let's now see types of calls
sns.countplot(x='type', data=df)
#We need to purify subtype column a little bit more - replacing (+ with &) and removing - sign.
df['subtype'] = df['subtype'].replace({'\+': '&', '\-': ''}, regex=True).map(lambda x: x.strip())
total = df['subtype'].value_counts().sort_values(ascending=False)
percent = (df['subtype'].value_counts()*100/df['subtype'].value_counts().sum()).sort_values(ascending=False)
subtype_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
subtype_data.head(10)
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
#Extracting Date and time
df['year'] = df['timeStamp'].dt.year
df['month'] = df['timeStamp'].dt.month
df['day'] = df['timeStamp'].dt.day
df['hour'] = df['timeStamp'].dt.hour
df['minute'] = df['timeStamp'].dt.minute
df['weekday'] = df['timeStamp'].dt.weekday_name
df = df.drop('timeStamp', axis=1)
df.head()
fig,ax = plt.subplots(3, 2, figsize=(20, 20))
df[['type','year']].pivot_table(index=['year'], columns=['type'], aggfunc=np.count_nonzero).plot(ax=ax[0][0])
df[['type','month']].pivot_table(index=['month'], columns=['type'], aggfunc=np.count_nonzero).plot(ax=ax[0][1])
df[['type','day']].pivot_table(index=['day'], columns=['type'], aggfunc=np.count_nonzero).plot(ax=ax[1][0])
df[['type','weekday']].pivot_table(index=['weekday'], columns=['type'], aggfunc=np.count_nonzero).plot(ax=ax[1][1])
df[['type','hour']].pivot_table(index=['hour'], columns=['type'], aggfunc=np.count_nonzero).plot(ax=ax[2][0])
df[['type','minute']].pivot_table(index=['minute'], columns=['type'], aggfunc=np.count_nonzero).plot(ax=ax[2][1])
sns.set()
def calls_heatmap(index,col):
    temp = df[[index,col]].pivot_table(index=[index], columns=[col], aggfunc=np.count_nonzero).fillna(0).astype(int)
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(temp, square=True, cmap="RdYlBu", cbar=False, ax=ax, linewidths=.5)
    plt.show()
#Emergency call frequency day-month basis
calls_heatmap('month', 'day')
#Emergency call frequency hour-day basis
calls_heatmap('hour', 'day')
#Emergency call frequency month-day basis
calls_heatmap('hour', 'minute')
#Emergency call frequency hour-day basis
calls_heatmap('weekday', 'day')
## 3 type of call-frequency comparison using heatmaps
sns.set()
temp_EMS = df.loc[df['type']=='EMS',['hour', 'day']].pivot_table(index=['hour'], columns=['day'], aggfunc=np.count_nonzero).fillna(0).astype(int)
temp_Fire = df.loc[df['type']=='Fire',['hour', 'day']].pivot_table(index=['hour'], columns=['day'], aggfunc=np.count_nonzero).fillna(0).astype(int)
temp_Traffic = df.loc[df['type']=='Traffic',['hour', 'day']].pivot_table(index=['hour'], columns=['day'], aggfunc=np.count_nonzero).fillna(0).astype(int)

fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
sns.heatmap(temp_EMS, square=True, cmap="RdYlBu", cbar=False, linewidths=.5, ax=ax1)
sns.heatmap(temp_Fire, square=True, cmap="RdYlBu", cbar=False, linewidths=.5, ax=ax2)
sns.heatmap(temp_Traffic, square=True, cmap="RdYlBu", cbar=False, linewidths=.5, ax=ax3)

ax1.set_title('EMS Calls')
ax2.set_title('Fire Calls')
ax3.set_title('Traffic Calls')

plt.show()
## 3 type of call-frequency comparison using heatmaps
sns.set()
temp_EMS1 = df.loc[df['type']=='EMS',['hour', 'weekday']].pivot_table(index=['hour'], columns=['weekday'], aggfunc=np.count_nonzero).fillna(0).astype(int)
temp_Fire1 = df.loc[df['type']=='Fire',['hour', 'weekday']].pivot_table(index=['hour'], columns=['weekday'], aggfunc=np.count_nonzero).fillna(0).astype(int)
temp_Traffic1 = df.loc[df['type']=='Traffic',['hour', 'weekday']].pivot_table(index=['hour'], columns=['weekday'], aggfunc=np.count_nonzero).fillna(0).astype(int)

fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(temp_EMS1, square=True, cmap="RdYlBu", cbar=False, linewidths=.5, ax=ax1)
sns.heatmap(temp_Fire1, square=True, cmap="RdYlBu", cbar=False, linewidths=.5, ax=ax2)
sns.heatmap(temp_Traffic1, square=True, cmap="RdYlBu", cbar=False, linewidths=.5, ax=ax3)

ax1.set_title('EMS Calls')
ax2.set_title('Fire Calls')
ax3.set_title('Traffic Calls')

plt.show()
sns.jointplot(x='lat', y='lng', data=df, kind='scatter')
# Removing outliers - SD of 4 and 10 as a limit of lat and lng respectively to categorize outliers
df_geo=df[(np.abs(df["lat"]-df["lat"].mean())<=(4*df["lat"].std())) & (np.abs(df["lng"]-df["lng"].mean())<=(10*df["lng"].std()))]
df_geo.reset_index().drop('index',axis=1,inplace=True)
sns.jointplot(data=df_geo,x='lng',y='lat',kind='scatter')
#standardizing the column values of lat and long
pd.options.mode.chained_assignment = None #Remove Error Message
x_mean=df_geo['lng'].mean()
y_mean=df_geo['lat'].mean()
df_geo['x']=df_geo['lng'].map(lambda v:v-x_mean)
df_geo['y']=df_geo['lat'].map(lambda v:v-y_mean)
sns.jointplot(data=df_geo,x='x',y='y',kind='scatter')
sns.lmplot(x='x', y='y', hue='type', data=df_geo,fit_reg=False)
sns.lmplot(x='x', y='y', hue='type',col='year', data=df_geo,fit_reg=False)
# Clustering lat-lng to map townships
group_town=df_geo.groupby('twp')
for name, group in group_town:
    plt.plot(group.x, group.y, marker='o', linestyle='', label=name)
plt.xlim(-0.5,0.4)
plt.title("Townships")
