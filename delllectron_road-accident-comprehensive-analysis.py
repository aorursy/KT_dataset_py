import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

#plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import folium
import datetime
import calendar



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#get the data
df = pd.read_csv('../input/nypd-motor-vehicle-collisions/nypd-motor-vehicle-collisions.csv', dtype=str)
#sohw dataframe
df.head(3)
#dataset shape
print('DATASET SHAPE: ', df.shape)
#show feature data types
df.info()
#replace capslock to lowercase
df.columns = [i.lower() for i in df.columns]
#date to pandas datetime object
df['accident date'] = pd.to_datetime(df['accident date'])
df['accident time'] = pd.to_datetime(df['accident time']).dt.time
#convert back the numeric features
num_feat = [i for i in df.columns if 'number' in i] + ['latitude', 'longitude']
df[num_feat] = df[num_feat].apply(pd.to_numeric, errors='coerce')
#show null value percentage per feature
pd.DataFrame(df.isnull().sum() / df.shape[0] *100, columns=['Missing Value %'])
plt.figure(figsize=(12,5))
plt.title('HEATMAP OF MISSING VALUES', fontsize=18)
sns.heatmap(df.isnull(), yticklabels=False)
plt.figure(figsize=(10,5))
plt.title('ACCIDENTS COUNTPLOT PER BOROUGH')
sns.barplot(x=df.groupby('borough').size().index,
            y=df.groupby('borough').size().values)
accidents_bor_df = df.groupby('borough')[['number of persons injured', 'number of persons killed']].sum()


fig, ax = plt.subplots(1,2,figsize=(14,5))
plt.suptitle('DISTRIBUTION PER NUMBER OF INJURED AND KILLED')

ax[1].set_xticklabels(labels=accidents_bor_df.index,rotation=30)
ax[0].set_xticklabels(labels=accidents_bor_df.index,rotation=30)


sns.barplot(accidents_bor_df.index, accidents_bor_df['number of persons injured'], ax=ax[0])
sns.barplot(accidents_bor_df.index, accidents_bor_df['number of persons killed'], ax=ax[1], palette='deep')


fig, ax = plt.subplots(1,2, figsize=(14,5))
plt.suptitle('DISTRIBUTION BY PERCENTAGE')

ax[1].set_xticklabels(labels=accidents_bor_df.index,rotation=30)
ax[0].set_xticklabels(labels=accidents_bor_df.index,rotation=30)
ax[0].set_title('INJURED PERCENTAGE', fontsize=12)
ax[1].set_title('KILLED PERCENTAGE', fontsize=12)


sns.barplot((accidents_bor_df['number of persons injured'] / df.groupby('borough').size() *100).index, 
           (accidents_bor_df['number of persons injured'] / df.groupby('borough').size() *100).values, ax=ax[0], palette='viridis')

sns.barplot((accidents_bor_df['number of persons killed'] / df.groupby('borough').size() *100).index, 
           (accidents_bor_df['number of persons killed'] / df.groupby('borough').size() *100).values, ax=ax[1], palette='magma')
print('MEAN INJURED: ',(accidents_bor_df['number of persons injured'] / df.groupby('borough').size() *100).values.mean())
print('MEAN KILLED: ',(accidents_bor_df['number of persons killed'] / df.groupby('borough').size() *100).values.mean())
datewise = df.groupby(['accident date', 'borough'])[[i for i in df.columns if 'number' in i]].sum()
fig = make_subplots(rows=2,cols=1, 
                    subplot_titles=('NUMBER OF INJURED PER DAY', 'NUMBER OF KILLED PER DAY'))
cols = ['QUEENS', 'BROOKLYN', 'MANHATTAN', 'BRONX', 'STATEN ISLAND']
feat  = [i for i in df.columns if 'number' in i] + ['accident date']

for i, bor in enumerate(cols):
    data_per_bor = df[df['borough']== bor][feat]
    data_per_bor = data_per_bor.groupby('accident date').sum()
    
    fig.add_trace(go.Scatter(x=data_per_bor.index, y=data_per_bor['number of persons injured'], name=bor), row=1,col=1)
    fig.add_trace(go.Scatter(x=data_per_bor.index, y=data_per_bor['number of persons killed'], name=bor), row=2, col=1)

fig.update_layout(template='plotly_dark', width=1000, height=800)
fig.show()
fig = make_subplots(rows=2,cols=1, 
                    subplot_titles=('NUMBER OF INJURED PER DAY', 'NUMBER OF KILLED PER DAY'))
cols = ['QUEENS', 'BROOKLYN', 'MANHATTAN', 'BRONX', 'STATEN ISLAND']
feat  = [i for i in df.columns if 'number' in i] + ['accident date']

for i, bor in enumerate(cols):
    data_per_bor = df[df['borough']== bor][feat]
    data_per_bor = data_per_bor.groupby('accident date').sum()[-365:]
    
    fig.add_trace(go.Scatter(x=data_per_bor.index, y=data_per_bor['number of persons injured'], name=bor), row=1,col=1)
    fig.add_trace(go.Scatter(x=data_per_bor.index, y=data_per_bor['number of persons killed'], name=bor), row=2, col=1)

fig.update_layout(template='plotly_dark', width=1000, height=800)
fig.show()
df.groupby('accident date').size().mean()
weekwise = df.copy()

def week_of_month(tgtdate):

    days_this_month = calendar.mdays[tgtdate.month]
    for i in range(1, days_this_month):
        d = datetime.datetime(tgtdate.year, tgtdate.month, i)
        if d.day - d.weekday() > 0:
            startdate = d
            break
    # now we canuse the modulo 7 appraoch
    return (tgtdate - startdate).days //7 + 1

weekwise['weekofmonth'] = weekwise['accident date'].apply(lambda d: (d.day-1) // 7 + 1)
weekwise['weekofyear'] = weekwise['accident date'].dt.weekofyear
weekwise['month'] = weekwise['accident date'].dt.month
weekwise['year'] = weekwise['accident date'].dt.year
weekwise_month = weekwise.groupby('weekofmonth')[[i for i in weekwise.columns if 'number' in i]].sum()

fig,ax = plt.subplots(1,2,figsize=(14,5))
plt.suptitle('COUNTPLOT OF INJURED AND KILLED BY WEEK OF MONTH', x=0.5, y=1.02, fontsize=20)
ax[0].set_title('INJURED', fontsize=14)
ax[1].set_title('KILLED', fontsize=14)


sns.barplot(x=weekwise_month['number of persons injured'].index ,y=weekwise_month['number of persons injured'], ax=ax[0], palette='tab20b')
sns.barplot(x=weekwise_month['number of persons killed'].index ,y=weekwise_month['number of persons killed'], ax=ax[1], palette='magma')
weekwise_year = weekwise.groupby('weekofyear')[[i for i in weekwise.columns if 'number' in i]].sum()

fig,ax = plt.subplots(1,2,figsize=(14,5))
ax[1].set_xticklabels(labels=accidents_bor_df.index,rotation=90)
ax[0].set_xticklabels(labels=accidents_bor_df.index,rotation=90)

plt.suptitle('COUNTPLOT OF INJURED AND KILLED BY WEEK OF MONTH', x=0.5, y=1.02, fontsize=20)
ax[0].set_title('INJURED', fontsize=14)
ax[1].set_title('KILLED', fontsize=14)


sns.barplot(x=weekwise_year['number of persons injured'].index ,y=weekwise_year['number of persons injured'], ax=ax[0], palette='tab20b')
sns.barplot(x=weekwise_year['number of persons killed'].index ,y=weekwise_year['number of persons killed'], ax=ax[1], palette='magma')
by_month = weekwise.groupby('month')[[i for i in weekwise.columns if 'number' in i]].sum()

fig,ax = plt.subplots(1,2,figsize=(14,5))
plt.suptitle('COUNTPLOT OF INJURED AND KILLED BY MONTH', x=0.5, y=1.02, fontsize=20)
ax[0].set_title('INJURED', fontsize=14)
ax[1].set_title('KILLED', fontsize=14)


sns.barplot(x=by_month['number of persons injured'].index ,y=by_month['number of persons injured'], ax=ax[0], palette='tab20')
sns.barplot(x=by_month['number of persons killed'].index ,y=by_month['number of persons killed'], ax=ax[1], palette='viridis')
by_year = weekwise.groupby('year')[[i for i in weekwise.columns if 'number' in i]].sum()

fig,ax = plt.subplots(1,2,figsize=(14,5))
plt.suptitle('COUNTPLOT OF INJURED AND KILLED BY YEAR', x=0.5, y=1.02, fontsize=20)
ax[0].set_title('INJURED', fontsize=14)
ax[1].set_title('KILLED', fontsize=14)


sns.barplot(x=by_year['number of persons injured'].index ,y=by_year['number of persons injured'], ax=ax[0], palette='spring')
sns.barplot(x=by_year['number of persons killed'].index ,y=by_year['number of persons killed'], ax=ax[1], palette='coolwarm')
per_day_val = round(df.shape[0]/df.groupby('accident date')['number of persons injured'].count().shape[0],2)
per_week_val = round(per_day_val * 7, 2)
per_month_val = round(per_day_val * 30, 2)
per_year_val = per_month_val * 12
per_hour_val = (per_day_val / 24)
per_5mins_val = (per_day_val / 24) /60  * 5

index = ['5mins', 'Hour', 'Day', 'Week', 'Month', 'Year']
data = [per_5mins_val, per_hour_val, per_day_val, per_week_val, per_month_val, per_year_val]
pd.DataFrame(index=index, data=data, columns=['Value']).T
gr_injured = df[[i for i in df.columns for c in ['pedestrians injured', 'cyclist injured', 'motorist injured'] if c in i]].sum()
gr_killed = df[[i for i in df.columns for c in ['pedestrians killed', 'cyclist killed', 'motorist killed'] if c in i]].sum()
gr_injured.index = ['Pedestrian', 'Cyclist', 'Motorist']
gr_killed.index = ['Pedestrian', 'Cyclist', 'Motorist']


fig, ax = plt.subplots(1,2,figsize=(14,5))
plt.suptitle('COUNTPLOT OF KILLED AND INJURED PER ACCIDENT TYPE', fontsize=20, x=0.5,y=1.02)
ax[0].set_title('INJURED', fontsize=14)
ax[1].set_title('KILLED', fontsize=14)

sns.barplot(gr_injured.index, gr_injured.values, ax=ax[0], palette='Greens')
sns.barplot(gr_killed.index, gr_killed.values, ax=ax[1], palette='Reds')



fig = make_subplots(rows=3,cols=1,
                    subplot_titles=('PEDESTRIAN', 'CYCLIST', 'MOTORIST'))

feat_in  = ['number of pedestrians injured',
         'number of cyclist injured',
         'number of motorist injured']

feat_killed = ['number of pedestrians killed', 
         'number of cyclist killed', 
         'number of motorist killed']

for i, atype in enumerate(feat_in):
    data_per_acc = df.groupby('accident date')[atype].sum()
    data_per_acc1 = df.groupby('accident date')[feat_killed[i]].sum()
    
    fig.add_trace(go.Scatter(x=data_per_acc.index, y=data_per_acc.values, name='Injured'), row=i+1,col=1)
    fig.add_trace(go.Scatter(x=data_per_acc1.index, y=data_per_acc1.values, name='Killed'), row=i+1, col=1)

fig.update_layout(title='NUMBER OF KILLED AND INJURED PER ACCIDENT TYPE',template='plotly_dark', width=1000, height=1100)
fig.show()
contri_df = df.groupby('contributing factor vehicle 1').size().sort_values(ascending=False)

plt.figure(figsize=(10,15))
plt.title('CF VEHICLE 1', fontsize=20)

sns.barplot(y = contri_df.index, x = contri_df.values)
contri_df = df.groupby('contributing factor vehicle 2').size().sort_values(ascending=False)

plt.figure(figsize=(10,15))
plt.title('CF VEHICLE 2', fontsize=20)

sns.barplot(y = contri_df.index, x = contri_df.values)