# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')



pd.options.display.max_columns = 50

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read the input data



AIR_VISIT_DATA = pd.read_csv('../input/air_visit_data.csv')

AIR_STORE_INFO = pd.read_csv('../input/air_store_info.csv')

HPG_STORE_INFO = pd.read_csv('../input/hpg_store_info.csv')

AIR_RESERVE = pd.read_csv('../input/air_reserve.csv')

HPG_RESERVE = pd.read_csv('../input/hpg_reserve.csv')

STORE_ID_RELATION = pd.read_csv('../input/store_id_relation.csv')

SAMPLE_SUBMISSION = pd.read_csv('../input/sample_submission.csv')

DATE_INFO = pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
AIR_VISIT_DATA.describe()

AIR_VISIT_DATA.head()
AIR_RESERVE.describe()

AIR_RESERVE.head()
HPG_RESERVE.describe()

HPG_RESERVE.head()
AIR_STORE_INFO.describe()

AIR_STORE_INFO.head()
HPG_STORE_INFO.describe()

HPG_STORE_INFO.head()
DATE_INFO.describe()

DATE_INFO.head()
STORE_ID_RELATION.describe()

STORE_ID_RELATION.head()
SAMPLE_SUBMISSION.describe()

SAMPLE_SUBMISSION.head()
data1 = sns.heatmap(AIR_VISIT_DATA.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data1.set_title('AIR_VISIT_DATA')
data2 = sns.heatmap(AIR_STORE_INFO.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data2.set_title('AIR_STORE_INFO')
data3 = sns.heatmap(HPG_STORE_INFO.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data3.set_title('HPG_STORE_INFO')
data4 = sns.heatmap(AIR_RESERVE.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data4.set_title('AIR_RESERVE')
data5 = sns.heatmap(HPG_RESERVE.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data5.set_title('HPG_RESERVE')
data6 = sns.heatmap(STORE_ID_RELATION.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data6.set_title('STORE_ID_RELATION')
data7 = sns.heatmap(SAMPLE_SUBMISSION.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data7.set_title('SAMPLE_SUBMISSION')
data8 = sns.heatmap(DATE_INFO.isnull(),yticklabels=False,cbar=False,cmap='viridis')

data8.set_title('DATE_INFO')
a = AIR_VISIT_DATA.groupby(AIR_VISIT_DATA['visit_date'])['visitors'].sum()

plt.figure(figsize=(15,7))

plt.plot(a.index, a)



plt.ylabel("Number of Visitors",fontsize= 20)

plt.legend()





AIR_VISIT_DATA['visit_date'] = pd.to_datetime(AIR_VISIT_DATA['visit_date'])

AIR_VISIT_DATA['day_of_week'] = AIR_VISIT_DATA['visit_date'].dt.dayofweek

b = AIR_VISIT_DATA.groupby(['day_of_week'])['visitors'].median()





AIR_VISIT_DATA['month'] = AIR_VISIT_DATA['visit_date'].dt.month

c = AIR_VISIT_DATA.groupby(['month'])['visitors'].median()



fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,figsize=(14,4))

sns.barplot(x=b.index, y=b, ax=ax1)

sns.barplot(x=c.index, y=c, ax=ax2)

plt.figure(figsize=(15,7))

sns.distplot(np.log(AIR_VISIT_DATA['visitors']))

plt.legend()
df = AIR_VISIT_DATA[((AIR_VISIT_DATA['visit_date'] > '2016-04-15') & (AIR_VISIT_DATA['visit_date'] < '2016-06-15'))]

df1 = df.groupby(df['visit_date'])['visitors'].sum()

plt.figure(figsize=(12,4))

plt.plot(df1.index,df1)

plt.xlabel("Date")

plt.ylabel("Visitors")

plt.legend()
AIR_RESERVE['visit_datetime'] = pd.to_datetime(AIR_RESERVE['visit_datetime'])

AIR_RESERVE['reserve_datetime'] = pd.to_datetime(AIR_RESERVE['reserve_datetime'])

AIR_RESERVE['visit_hour'] = AIR_RESERVE['visit_datetime'].dt.hour

AIR_RESERVE['visit_date'] = AIR_RESERVE['visit_datetime'].dt.date



air_reserve_date = AIR_RESERVE.groupby(['visit_date'])['reserve_visitors'].sum()

plt.figure(figsize=(12,4))

plt.plot(air_reserve_date.index,air_reserve_date,lw = 2)

plt.xlabel("Visit_Date")

plt.ylabel("Visitors")

plt.legend()



air_reserve_hour = AIR_RESERVE.groupby(['visit_hour'])['reserve_visitors'].sum()

plt.figure(figsize=(12,4))

plt.bar(air_reserve_hour.index,air_reserve_hour)

plt.xlabel("Visit_hour")

plt.ylabel("Visitors")

plt.legend()
AIR_RESERVE['delta'] = AIR_RESERVE['visit_datetime']-AIR_RESERVE['reserve_datetime']

AIR_RESERVE['delta1'] = AIR_RESERVE['delta'].apply(lambda x: (x.seconds/3600))

d = AIR_RESERVE.groupby(AIR_RESERVE['delta1'])['reserve_visitors'].sum().reset_index()
HPG_RESERVE['visit_datetime'] = pd.to_datetime(HPG_RESERVE['visit_datetime'])

HPG_RESERVE['reserve_datetime'] = pd.to_datetime(HPG_RESERVE['reserve_datetime'])

HPG_RESERVE['visit_hour'] = HPG_RESERVE['visit_datetime'].dt.hour

HPG_RESERVE['visit_date'] = HPG_RESERVE['visit_datetime'].dt.date



hpg_reserve_date = HPG_RESERVE.groupby(['visit_date'])['reserve_visitors'].sum()

plt.figure(figsize=(12,4))

plt.plot(hpg_reserve_date.index,hpg_reserve_date,lw = 2)

plt.xlabel("Visit_Date")

plt.ylabel("Visitors")

plt.legend()



hpg_reserve_hour = HPG_RESERVE.groupby(['visit_hour'])['reserve_visitors'].sum()

plt.figure(figsize=(12,4))

plt.bar(hpg_reserve_hour.index,hpg_reserve_hour)

plt.xlabel("Visit_hour")

plt.ylabel("Visitors")

plt.legend()
import folium

from folium import plugins
m = folium.Map([AIR_STORE_INFO['latitude'].min(), AIR_STORE_INFO['longitude'].max()], zoom_start=4)

m
pd.options.display.max_rows = 4000

pd.options.display.max_seq_items = 2000



air_store_genre = AIR_STORE_INFO.groupby(AIR_STORE_INFO['air_genre_name'])['air_store_id'].count().reset_index()

air_store_genre = air_store_genre.sort_values(['air_store_id'],ascending=False)



plt.figure(figsize=(8,5))

sns.barplot(x='air_store_id', y='air_genre_name', data = air_store_genre)

plt.xlabel("Number of Restaurants")

plt.ylabel("Type of Cuisine")

plt.legend()





air_area = AIR_STORE_INFO.groupby(AIR_STORE_INFO['air_area_name'])['air_store_id'].count().reset_index()

air_area = air_area.sort_values(['air_store_id'],ascending=False)

air_area = air_area.head(15)





plt.figure(figsize=(8,5))

sns.barplot(x = 'air_store_id',y = air_area['air_area_name'],data = air_area)

plt.xlabel("Number of Restaurants")

plt.ylabel("Area")

plt.legend()
HPG_STORE_INFO.head()
hpg_store_genre = HPG_STORE_INFO.groupby(HPG_STORE_INFO['hpg_genre_name'])['hpg_store_id'].count().reset_index()

hpg_store_genre = hpg_store_genre.sort_values(['hpg_store_id'],ascending=False)



fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(12,10))

plt1 = sns.barplot(x= 'hpg_store_id', y='hpg_genre_name',data = hpg_store_genre,ax = ax1)

plt1.set(xlabel="Number of HPG Restaurants",ylabel='HPG Genre Name')



hpg_store_area = HPG_STORE_INFO.groupby(HPG_STORE_INFO['hpg_area_name'])['hpg_store_id'].count().reset_index()

hpg_store_area = hpg_store_area.sort_values(['hpg_store_id'],ascending=False)

hpg_store_area1 = hpg_store_area.head(15)



plt2 = sns.barplot(x= 'hpg_store_id', y='hpg_area_name',data = hpg_store_area1, ax = ax2)

plt2.set(xlabel="Number of HPG Restaurants",ylabel="Area Name")

plt.tight_layout()
DATE_INFO['visit_date'] = pd.to_datetime(DATE_INFO['visit_date'])

holidays16 = DATE_INFO[((DATE_INFO['visit_date'] >'2016-04-15') & (DATE_INFO['visit_date'] < '2016-06-01'))]

holidays17 = DATE_INFO[((DATE_INFO['visit_date'] >'2017-04-15') & (DATE_INFO['visit_date'] < '2017-06-01'))]




sns.countplot(x="holiday_flg",data = DATE_INFO)



fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(10,4))

plt2 = sns.stripplot(x='visit_date',y='holiday_flg',data=holidays16, ax=ax1)

plt2.set_xticks([])







plt3 = sns.stripplot(x='visit_date',y='holiday_flg',data=holidays17, ax=ax2)

plt3.set(xticks=[])



plt.tight_layout()
air_visit = AIR_VISIT_DATA.groupby(AIR_VISIT_DATA['visit_date'])['visitors'].sum().reset_index()

air_visit['visit_date'] = pd.to_datetime(air_visit['visit_date'])

air_visit['year'] = air_visit['visit_date'].dt.year

air_visit['month'] = air_visit['visit_date'].dt.month

air_visit['day'] = air_visit['visit_date'].dt.day
air_visit[['visit_date','year']].set_index('visit_date').plot()
SAMPLE_SUBMISSION['date'] = SAMPLE_SUBMISSION['id'][0].split('_')[2]

SAMPLE_SUBMISSION['date'] = pd.to_datetime(SAMPLE_SUBMISSION['date'])

SAMPLE_SUBMISSION['year'] = SAMPLE_SUBMISSION['date'].dt.year
test = SAMPLE_SUBMISSION.groupby(SAMPLE_SUBMISSION['date'])['year'].max().reset_index()
ab = pd.merge(AIR_VISIT_DATA,AIR_STORE_INFO,on='air_store_id')

ab1 = ab.groupby(['visit_date','air_genre_name'])['visitors'].mean().reset_index()

ab13 = ab1.pivot_table(values='visitors',index='visit_date',columns='air_genre_name')

ab13.plot(subplots=True,figsize=(12,60))