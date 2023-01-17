# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import bq_helper
import seaborn as sns
from bq_helper import BigQueryHelper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="chicago_crime")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
select_query = """SELECT date,district,primary_type,location_description,ward,arrest,domestic,community_area,year,latitude,longitude,location
            FROM `bigquery-public-data.chicago_crime.crime`
            LIMIT 300000"""
crime_data = chicago_crime.query_to_pandas_safe(select_query)

print(crime_data.head())
crime_data.tail()
crime_data.sample(10)
crimeType = crime_data.groupby("primary_type")["district"].count()
print(crimeType)
crimeTypeSort = crime_data.groupby("primary_type")["district"].count().nlargest(20)
print(crimeTypeSort)
plot = crimeTypeSort.plot.pie(figsize=(10, 10))
top10 = crime_data.groupby("primary_type")["district"].count().nlargest(10)

plot = top10.plot.pie(figsize=(6, 6))
crimeTypeYear = crime_data.groupby("year")["primary_type"].count()
print(crimeTypeYear)
crime_data[['date', 'primary_type']]
cd = crime_data
cd['date'] = pd.to_datetime(cd['date'])
cd['year'] = cd['date'].dt.year

cd['primary_type'].value_counts()
cd['primary_type'].value_counts().plot.barh(x='crimes', y='count', figsize=(10, 10))
total = cd['primary_type'].value_counts()[:20]
cd['primary_type'].value_counts()[:10].plot(kind='barh')

first = cd[cd.year == 2001]
print(first.year.count())
last = cd[cd.year ==2018]
#np.sum('year'==2001)
#np.sum('year'== 2017)
firstGraph = first['primary_type'].value_counts()[:20]
first['primary_type'].value_counts()[:20].plot(kind="barh")
lastGraph = last['primary_type'].value_counts()[:20]

ax = lastGraph.plot()
firstGraph.plot(ax=ax)
ax.legend(["2001", "2018"]);

yr2017 = cd[cd.year == 2017]
yr2017Graph = yr2017['primary_type'].value_counts()[:20]
ax = yr2017Graph.plot()
firstGraph.plot(ax=ax)
ax.legend(["2001", "2017"]);
ax = lastGraph.plot()
yr2017Graph.plot(ax=ax, kind="bar")
ax.legend(["2018", "2017"])
twentyEighteen = cd[cd.year == 2018]
#np.sum('year'==2018)
print(twentyEighteen.year.count())
first['primary_type'].value_counts()[:20].plot(kind='barh')
twentyEighteen['primary_type'].value_counts()[:10].plot(kind='barh')
test=twentyEighteen['primary_type'].value_counts()[:10]
fig, ax = plt.subplots(figsize=(20,20))
#ax.set_xticklabels([])
ax.legend_ = None
#draw()
trial = cd.drop(['location_description', 'ward', 'domestic','community_area', 'district', 'longitude', 'arrest', 'latitude', 'location'], axis=1)
trial.groupby(['year','primary_type']).count().unstack().plot(ax=ax)
plt.show()
fig, ax = plt.subplots(figsize=(10,8))
trial = cd.drop(['location_description', 'ward', 'domestic','community_area', 'district', 'longitude', 'arrest', 'latitude', 'location'], axis=1)
trial = trial.groupby(['primary_type'])
theft = trial.get_group('THEFT') 
theft.groupby(['year', 'primary_type']).count().unstack().plot(ax=ax)
plt.show()
fig, ax = plt.subplots(figsize=(10,8))
trial = cd.drop(['location_description', 'ward', 'domestic','community_area', 'district', 'longitude', 'arrest', 'latitude', 'location'], axis=1)
trial = trial.groupby(['primary_type'])
theft = trial.get_group('BATTERY') 
theft.groupby(['year', 'primary_type']).count().unstack().plot(ax=ax)
plt.show()
fig, ax = plt.subplots(figsize=(10,8))
trial = cd.drop(['location_description', 'ward', 'domestic','community_area', 'district', 'longitude', 'arrest', 'latitude', 'location'], axis=1)
trial = trial.groupby(['primary_type'])
theft = trial.get_group('CRIMINAL DAMAGE') 
theft.groupby(['year', 'primary_type']).count().unstack().plot(ax=ax)
plt.show()
cd.describe()
cd.plot(kind='scatter',x='longitude',y='latitude',color='red')
plt.show()
cd['district'].hist()
plt.show()
crimeData = crime_data[crime_data.primary_type == 'CRIMINAL DAMAGE']
crimeData.plot(kind='scatter',x='longitude',y='latitude',color='green')
plt.show()

crimeData["district"].hist()
plt.show()
weaponData = crime_data[crime_data.primary_type == 'WEAPONS VIOLATION']
weaponData["district"].hist()
plt.show()

batteryData = crime_data[crime_data.primary_type == 'BATTERY']
batteryData["district"].hist()
plt.show()
x =first["district"].hist()
plt.show()
y = twentyEighteen['district'].hist()
plt.show()
plt.matshow(cd.corr())
plt.show()
corr = cd.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)