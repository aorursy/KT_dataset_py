%matplotlib inline
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as m
import seaborn as sns
m.style.use('ggplot')
import numpy as np
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/earthquake.csv",parse_dates=[['Date', 'Time']])
data.head(10)


data.info()
data['Date_Time'] = pd.to_datetime(data['Date_Time'],errors='coerce')


data['Date_Time'].dtypes
data['Date_Time'].loc[data['Date_Time'].isnull()]
data['Date_Time'].loc[[3378,7512,20650]]
data['Date_Time'][3378] = '1975-02-23'
data['Date_Time'][7512] = '1985-04-28'
data['Date_Time'][20650] = '2011-03-13'
data['Date_Time'].loc[data['Date_Time'].isnull()]
day_of_month_earthquakes = data['Date_Time'].dt.day
day_of_month_earthquakes.head(10)
# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
data['Type'].unique()

Disaster_type= data['Type'].value_counts()
df = pd.DataFrame([Disaster_type]).T
print (df)
df.plot(kind='bar',stacked=True, figsize=(15,8))
data = data[data.Type != 'Nuclear Explosion']
data = data[data.Type != 'Explosion']
data = data[data.Type != 'Rock Burst']

data['Type'].unique()

sns.heatmap(data.isnull(), cbar=False)
data.describe()
data.drop(['Depth Error','Depth Seismic Stations','Magnitude Error','Magnitude Seismic Stations','Azimuthal Gap','Horizontal Distance','Horizontal Error'], axis = 1, inplace = True)
data.head()
sns.heatmap(data.isnull(), cbar=False)
data['Magnitude Type'].isnull().sum()
data['Magnitude Type'].loc[data['Magnitude Type'].isnull()]
data = data.drop(data.index[[6703,7294,7919]])
data['Magnitude Type'].isnull().sum()
data.pivot_table(index = "Type", values = "Magnitude", aggfunc=len)
data.info()
data.describe()
data.fillna(data.mean(), inplace=True)
sns.heatmap(data.isnull(), cbar=False)
var = 'Magnitude'
fig = sns.boxplot(x=var, data=data)

data.head()
from scipy import stats
#Now to find the total number of outliers, we have kept the data within 3 Standard Deviation in the "new" dataset 

outlier_data = data[abs(stats.zscore(data['Magnitude']))>3.0]
outlier_data

outlier_data.shape
print(outlier_data.Magnitude.min())
print(outlier_data.Magnitude.max())
sns.distplot(outlier_data['Magnitude']);
#Magnitude of the Earthquakes throughout the timeline
data['Magnitude'].describe()
#histogram and normal probability plot
sns.distplot(data['Magnitude']);
fig = plt.figure()
res = stats.probplot(data['Magnitude'], plot=plt)
datanew=data
data2=data
data3=data
data4=data
data5=data
datanew['Magnitude'] = np.cbrt(data['Magnitude'])

sns.distplot(datanew['Magnitude']);
fig = plt.figure()
res = stats.probplot(datanew['Magnitude'], plot=plt)
data2['Magnitude'] = np.log(data['Magnitude'])

sns.distplot(data2['Magnitude']);
fig = plt.figure()
res = stats.probplot(data2['Magnitude'], plot=plt)
data3['Magnitude'] = np.exp(data['Magnitude'])

sns.distplot(data3['Magnitude']);
fig = plt.figure()
res = stats.probplot(data3['Magnitude'], plot=plt)
data4['Magnitude'] = np.sqrt(data['Magnitude'])

sns.distplot(data4['Magnitude']);
fig = plt.figure()
res = stats.probplot(data4['Magnitude'], plot=plt)
data.head()
sns.distplot(data['Depth']);
plt.scatter(data["Magnitude"],data["Depth"])
corrmat = data.corr()
k = 2 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Depth')['Depth'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
data = data.set_index(['Date_Time'])
#data.drop(['Date_Time'], axis = 1, inplace = True)
def f(x):
     return pd.Series(dict(Count_of_Disaster = x['Type'].count(),))
    

weekday_count = data.groupby(data.index.weekday).apply(f)
print (len(weekday_count))
weekday_count
yearly_count = data.groupby(data.index.year).apply(f)
print (len(yearly_count))
yearly_count.head()

yearly_plot = yearly_count['Count_of_Disaster'].plot()
monthly_count = data.groupby(data.index.month).apply(f)
print (len(monthly_count))
monthly_count

monthly_count = monthly_count['Count_of_Disaster'].plot()
from mpl_toolkits.basemap import Basemap
m = Basemap(projection='mill',llcrnrlat=-75,urcrnrlat=75, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
longitudes = data["Longitude"].tolist()
latitudes = data["Latitude"].tolist()

x,y = m(longitudes,latitudes)
fig = plt.figure(figsize=(12,10))
plt.title("Worldwide Map for the areas affected by earthquake")
m.plot(x, y, "o", markersize = 3, color = 'red')
m.drawcoastlines()
m.fillcontinents(color='grey',lake_color='blue')
m.drawmapboundary()
m.drawcountries()
plt.show()