# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns


img=np.array(Image.open('../input/indiarain/Annual-mean-rainfall-map-of-India.png'))

fig=plt.figure(figsize=(10,10))

plt.imshow(img,interpolation='bilinear')

plt.axis('off')

plt.ioff()

plt.show()
India = pd.read_csv("../input/rainfall-in-india/rainfall in india 1901-2015.csv",sep=",")
India.head()
print('Rows     :',India.shape[0])

print('Columns  :',India.shape[1])

print('\nFeatures :\n     :',India.columns.tolist())

print('\nMissing values    :',India.isnull().values.sum())

print('\nUnique values :  \n',India.nunique())
total = India.isnull().sum().sort_values(ascending=False)

percent = (India.isnull().sum()/India.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()
India.info()
India['JAN'].fillna((India['JAN'].mean()), inplace=True)

India['FEB'].fillna((India['FEB'].mean()), inplace=True)

India['MAR'].fillna((India['MAR'].mean()), inplace=True)

India['APR'].fillna((India['APR'].mean()), inplace=True)

India['MAY'].fillna((India['MAY'].mean()), inplace=True)

India['JUN'].fillna((India['JUN'].mean()), inplace=True)

India['JUL'].fillna((India['JUL'].mean()), inplace=True)

India['AUG'].fillna((India['AUG'].mean()), inplace=True)

India['SEP'].fillna((India['SEP'].mean()), inplace=True)

India['OCT'].fillna((India['OCT'].mean()), inplace=True)

India['NOV'].fillna((India['NOV'].mean()), inplace=True)

India['DEC'].fillna((India['DEC'].mean()), inplace=True)

India['ANNUAL'].fillna((India['ANNUAL'].mean()), inplace=True)

India['Jan-Feb'].fillna((India['Jan-Feb'].mean()), inplace=True)

India['Mar-May'].fillna((India['Mar-May'].mean()), inplace=True)

India['Jun-Sep'].fillna((India['Jun-Sep'].mean()), inplace=True)

India['Oct-Dec'].fillna((India['Oct-Dec'].mean()), inplace=True)
India.describe().T
ax=India.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(600,2200),color='b',marker='o',linestyle='-',linewidth=2,figsize=(12,8));

India['MA10'] = India.groupby('YEAR').mean()['ANNUAL'].rolling(10).mean()

India.MA10.plot(color='r',linewidth=4)

plt.xlabel('Year',fontsize=20)

plt.ylabel('Annual Rainfall (in mm)',fontsize=20)

plt.title('Annual Rainfall in India from Year 1901 to 2015',fontsize=25)

ax.tick_params(labelsize=15)

plt.grid()

plt.ioff()
India[['YEAR','Jan-Feb', 'Mar-May',

       'Jun-Sep', 'Oct-Dec']].groupby("YEAR").mean().plot(figsize=(13,8));

plt.xlabel('Year',fontsize=20)

plt.ylabel('Seasonal Rainfall (in mm)',fontsize=20)

plt.title('Seasonal Rainfall from Year 1901 to 2015',fontsize=25)

ax.tick_params(labelsize=15)

plt.grid()

plt.ioff()
India[['SUBDIVISION', 'Jan-Feb', 'Mar-May',

       'Jun-Sep', 'Oct-Dec']].groupby("SUBDIVISION").mean().sort_values('Jun-Sep').plot.bar(width=0.5,edgecolor='k',align='center',stacked=True,figsize=(16,8));

plt.xlabel('Subdivision',fontsize=30)

plt.ylabel('Rainfall (in mm)',fontsize=20)

plt.title('Rainfall in Subdivisions of India',fontsize=25)

ax.tick_params(labelsize=15)

plt.grid()

plt.ioff()
drop_col = ['ANNUAL','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']



fig, ax = plt.subplots()



(India.groupby(by='YEAR')

 .mean()

 .drop(drop_col, axis=1)

 .T

 .plot(alpha=0.1, figsize=(12, 6), legend=False, fontsize=12, ax=ax)

)

ax.set_xlabel('Months', fontsize=12)

ax.set_ylabel('Rainfall (in mm)', fontsize=12)

plt.grid()

plt.ioff()


plt.rcParams['figure.figsize']=(23,10)

ax = sns.boxplot(x="SUBDIVISION", y="ANNUAL", data=India,width=0.8,linewidth=3)

ax.set_xlabel('Subdivision',fontsize=30)

ax.set_ylabel('Annual Rainfall (in mm)',fontsize=30)

plt.title('Annual Rainfall in Subdivisions of India',fontsize=40)

ax.tick_params(axis='x',labelsize=20,rotation=90)

ax.tick_params(axis='y',labelsize=20,rotation=0)

plt.grid()

plt.ioff()
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

India.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'].plot('bar', color='b',width=0.65,linewidth=3,edgecolor='k',align='center',title='Subdivision wise Average Annual Rainfall', fontsize=20)

plt.xticks(rotation = 90)

plt.ylabel('Average Annual Rainfall (in mm)')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)

#print(India.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[0,1,2]])

#print(India.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[33,34,35]])

ax=India[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2,figsize=(16,8))

plt.xlabel('Month',fontsize=30)

plt.ylabel('Monthly Rainfall (in mm)',fontsize=20)

plt.title('Monthly Rainfall in India',fontsize=25)

ax.tick_params(labelsize=20)

plt.grid()

plt.ioff()
India[['AUG']].mean()
#India1=India['JAN','FEB','ANNUAL']

fig=plt.gcf()

fig.set_size_inches(15,15)

fig=sns.heatmap(India.corr(),annot=True,cmap='summer',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
img=np.array(Image.open('../input/keraladistricts/kerala-map_1.jpg'))

fig=plt.figure(figsize=(10,10))

plt.imshow(img,interpolation='bilinear')

plt.axis('off')

plt.ioff()

plt.show()
Kerala =India[India.SUBDIVISION == 'KERALA']

#Kerala
ax=Kerala.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(1000,5000),color='b',marker='o',linestyle='-',linewidth=2,figsize=(12,8));

#Kerala['MA10'] = Kerala.groupby('YEAR').mean()['ANNUAL'].rolling(10).mean()

#Kerala.MA10.plot(color='r',linewidth=4)

plt.xlabel('Year',fontsize=20)

plt.ylabel('Kerala Annual Rainfall (in mm)',fontsize=20)

plt.title('Kerala Annual Rainfall from Year 1901 to 2015',fontsize=25)

ax.tick_params(labelsize=15)

plt.grid()

plt.ioff()
print('Average annual rainfall received by Kerala=',int(Kerala['ANNUAL'].mean()),'mm')
print('Kerala received 4257.8 mm of rain in the year 1961')

a=Kerala[Kerala['YEAR']==1961]

a
print('Kerala received 4226.4 mm of rain in the year 1924')

b=Kerala[Kerala['YEAR']==1924]

b
Dist = pd.read_csv("../input/rainfall-in-india/district wise rainfall normal.csv",sep=",")
Dist.head()
KDist=Dist[Dist.STATE_UT_NAME == 'KERALA']

k=KDist.sort_values(by=['ANNUAL'])

ax=k.plot.bar(x='DISTRICT',y='ANNUAL',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

plt.xlabel('District',fontsize=30)

plt.ylabel('Annual Rainfall (in mm)',fontsize=20)

plt.title('Rainfall in Districts of Kerala',fontsize=25)

ax.tick_params(labelsize=20)

plt.grid()

plt.ioff()
Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().head(5)
ax=Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

#ax=k.plot.bar(x='DISTRICT',y='ANNUAL',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

plt.xlabel('District',fontsize=30)

plt.ylabel('Annual Rainfall (in mm)',fontsize=20)

plt.title('Districts with Minumum Rainfall in India',fontsize=25)

ax.tick_params(labelsize=20)

plt.grid()

plt.ioff()
Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().tail(5)
ax=Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().tail(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

#ax=k.plot.bar(x='DISTRICT',y='ANNUAL',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

plt.xlabel('District',fontsize=30)

plt.ylabel('Annual Rainfall (in mm)',fontsize=20)

plt.title('Districts with Maximum Rainfall in India',fontsize=25)

ax.tick_params(labelsize=20)

plt.grid()

plt.ioff()
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

m=Basemap(projection='mill',llcrnrlat=0,urcrnrlat=40,llcrnrlon=50,urcrnrlon=100,resolution='c')

m.drawcoastlines()

m.drawcountries()

m.drawstates()

#m.fillcontinents()

m.fillcontinents(color='coral',lake_color='aqua')

#m.drawmapboundary()

m.drawmapboundary(fill_color='aqua')

#m.bluemarble()

#x, y = m(25.989836, 79.450035)

#plt.plot(x, y, 'go', markersize=5)

#plt.text(x, y, ' Trivandrum', fontsize=12);

lat,lon=13.340881,74.742142

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, ' Udupi (4306mm)', fontsize=12);

lat,lon=28.879720,94.796970

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, ' Upper Siang(4402mm)', fontsize=12);

"""lat,lon=25.578773,91.893257

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, 'East Kashi Hills (6166mm)', fontsize=12);

lat,lon=25.389820,92.394913

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, 'Jaintia Hills (6379mm)', fontsize=10);"""

lat,lon=24.987934,93.495293

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, 'Tamenglong (7229mm)', fontsize=12);

lat,lon=34.136389,77.604139

x,y=m(lon,lat)

m.plot(x,y,'ro')

plt.text(x, y, ' Ladakh(94mm)', fontsize=12);

"""lat,lon=25.759859,71.382439

x,y=m(lon,lat)

m.plot(x,y,'ro')

plt.text(x, y, ' Barmer(268mm)', fontsize=12);"""

lat,lon=26.915749,70.908340

x,y=m(lon,lat)

m.plot(x,y,'ro')

plt.text(x, y, ' Jaisalmer(181mm)', fontsize=12);

plt.title('Places with Heavy and Scanty Rainfall in India',fontsize=20)

plt.ioff()

plt.show()
India.groupby("YEAR").mean()['ANNUAL'].sort_values(ascending=False).head(10)
India.groupby("YEAR").mean()['ANNUAL'].sort_values(ascending=False).tail(10)