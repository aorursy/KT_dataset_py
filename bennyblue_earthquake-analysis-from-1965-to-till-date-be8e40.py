#importing all important libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import datetime as datetime

from mpl_toolkits.basemap import Basemap
#Reading in data and looking at the number of rows and columns using shape 

earthquake = pd.read_csv('../input/database.csv')

print (earthquake.shape)

print (earthquake.columns)

type (earthquake)
#Viewing head and tail of the dataset

earthquake.head(2)


eq_imp_col = earthquake[['Date', 'Latitude', 'Longitude', 'Type', 'Depth', 'Magnitude','Magnitude Source']]

eq_imp_col.head(2)
import calendar

eq_imp_col['Date'] = pd.to_datetime(eq_imp_col['Date'])
Final_eq = eq_imp_col.set_index(['Date'])

Final_eq.head(2)
plt.hist(Final_eq["Magnitude"], bins=20)

plt.xlabel("Magnitude")

plt.ylabel("Number of Earthquake")

plt.title("Bar chart")#making date as index

plt.scatter(Final_eq['Magnitude'], Final_eq['Depth'])

plt.xlabel("Magnitude")

plt.ylabel("Depth of Earthquake")

plt.title("Scatter Plot")
def f(x):

     return pd.Series(dict(Number_Earthquakes = x['Type'].count(), 

                        ))
yearly_eq = Final_eq.groupby(Final_eq.index.year).apply(f)

yearly_eq.head()
yearly_plot = yearly_eq['Number_Earthquakes'].plot(kind='line')
monthly_eq = Final_eq.groupby(Final_eq.index.month).apply(f)



monthly_eq['Month'] = ['Jan','Feb', 'March', 'April', 'May', 'June', 'July', 'August','September', 

                       'October', 'November', 'December']

monthly_eqyearly_plot = yearly_eq['Number_Earthquakes'].plot(kind='bar')
monthly_plot = monthly_eq['Number_Earthquakes'].plot(kind='line')

plt.xticks( np.arange(13), calendar.month_name[0:13], rotation=90 )

plt.xlabel('Month')

plt.ylabel('Number of Earthquake')

plt.title(' Earthquake on a Monthly Basis')
monthly_plot = monthly_eq['Number_Earthquakes'].plot(kind='bar')

plt.xticks( np.arange(13), calendar.month_name[1:13], rotation=90 )

plt.xlabel('Month')

plt.ylabel('Number of Earthquake')

plt.title(' Earthquake on a Monthly Basis')
def f(x):

     return pd.Series(dict(Mag_Source =x['Magnitude Source'].min(), 

                           Mgt_min= x['Magnitude'].min(), Mgt_max=x['Magnitude'].max()))
Eq_count = Final_eq.groupby('Type').apply(f)

Eq_count
def f(x):

     return pd.Series(dict(Magnitude =x['Magnitude'].count(),))
Eq_count1 = Final_eq.groupby('Type').apply(f)

Eq_count1
def f(x):

     return pd.Series(dict(Type =x['Type'].min(), 

                           Mgt_min= x['Magnitude'].min(), Mgt_max=x['Magnitude'].max()))
Eq_count2 = Final_eq.groupby('Magnitude Source').apply(f)

Eq_count2.head()
#Mercator projection to understand areas which have experienced Earthquake since 1965

m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

longitudes = eq_imp_col["Longitude"].tolist()

latitudes = eq_imp_col["Latitude"].tolist()

x,y= m(longitudes,latitudes)

plt.figure(figsize =(10,10))

m.plot(x,y, "x", color = 'black')

m.drawcoastlines()

m.fillcontinents(color='yellow',lake_color='aqua')

m.drawmapboundary(fill_color='blue')

plt.title("Earthquake Observed since 1965 in the world")

plt.show()