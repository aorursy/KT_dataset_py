# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



data = pd.read_csv('../input/BPD_Arrests.csv')

data.head(3)
# check the dimension

data.shape
# check the data information

data.info()
dt_sex = data.Sex.value_counts()



fig = plt.figure(figsize=(6,4))

dt_sex.plot(kind='bar')



plt.title('Arrest by Gender')

plt.ylabel('Total Cases')

plt.xticks(range(2), ('Male', 'Female'), rotation=40)

plt.show()
dt_race = data.Race.value_counts()



fig = plt.figure(figsize=(6,4))

dt_race.plot(kind='bar')



plt.title('Arrest by Race')

plt.ylabel('Total Cases')

plt.show()
def age_bucket(x):

    if x <= 20 and x >= 0:

        return "less than 20"

    elif x <= 30:

        return "between 20 and 30"

    elif x <= 40:

        return "between 30 and 40"

    elif x <= 50:

        return "between 40 and 50"

    elif x <= 60:

        return "between 50 and 60"    

    elif x > 60:

        return "60 and older "        

    else:

        return "Unknown"



dt_age = data.Age.map(lambda x: age_bucket(x)).value_counts()   



fig = plt.figure(figsize=(6,6))

labels=('Between 20 and 30', 'Between 30 and 40', 'Between 40 and 50', 'Between 50 and 60', 'Equal and Less than 20', 'More than 60', 'Unknown')

plt.pie(dt_age,explode=(0, 0, 0, 0, 0,0.05,0.1), autopct='%1.2f%%', shadow=False, labels=labels, \

       startangle=70,radius=.9 )

plt.title('Arrest by Age Group')

plt.show()
from datetime import datetime

data['ArrestYear'] =data.ArrestDate.map(lambda x: datetime.strptime(x, '%m/%d/%Y').year)

data['ArrestMonth'] =data.ArrestDate.map(lambda x: datetime.strptime(x, '%m/%d/%Y').month)



sns.set()

sns.factorplot(hue='ArrestYear', x='Race', col='Sex', data=data, kind='count',size=3 )
dt = data[(data['Age']!=0) & (data['Age'].notnull()) ]

sns.factorplot(data=data, x="ArrestYear", col="Sex", y='Age', size=4)  
g = sns.FacetGrid(dt, row='ArrestYear', col="Sex", size=4)  

g.map(sns.distplot, "Age")  
dt_district = data[data['District'].notnull()]

sns.factorplot(x='ArrestYear', hue='District', data=dt_district, kind='count', size=6)
from mpl_toolkits.basemap import Basemap

from matplotlib import cm



locs = data['Location 1'][data['Location 1'].notnull()] 

locs_lon=[]

locs_lat=[]



for loc in locs:

    lat, lon = loc[1:-1].split(', ')

    locs_lon.append(float(lon))

    locs_lat.append(float(lat))
west, south, east, north = min(locs_lon), min(locs_lat), max(locs_lon), max(locs_lat)

 

fig = plt.figure(figsize=(9,9))

m = Basemap(projection='gall', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='l')

x, y = m(pd.Series(locs_lon).values, pd.Series(locs_lat).values)

m.hexbin(x, y, gridsize=100, bins='log', cmap=cm.Blues);