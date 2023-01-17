# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pas = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_passenger_air_service_2.csv')

car = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_air_service_CARGO_AND_PARCELS.csv')
pas.head()
pas.sort_values(by='Whole year', ascending=False)
car.head()
pas['Whole Year'] = pas.iloc[:, 1:-2].sum(axis=1)
# In 2019

g=pas[pas.Year == 2019].sort_values(by='Whole year', ascending=False)[:10].plot.bar(x='Airport name', y='Whole year', rot=45, figsize=(15,5))
airports = pas[pas.Year == 2019].sort_values(by='Whole year', ascending=False)[:10][['Airport name', 'Airport coordinates']]

airports = airports.drop_duplicates()

airports = airports.reset_index()
airports
# Keep one Airpor in Moscow

airports = airports.drop(airports.index[[1,2]])
# Remove parenthis



airports['Airport name'] = airports['Airport name'].str.split('(').str[0]
# Coordonates

airports['x'] = airports['Airport coordinates'].str.split('\'').str[1]

airports['y'] = airports['Airport coordinates'].str.split('\'').str[3]
# Top 10 most used airports



from mpl_toolkits.basemap import Basemap



fig = plt.figure(figsize=(12, 10))



m = Basemap(llcrnrlon=10.,llcrnrlat=40.,urcrnrlon=200.,urcrnrlat=80.,\

            rsphere=(6378137.00,6356752.3142),\

            resolution='l',projection='merc')



for i, row in airports.iterrows():

    x, y= m(float(row['x']),float(row['y']))

    plt.plot(x, y, 'ok', markersize=5, color="red")

    plt.text(x, y, row['Airport name'], fontsize=10)

    

m.shadedrelief()

m.drawcountries(color="red")
year_evo = pas.groupby('Year')['Whole Year'].mean().reset_index()
# In 2019

year_evo.plot.bar(x='Year', y='Whole Year', rot=45, figsize=(15,5))
# 2014 : Olympics in Sotchi

# 2018 : FIFA World Cup
evo = pas.iloc[:,1:14]
months = evo.columns[1:].tolist()
# Transform dataframe to Year, month, value

evo = evo.melt(id_vars =  'Year', value_vars = months)

evo.columns = ['Year', 'Month', 'Passengers']
evo = evo.groupby(['Year', 'Month']).sum().reset_index()
evo.head()
plt.figure(figsize = (20,8))



sns.set(style="darkgrid")

sns.lineplot( x = range(0,len(evo)), y = 'Passengers', data = evo)
months_evo = evo[['Month', 'Passengers']]

months_evo = months_evo.groupby('Month').mean().reset_index()

months_evo['Month'] = pd.Categorical(months_evo['Month'], categories=months, ordered=True)

months_evo = months_evo.sort_values(by='Month')
plt.figure(figsize = (15,8))



g=sns.barplot( x = 'Month', y = 'Passengers', data = months_evo)
# Less airplanes in December / Peak in summer : July, August
august = evo[(evo.Month == 'August') & (evo.Year != 2020)]
sns.set(style="darkgrid")

sns.lmplot( x = 'Year', y = 'Passengers', data = august, height=8)
X = august['Year'].values.reshape(-1, 1)  # values converts it into a numpy array

Y = august['Passengers'].values.reshape(-1, 1)  # values converts it into a numpy array
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X,Y)
X_pred = np.array(2020).reshape(-1, 1)
# Predict 2020

Y_pred = model.predict(X_pred)
print("Peak traffic in 2020 if no COVID : %d passengers" %Y_pred[0][0] )
august.loc[-1] = [2020, 'August', Y_pred[0][0]]
plt.figure(figsize = (15,8))



sns.set(style="darkgrid")

sns.lineplot( x = 'Year', y = 'Passengers', data = august)