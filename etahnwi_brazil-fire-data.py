# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import geopandas as gpd

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding = "latin1")

data.head(20)
def explore(data):

    summaryDF = pd.DataFrame(data.dtypes, columns=['dtypes'])

    summaryDF = summaryDF.reset_index()

    summaryDF['Name'] = summaryDF['index']

    summaryDF['Missing'] = data.isnull().sum().values

    summaryDF['Total'] = data.count().values

    summaryDF['MissPerc'] = (summaryDF['Missing']/data.shape[0])*100

    summaryDF['NumUnique'] = data.nunique().values

    summaryDF['UniqueVals'] = [data[col].unique() for col in data.columns]

    

    





    print(summaryDF.head(30))
explore(data)
latitude={'Acre':-9.02,'Alagoas':-9.57,'Amapa':02.05,'Amazonas':-5.00,'Bahia':-12.00,'Ceara':-5.00,

          

          'Distrito Federal':-15.45,'Espirito Santo':-20.00,'Goias':-15.55,'Maranhao':-5.00,'Mato Grosso':-14.00

          

          ,'Minas Gerais':-18.50,'Pará':-3.20,'Paraiba':-7.00,'Pernambuco':-8.00,'Piau':-7.00,'Rio':-22.90,

          

          'Rondonia':-11.00,'Roraima':-2.00,'Santa Catarina':-27.25,'Sao Paulo':-23.32,'Sergipe':-10.30,

         

         'Tocantins':-10.00

         }





longitude={

    'Acre':-70.8120,'Alagoas':-36.7820,'Amapa':-50.50,'Amazonas':-65.00,'Bahia':-42.00,'Ceara':-40.00,

    

    'Distrito Federal':-47.45,'Espirito Santo':-40.45,'Goias':-50.10,'Maranhao':-46.00,'Mato Grosso':-55.00,

    

    'Minas Gerais':-46.00,'Pará':-52.00,'Paraiba':-36.00,'Pernambuco':-37.00,'Piau':-73.00, 'Rio':-43.17,

    

    'Rondonia':-63.00,'Roraima':-61.30,'Santa Catarina':-48.30,'Sao Paulo':-46.37,'Sergipe':-37.30,

    

    'Tocantins':-48.00

}
data['Latitude']=data['state'].map(latitude)

data['Longitude']=data['state'].map(longitude)

data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Longitude, data.Latitude))

data.crs = {'init': 'epsg:4326'}

data.head(50)
ax = data.plot(figsize=(20,8), color='whitesmoke', linestyle=':', edgecolor='black')

data.to_crs(epsg=32630).plot(markersize=1, ax=ax)
data['date'] = data['date'].astype('datetime64[ns]')

data['number'] = data['number'].astype(int)
plt.figure(figsize=(20,10))

sns.countplot(data.state)

plt.xticks(rotation=90)



state_fire = data.groupby(['state'])['number'].sum().reset_index(name='counts')



state_fire.sort_values(by='counts',ascending=False)

plt.figure(figsize=(20,5))

sns.barplot(data=state_fire, x="state", y="counts")

plt.ylabel('Count')

plt.xlabel('State')

plt.title('Total Fire Count By State')

plt.xticks(rotation=90)
year_df = data.groupby(['year'])['number'].sum().reset_index(name='sum')

plt.figure(figsize=(20,10))

sns.barplot(data=year_df, x='year', y='sum')

px.line(year_df,x='year',y='sum')



year_df['sum'].describe()
mato_g = data.loc[data['state']=="Mato Grosso"]

mato_g_data = mato_g.groupby(['year'])['number'].sum().reset_index(name='number')



plt.figure(figsize=(10,5))

sns.lineplot(data=mato_g_data, x='year', y='number')



plt.figure(figsize=(20,10))

sns.barplot(data=mato_g_data, x='year',y='number')
month_mont = mato_g.groupby(['month'])['number'].sum().reset_index(name='number')

plt.figure(figsize=(20,10))

sns.barplot(data=month_mont, x='month', y='number')
pct_change = pd.DataFrame(data.groupby(['year'])['number'].sum())

pct_change['pct_change'] = data.groupby(['year'])['number'].sum().pct_change()*100

pct_change