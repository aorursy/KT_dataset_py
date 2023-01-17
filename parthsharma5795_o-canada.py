# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('../input/Canada.xlsx',

                     sheet_name='Canada by Citizenship',

                     skiprows = range(20),

                     skipfooter = 2)



# getting the shape of the data

df.shape
df.head(5)

print("The number of nulls in each column are \n", df.isna().sum())
df.iloc[2].nunique()
df = df.drop(['AREA','REG','DEV','Type','Coverage'], axis = 1)



# adding a Total column to add more information

df['Total'] = df.sum(axis = 1)



# let's check the head of the cleaned data

df.head()
df['decade1']=df.iloc[:,4:14].sum(axis=1)

df['decade2']=df.iloc[:,14:24].sum(axis=1)

df['decade3']=df.iloc[:,24:34].sum(axis=1)

df['decade4']=df.iloc[:,34:38].sum(axis=1)

df.head()
df1=df[['AreaName','decade1']].groupby(['AreaName']).sum(axis=1).sum(

    level=['AreaName'])

df2=df[['AreaName','decade2']].groupby(['AreaName']).sum(axis=1).sum(

    level=['AreaName'])

df3=df[['AreaName','decade3']].groupby(['AreaName']).sum(axis=1).sum(

    level=['AreaName'])

df4=df[['AreaName','decade4']].groupby(['AreaName']).sum(axis=1).sum(

    level=['AreaName'])

new_df=pd.merge(df1, df2, how='inner', on = 'AreaName')

new_df=pd.merge(new_df, df3, how='inner', on = 'AreaName')

new_df=pd.merge(new_df, df4, how='inner', on = 'AreaName')

print("Total number of immigrants per decade \n ",new_df)

ax2=new_df.plot(kind = 'bar', color=['red', 'green', 'blue','yellow'], figsize = (15,6), rot = 70)

labels = ['decade1','decade2','decade3','decade4']

ax2.legend(labels = labels)

ax2.set_xlabel('Immigration by continents')

ax2.set_ylabel('Immigrants')

plt.show()
new_df.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=True)
def create_plot(newc,decade):

    newc.plot(kind="bar",figsize=(10, 10))

    plt.ylabel('Number of immigrants')

    plt.xlabel('Countries')

    plt.title('Immigrant distribution for '+decade)
decades = ['decade1','decade2','decade3','decade4']

plt.figure(1,figsize=(10, 10))

for decade in decades:

    country=df[['OdName',decade]].groupby(['OdName']).sum(axis=1).sum(level=['OdName'])

    # print(country)

    mean=country.mean()

    newc = country[(country > mean).all(axis=1)]

#     print (newc)

#     indices= decades.index(decade)

#     print(indices+1)

#     plt.subplot(4,1,indices+1)

#     plt.subplots_adjust(hspace=0.9)

    create_plot(newc,decade)
plt.style.use('_classic_test')



colors = plt.cm.cool(np.linspace(0, 50, 100))

df['DevName'].value_counts().plot.pie(colors = colors,

                                       figsize = (10, 10))



plt.title('Types of Countries', fontsize = 20, fontweight = 30)

plt.axis('off')

plt.legend()

plt.show()
df1=df[['DevName','decade1']].groupby(['DevName']).sum(axis=1).sum(

    level=['DevName'])

df2=df[['DevName','decade2']].groupby(['DevName']).sum(axis=1).sum(

    level=['DevName'])

df3=df[['DevName','decade3']].groupby(['DevName']).sum(axis=1).sum(

    level=['DevName'])

df4=df[['DevName','decade4']].groupby(['DevName']).sum(axis=1).sum(

    level=['DevName'])

new_df=pd.merge(df1, df2, how='inner', on = 'DevName')

new_df=pd.merge(new_df, df3, how='inner', on = 'DevName')

new_df=pd.merge(new_df, df4, how='inner', on = 'DevName')

print("Background of immigrants per decade \n ",new_df)

def create_pi_plot(new_df,decade):

    new_df.plot(kind="bar", 

                 figsize=(8,8),

                 stacked=True)
plt.figure(1,figsize=(10, 10))

create_pi_plot(new_df,decades)
# download countries geojson file

import folium

# download countries geojson file

!wget --quiet https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/world_countries.json -O world_countries.json

    

print('GeoJSON file downloaded!')

world_geo = r'world_countries.json' # geojson file



# create a plain world map

world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')

import warnings

warnings.filterwarnings('ignore')



# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013

world_map.choropleth(

    geo_data=world_geo,

    data=df,

    columns=['OdName', 'Total'],

    key_on='feature.properties.name',

    fill_color='Greens', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Immigration to Canada'

)



# display map

world_map