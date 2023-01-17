import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")

import math

import geopandas as gpd
data_2015 = pd.read_csv('../input/world-happiness/2015.csv')

data_2016 = pd.read_csv('../input/world-happiness/2016.csv')

data_2017 = pd.read_csv('../input/world-happiness/2017.csv')

data_2018 = pd.read_csv('../input/world-happiness/2018.csv')

data_2019 = pd.read_csv('../input/world-happiness/2019.csv')



data_2019.head(5)
map_df = gpd.read_file('../input/shapefile/World_Countries__Generalized_.shp')



map_df.plot()
map_df = map_df.replace({'Russian Federation':'Russia',

                        'Trinidad and Tobago': 'Trinidad & Tobago',

                        "Côte d'Ivoire": 'Ivory Coast',

                        'Congo': 'Congo (Brazzaville)',

                        'Congo DRC':'Congo (Kinshasa)',

                        'Palestinian Territory':'Palestinian Territories'})
data_2019 = data_2019.rename(index = str, columns = {'Country or region':"Country"})

df = data_2019[['Country','Score']]



df.head(5)
merged = map_df.set_index('COUNTRY').join(df.set_index('Country'))



merged.head(5)
print(merged.Score.isnull().sum())
df = df.set_index('Country').T.to_dict('list')

updates = {'Maldives':[5.20],

          'Oman':[6.853],

          'Sudan':[4.14],

          'Djibouti':[4.37],

          'Angola':[3.80],

          'Belize':[5.95599985122681]

          }

df.update(updates)



df
for i in range (len(merged)):

    if math.isnan(merged['Score'][i]):

        if (str(merged['COUNTRYAFF'][i])) in df:

            merged['Score'][i] = float(df[str(merged['COUNTRYAFF'][i])][0])



merged.head(5)
print(merged.Score.isnull().sum())
variable = 'Score'



vmin, vmax = 2.853,7.021 
fig,ax = plt.subplots(1,figsize = (40,24))



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')
fig,ax = plt.subplots(1,figsize = (40,24))



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')



ax.axis('off')



ax.set_title('World Happiness Score',fontdict = {'fontsize':'40'})





sm = plt.cm.ScalarMappable(cmap='viridis',norm = plt.Normalize(vmin = vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')
variable = 'Perceptions of corruption'



df = data_2019[['Country',variable]]



print(df.head(5))



merged = map_df.set_index('COUNTRY').join(df.set_index('Country'))



df = df.set_index('Country').T.to_dict('list')



for i in range (len(merged)):

    if math.isnan(merged[variable][i]):

        if (str(merged['COUNTRYAFF'][i])) in df:

            merged[variable][i] = float(df[str(merged['COUNTRYAFF'][i])][0])



vmin, vmax = merged[variable].min(),merged[variable].max()



fig,ax = plt.subplots(1,figsize = (40,24))



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')



ax.axis('off')



ax.set_title(variable,fontdict = {'fontsize':'40'})





sm = plt.cm.ScalarMappable(cmap='viridis',norm = plt.Normalize(vmin = vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')
variable = 'Generosity'



df = data_2019[['Country',variable]]



print(df.head(5))



merged = map_df.set_index('COUNTRY').join(df.set_index('Country'))



df = df.set_index('Country').T.to_dict('list')



for i in range (len(merged)):

    if math.isnan(merged[variable][i]):

        if (str(merged['COUNTRYAFF'][i])) in df:

            merged[variable][i] = float(df[str(merged['COUNTRYAFF'][i])][0])



vmin, vmax = merged[variable].min(),merged[variable].max()



fig,ax = plt.subplots(1,figsize = (40,24))



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')



ax.axis('off')



ax.set_title(variable,fontdict = {'fontsize':'40'})





sm = plt.cm.ScalarMappable(cmap='viridis',norm = plt.Normalize(vmin = vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')
variable = 'Freedom to make life choices'



df = data_2019[['Country',variable]]



print(df.head(5))



merged = map_df.set_index('COUNTRY').join(df.set_index('Country'))



df = df.set_index('Country').T.to_dict('list')



for i in range (len(merged)):

    if math.isnan(merged[variable][i]):

        if (str(merged['COUNTRYAFF'][i])) in df:

            merged[variable][i] = float(df[str(merged['COUNTRYAFF'][i])][0])



vmin, vmax = merged[variable].min(),merged[variable].max()



fig,ax = plt.subplots(1,figsize = (40,24))



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')



ax.axis('off')



ax.set_title(variable,fontdict = {'fontsize':'40'})





sm = plt.cm.ScalarMappable(cmap='viridis',norm = plt.Normalize(vmin = vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')
variable = 'Healthy life expectancy'



df = data_2019[['Country',variable]]



print(df.head(5))



merged = map_df.set_index('COUNTRY').join(df.set_index('Country'))



df = df.set_index('Country').T.to_dict('list')



for i in range (len(merged)):

    if math.isnan(merged[variable][i]):

        if (str(merged['COUNTRYAFF'][i])) in df:

            merged[variable][i] = float(df[str(merged['COUNTRYAFF'][i])][0])



vmin, vmax = merged[variable].min(),merged[variable].max()



fig,ax = plt.subplots(1,figsize = (40,24))



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')



ax.axis('off')



ax.set_title(variable,fontdict = {'fontsize':'40'})





sm = plt.cm.ScalarMappable(cmap='viridis',norm = plt.Normalize(vmin = vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')
variable = 'Social support'



df = data_2019[['Country',variable]]



print(df.head(5))



merged = map_df.set_index('COUNTRY').join(df.set_index('Country'))



df = df.set_index('Country').T.to_dict('list')



for i in range (len(merged)):

    if math.isnan(merged[variable][i]):

        if (str(merged['COUNTRYAFF'][i])) in df:

            merged[variable][i] = float(df[str(merged['COUNTRYAFF'][i])][0])



vmin, vmax = merged[variable].min(),merged[variable].max()



fig,ax = plt.subplots(1,figsize = (40,24))



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')



ax.axis('off')



ax.set_title(variable,fontdict = {'fontsize':'40'})





sm = plt.cm.ScalarMappable(cmap='viridis',norm = plt.Normalize(vmin = vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')
variable = 'GDP per capita'



df = data_2019[['Country',variable]]



print(df.head(5))



merged = map_df.set_index('COUNTRY').join(df.set_index('Country'))



df = df.set_index('Country').T.to_dict('list')



for i in range (len(merged)):

    if math.isnan(merged[variable][i]):

        if (str(merged['COUNTRYAFF'][i])) in df:

            merged[variable][i] = float(df[str(merged['COUNTRYAFF'][i])][0])



vmin, vmax = merged[variable].min(),merged[variable].max()



fig,ax = plt.subplots(1,figsize = (40,24))



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')



ax.axis('off')



ax.set_title(variable,fontdict = {'fontsize':'40'})





sm = plt.cm.ScalarMappable(cmap='viridis',norm = plt.Normalize(vmin = vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)



merged.plot(column = variable, cmap = 'viridis', linewidth = 0.3, ax=ax,edgecolor = '0.5')