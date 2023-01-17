# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# The first thing I did was to reformat 'COVID DEPARTAMENTOS.csv' file. Apparently, I was getting all the values under 
# one column.

#Lo primero que hize fue actualizar "PERU DEPARTAMENTOS.CSV" ya que estaba en un formato differente (me salian comillas
# y todo en la una columna).

df = pd.read_excel('../input/coviddept/PeruDepartamentos(formated).xlsx')
df.head()
# Number of cases by department (sum)
# Suma total de casos por departamento


df = df.groupby(['DEPARTAMENTO'])['CASOS'].sum().to_frame()

# reiniciando el index
# Resetting the index
df.reset_index(level=0, inplace=True)
df.head()
# Un Bar para ver el number de casos por departamento (Lima como lider)
# bar plot showing the number of cases by department (Lima being the first one)
import plotly.express as px
fig = px.bar(df[df['DEPARTAMENTO']!= 'PERU'], y="CASOS", x="DEPARTAMENTO", color="DEPARTAMENTO"
            )
fig.show()


# Using a Geojson map of Peru to see a number of cases by Department
# A big Thanks to Juan Eladio Sánchez Rosas for the file!!

# github: 
#https://github.com/juaneladio/peru-geojson

import json
with open('../input/geojson-departamentos-peru/peru_departamental_simple.geojson') as response:
    peru_geo = json.load(response)
# Creando un Choropleth:
fig = px.choropleth(df[df['DEPARTAMENTO']!= 'PERU'], geojson=peru_geo, color="CASOS",
                    locations="DEPARTAMENTO", featureidkey="properties.NOMBDEP",
                    projection="mercator", range_color= (0,300),
                    labels = {'CASOS':"Casos",'DEPARTAMENTO': 'Departamento'}
                   )
# ajustando la resolucion
fig.update_geos(fitbounds="locations", visible=False,showcountries = True,
                resolution=50,
                showcoastlines=True, coastlinecolor="Black",
               showocean=True, oceancolor="LightBlue",
               showlakes=True, lakecolor="LightBlue",
              )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, 
                  title_text = 'Covid-19 en el Peru (Por Departamento)')


fig.show()