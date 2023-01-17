# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

%matplotlib inline

#%pylab inline

import folium



opioids = pd.read_csv("/kaggle/input/us-opiate-prescriptions/opioids.csv")

presinfo = pd.read_csv("/kaggle/input/us-opiate-prescriptions/prescriber-info.csv")

overdoses = pd.read_csv("/kaggle/input/us-opiate-prescriptions/overdoses.csv")

presinfo.head()
presinfo[presinfo['Gender'].isin(['M','F'])]["Gender"].value_counts().plot(kind='pie',title= "Gender Overall",colors= ['green','black'], table =True)
presinfo.Gender.value_counts()
presinfo.Gender.value_counts().plot.bar()
males = presinfo['Opioid.Prescriber'][presinfo['Gender'] == 'M']

males.value_counts()
males.value_counts().plot.bar()
females = presinfo['Opioid.Prescriber'][presinfo['Gender'] == 'F']

females.value_counts() 
females.value_counts().plot.bar()
import plotly.express as px



pres = presinfo.groupby(['Gender'], as_index=False)['Opioid.Prescriber'].sum()



fig = px.bar(pres, x='Gender', y='Opioid.Prescriber')

#fig.xticks()

fig.show()
pr = presinfo[presinfo['Opioid.Prescriber']==1].groupby('Specialty').count()['State'][:15]

pr.sort_values(ascending=False) 
pr = presinfo[presinfo['Opioid.Prescriber']==1].groupby('Specialty').count()['State'][:15]



pr.sort_values(ascending=True).plot(

    kind="barh",

    figsize=(12,3),

    title="Kind of Medicine prescribing opiates")
import plotly.express as px

#presinfo[presinfo['Opioid.Prescriber']==1]

spec = presinfo.groupby(['Specialty'], as_index=False)['Opioid.Prescriber'].count().head(40)#Specialty



fig = px.bar(spec, x='Specialty', y='Opioid.Prescriber') 

fig.show()
spec2 = presinfo.groupby(['Specialty'], as_index=False)['Opioid.Prescriber'].sum()#[:50]#Specialty



fig = px.bar(spec2, x='Specialty', y='Opioid.Prescriber', color='Opioid.Prescriber') 

fig.show()
presinfo["Credentials"][presinfo["Opioid.Prescriber"] == 1].value_counts(normalize=True)



presinfo["Credentials"].value_counts(normalize=True).head(22)
presinfo["Credentials"][presinfo["Opioid.Prescriber"] == 1].value_counts(normalize=True)



# Okay seems like MD prescribe the bulk, what's the proportion of mds?

presinfo["Credentials"].value_counts(normalize=True).head(20).plot.bar()
presinfo["State"].unique()
presinfo["State"].value_counts().head(22).plot.bar()
state = presinfo["State"][presinfo["Opioid.Prescriber"] == 1].value_counts(normalize=True)[:12]



state#.head(12)#.value_counts()
state = presinfo["State"][presinfo["Opioid.Prescriber"] == 1].value_counts(normalize=True)[:12]

state.sort_values(ascending=False).plot(

    kind="bar",

    figsize=(12,3),

    title="Number of US States where opiates are  prescribed the most")
import plotly.express as px



state = presinfo.groupby(['State'], as_index=False)['Opioid.Prescriber'].sum() 



fig = px.bar(state, x='State', y='Opioid.Prescriber')#, color='Opioid.Prescriber')

fig.show()
#import folium

map = folium.Map(location=[39.381266, -97.922211],

                        tiles = "Stamen Toner",

                        zoom_start = 4)



folium.Marker([36.778261, -119.417932],

              popup='California',

              icon=folium.Icon(color='red')

             ).add_to(map)



folium.Marker([31.968599, -99.901813], 

              popup='Texas',

              icon=folium.Icon(color='red') 

             ).add_to(map)



folium.Marker([27.664827, -81.515754], 

              popup='Florida',

              icon=folium.Icon(color='red') 

             ).add_to(map)

             # icon=folium.Icon(color='red',icon='bicycle', prefix='fa')



folium.Marker([43.299428, -74.217933], 

              popup='New York',

              icon=folium.Icon(color='red') 

             ).add_to(map)

    

folium.Marker([41.203322, -77.194525], 

              popup='Pennsylvania',

              icon=folium.Icon(color='green',icon='bar-chart', prefix='fa') 

             ).add_to(map)



folium.Marker([40.417287, -82.907123], 

              popup='Ohio',

              icon=folium.Icon(color='lightgreen',icon='bar-chart', prefix='fa') 

             ).add_to(map)



folium.Marker([40.633125, -89.398528], 

              popup='Illinois',

              icon=folium.Icon(color='lightgreen',icon='bar-chart', prefix='fa') 

             ).add_to(map)



folium.Marker([44.314844, -85.602364], 

              popup='Michigan',

              icon=folium.Icon(color='lightgreen',icon='bar-chart', prefix='fa') 

             ).add_to(map)



folium.Marker([35.75957, -79.0193], 

              popup='North Carolina',

              icon=folium.Icon(color='blue',icon='bar-chart', prefix='fa') 

             ).add_to(map)



folium.Marker([32.157435, -82.907123], 

              popup='Georgia',

              icon=folium.Icon(color='blue',icon='bar-chart', prefix='fa') 

             ).add_to(map)



folium.Marker([42.407211, -71.382437], 

              popup='Massachusetts',

              icon=folium.Icon(color='blue',icon='bar-chart', prefix='fa') 

             ).add_to(map)



folium.Marker([47.751074, -120.740139], 

              popup='Washington',

              icon=folium.Icon(color='blue',icon='bar-chart', prefix='fa') 

             ).add_to(map)



#map.add_child(folium.ClickForMarker(popup="World is awesome"))



map