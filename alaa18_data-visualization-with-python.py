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
#Importing necessary librarires for data wrangling

import numpy as np  
import pandas as pd
df = pd.read_csv('https://cocl.us/datascience_survey_data', sep=',', index_col=0) #read the survey
df.head() #print the first 5 rows data frame
print(df.shape)
df.info()
%matplotlib inline 

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot') # optional: for ggplot-like style

# check for latest version of Matplotlib
print ('Matplotlib version: ', mpl.__version__) # >= 2.0.0
df.sort_values(['Very interested'], ascending=False, axis=0, inplace=True)
df = round((df/2233)*100,2)
df.head()
# Plotting
ax = df.plot(kind='bar', 
                figsize=(20, 8),
                rot=90,color = ['#5cb85c','#5bc0de','#d9534f'],
                width=.8,
                fontsize=14)


# Setting plot title
ax.set_title('Percentage of Respondents Interest in Data Science Areas',fontsize=16)

# Setting figure background color
ax.set_facecolor('white')

# setting legend font size
ax.legend(fontsize=14,facecolor = 'white') 

# Removing the Border 
ax.get_yaxis().set_visible(False)

# Creating a function to display the percentage.

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14
               )
plt.show()
df_sfc = pd.read_csv('https://cocl.us/sanfran_crime_dataset')

print('Dataset downloaded and read into a pandas dataframe!')
df_sfc.head()
df_sfc.info()
neigh = df_sfc['PdDistrict'].value_counts()
# Assigning the values of the variable to a Pandas Data frame
df_neigh = pd.DataFrame(data= neigh.values , index = neigh.index , columns = ['Counts'])

# Reindexing the data frame to the requirement
df_neigh = df_neigh.reindex(["CENTRAL", "NORTHERN", "PARK", "SOUTHERN", "MISSION", "TENDERLOIN", "RICHMOND", "TARAVAL", "INGLESIDE", "BAYVIEW"])

# Resetting the index
df_neigh = df_neigh.reset_index()

# Assignming the column names
df_neigh.rename({'index': 'Neighborhood'}, axis='columns', inplace=True)

# View the data frame
df_neigh
import folium
!wget --quiet https://cocl.us/sanfran_geojson -O sanfran_geo.json
    
print('GeoJSON file downloaded!')
sf_geo = r'sanfran_geo.json' # geojson file

# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42

# create a plain world map
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)

# generate choropleth map
sanfran_map.choropleth(
    geo_data=sf_geo,
    data=df_neigh,
    columns=['Neighborhood', 'Counts'],
    key_on='feature.properties.DISTRICT',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Crime Rate in San Fransisco'
)

# display map
sanfran_map