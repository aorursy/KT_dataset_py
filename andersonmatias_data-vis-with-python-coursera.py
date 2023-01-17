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

import matplotlib as mpl
import matplotlib.pyplot as plt
print("Done!")
file =  'https://cocl.us/datascience_survey_data'
df = pd.read_csv(file)
df
%matplotlib inline
df.sort_values(by=['Very interested'], inplace=True, ascending=False)
df.rename(columns={'Unnamed: 0':'Topic'},inplace=True)
df_perc = df[['Topic']]
df_perc = df_perc.join((df[['Very interested','Somewhat interested','Not interested']]/2233)*100)
df_perc.set_index('Topic', inplace=True)
df_perc.round(2)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels =['Data Analysis / Statistics','Machine Learning','Data Visualization','Big Data (Spark / Hadoop)','Deep Learning','Data Journalism']
very_int = df_perc['Very interested']
some_int = df_perc['Somewhat interested']
not_int = df_perc['Not interested']

ind = np.arange(len(very_int))  
width = 0.3

fig, ax = plt.subplots(figsize=(20,8))
rects1 = ax.bar(ind - width, very_int, width, label='Very interested', color='#5cb85c')
rects2 = ax.bar(ind, some_int, width, label='Somewhat interested', color='#5bc0de')
rects3 = ax.bar(ind + width, not_int, width, label='Notr interested', color='#d9534f')

ax.set_title("Percentage of Respondents' Interest In Data Science Areas", fontsize=16)
ax.set_xticks(ind)
ax.set_xticklabels((labels))
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=14)


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height().round(2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom', fontsize=14)


autolabel(rects1, "center")
autolabel(rects2, "center")
autolabel(rects3, "center")

fig.tight_layout()

plt.show()
file =  'https://cocl.us/sanfran_crime_dataset'
df_sf = pd.read_csv(file)
df_sf.head()
df_sf_neigh = df_sf.groupby(["PdDistrict"]).count().reset_index()
df_sf_neigh.drop(df_sf_neigh.columns.difference(['PdDistrict','IncidntNum']), 1, inplace=True)
df_sf_neigh.rename(columns={'PdDistrict':'Neighborhood','IncidntNum':'Count'}, inplace=True)
df_sf_neigh
!wget --quiet https://cocl.us/sanfran_geojson
!conda install -c conda-forge folium=0.5.0 --yes
import folium

print('Folium installed and imported!')
print('GeoJSON file downloaded!')
sf_geo = 'https://cocl.us/sanfran_geojson'

sf_latitude = 37.77
sf_longitude = -122.42
sf_map = folium.Map(location=[sf_latitude,sf_longitude], zoom_start=12)
sf_map.choropleth(
    geo_data=sf_geo,
    data=df_sf_neigh,
    columns=['Neighborhood', 'Count'],
    key_on='feature.properties.DISTRICT',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Crime Rate per District in San Francisco')

sf_map
