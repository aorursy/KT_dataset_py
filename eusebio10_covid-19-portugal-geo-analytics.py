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
from IPython.core.display import HTML
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from pathlib import Path
import plotly.offline as py
import plotly.express as px
import cufflinks as cf
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv
HTML('''<div class="flourish-embed flourish-map" data-src="visualisation/1914665" data-url="https://flo.uri.sh/visualisation/1914665/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
HTML('''<div class="flourish-embed" data-src="visualisation/1918080" data-url="https://flo.uri.sh/visualisation/1918080/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
df_covid19_cod = pd.read_csv('/kaggle/input/covid19-cod-041220/COVID-19 cod.csv',encoding='latin1',sep=";")
df_covid19_cod = df_covid19_cod.drop(columns=['Unnamed: 23'])
df_covid19_cod_melt = pd.melt(df_covid19_cod, id_vars=['concelho','longitude','latitude'], var_name="data", value_name="confirmados")
df_covid19_cod_melt = df_covid19_cod_melt.dropna()
df_importados = pd.read_excel('/kaggle/input/casos-importados/Casos Importados.xlsx',header = 0,skiprows=0 )
df_importados_melt = pd.melt(df_importados, id_vars=[ 'LATITUDE INFETADO', 'LONGITUDE INFETADO', 'PAÍS ORIGEM', 'CODIGO INFETADO', 'CODIGO PAÍS ORIGEM',
                                                     'PAÍS INFETADO','longitude','latitude'], var_name="data", value_name="confirmados")
df_importados_melt.to_csv('df_importados_meltv5.csv', sep=';', encoding='latin1',index=False) # saving df 
df_covid19_cod_melt.to_csv('COVID-19_041220_meltv5.csv', sep=';', encoding='latin1',index=False) # saving df 
df_importados.head(4)
df_covid19_cod_melt.head(10)
df_covid19_cod_melt.data.max()
df_covid19_cod_melt[df_covid19_cod_melt['data'] == "2020/04/12"].sort_values('confirmados', ascending = False)['concelho'].head(10)
sns.jointplot(x='longitude', y='latitude', data=df_covid19_cod_melt, kind='kde', annot_kws=dict(stat="r") )
plt.title('Covid19 Concelhos density')
center_point = dict(lon= -9.156161655,lat= 38.74267955)
figx = px.density_mapbox(df_covid19_cod_melt, lat='latitude', lon='longitude', z="confirmados",
                        center = center_point, hover_name='concelho', zoom = 5,
                         range_color= [10, 20] , radius=10,
                        mapbox_style= 'open-street-map', title='Heatmap COVID-19 in Portugal',
                        animation_frame='data')
figx.update(layout_coloraxis_showscale=False)
figx.show()
size = df_covid19_cod_melt.confirmados.pow(0.4)
figy = px.scatter(df_covid19_cod_melt, x='latitude', y='longitude',
                  color="confirmados", 
                        hover_name='concelho', size=size,
                         title='Covid19 Portugal',
                        animation_frame='data')
figy.update(layout_coloraxis_showscale=False)
figy.show()
figs = px.scatter_3d(df_covid19_cod_melt, x='longitude', y='latitude', z='confirmados',
                 hover_name='concelho', 
                    size= size, opacity=0.7,
                     animation_frame='data', color='confirmados')
figs.update(layout_coloraxis_showscale=False)
figs.show()
K_clusters = range(1,7)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = df_covid19_cod_melt[['latitude']]
X_axis = df_covid19_cod_melt[['longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# Visualize
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
n_clusters = 3 # Selected by Elbow Curve
data = df_covid19_cod_melt
data['cluster'] = KMeans(n_clusters=n_clusters, n_init=1, max_iter=50, random_state=42).fit_predict(data[['longitude', 'latitude']])
data_groupby_cluster = data.groupby('cluster')
data_by_cluster = data_groupby_cluster.sum().reset_index(drop=True)
data_by_cluster['longitude'] = data_groupby_cluster['longitude'].mean()
data_by_cluster['latitude'] = data_groupby_cluster['latitude'].mean()
#alternative
kmeans = KMeans(n_clusters = 3, init ='k-means++')
centers = kmeans.fit(data[['longitude', 'latitude']]).cluster_centers_
data.plot.scatter(x = 'latitude', y = 'longitude', c='cluster', s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
