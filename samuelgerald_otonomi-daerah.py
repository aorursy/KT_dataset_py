import pandas as pd
import numpy as np


df = pd.read_csv('../input//indodapoer-world-bank/INDODAPOERData2.csv')
df
df_series = pd.read_csv('../input/indodapoer-world-bank/INDODAPOERSeries2.csv')
list(df_series['Indicator Name'])
DAU = df['Indicator Name'] == 'Total Special Allocation Grant/DAK (in IDR)'
df_dau = df[(DAU)]
peta = []
kode = df_dau.loc[1::,'Country Code']
a =list(kode)
for index, value in enumerate(df_dau['Country Code']):
    if len(a[index]) >6:
          peta.append(value)
df2 = df_dau.reset_index()
df3 = ['Country Name'] + list(df2.describe())
df4 = df2[df3].drop(['index'], axis=1)
df5 = df4.iloc[:,19:39]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import plotly.graph_objs as go

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(120,150))

x= np.arange(0,20,1)
for row in df5.iterrows():
      ax.plot(x, row[1], label=row[0], lw=5)
plt.xticks(x, df5.columns, fontsize=60)
plt.yticks(fontsize=60)
ax.set_title('DAU Daerah Indonesia',fontsize=60)
plt.show()
a = list(df5.columns)
df_pivot_dau = pd.melt(df4.reset_index(), id_vars =['Country Name'], 
        value_vars =a, var_name = "Tahun", value_name="DAU") 
df5.describe()
temp = df_pivot_dau[df_pivot_dau['DAU']>0].sort_values('Country Name', ascending=False)
fig = px.scatter(temp, x='Tahun', y='Country Name', size='DAU', color='DAU', height=9000, 
           color_continuous_scale=px.colors.sequential.Viridis)
fig.update_layout(yaxis = dict(dtick = 1))
fig.update(layout_coloraxis_showscale=False)
fig.show()
fig = go.Figure(data=go.Heatmap(
        z=df_pivot_dau['DAU'],
        x=df_pivot_dau['Tahun'],
        y=df_pivot_dau['Country Name'],
        colorscale='GnBu',
        showlegend=False,
        text=df_pivot_dau['Country Name']))

fig.update_layout(yaxis = dict(dtick = 1))
fig.update_layout(height=9000)
fig.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

# Import GeoJSON Data'
df_geo = gpd.read_file('../input/peta-kabupatenkota-indonesia/county.geojson')
df_peta = df_dau[df_dau['Country Code'].isin(peta)]
import re
from functools import partial
from collections import Counter
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
def countkab(text):
    """ Input: a text, Output: berapa kab """
    return len(re.findall('kab.', text))
from collections import Counter
cnt = Counter()
for text in df_peta['Country Name'].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(1)
FREQWORDS = set([w for (w, wc) in cnt.most_common(1)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])
df_peta["Country Name_2"] = df_peta["Country Name"].apply(lambda text: remove_freqwords(text))
def remove_coma(string):
    emoji_pattern = re.compile("["
                           u","
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
df_peta["Country Name_3"] = df_peta["Country Name_2"].apply(lambda text: remove_coma(text))
from collections import Counter
cnt_kota = Counter()
for text in df_peta['Country Name_3'].values:
    for word in text.split():
        cnt_kota[word] += 1
        
cnt_kota.most_common(1)
KOTA = set([w for (w, wc) in cnt_kota.most_common(1)])
def replace_kota(text):
    if (text.find('Kota')!=-1):
         return  'Kota '+" ".join([word for word in str(text).split() if word not in KOTA])
    else: 
         return text
df_peta["Country Name_4"] = df_peta["Country Name_3"].apply(lambda text: replace_kota(text))
df_geojson = df_peta.drop(['Country Name_2'], axis=1)
#combine those datasets to become one GeoDataFrame and examine the results.
df_join = df_geo.merge(df_geojson, how='inner', left_on="NAME_2", right_on="Country Name_4")
df_join = df_join[['Country Name_4','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','geometry']]
df_join.describe()
#Create Maps
df_geo.plot()
def make_plot(geo_df,values):
    # set a variable that will call whatever column we want to visualise on the map
    values = geo_df.columns[values]
# set the value range for the choropleth
    vmin, vmax = 0,500000000000
# create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(30, 10))
# remove the axis
    ax.axis('off')
# add a title
    title = 'DAK {}'.format(values)
    ax.set_title(title, fontdict={'fontsize': '25', 'fontweight' : '3'})
# create an annotation for the data source
    ax.annotate('Source: Indodapoer -World Bank',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12 ,color='#555555')
# Create colorbar as a legend
    sm = plt.cm.ScalarMappable(cmap='OrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# add the colorbar to the figure
    cbar = fig.colorbar(sm)
# create map
    geo_df.plot(column=values, cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8',norm=plt.Normalize(vmin=vmin, vmax=vmax))
    
    # Add Labels
    geo_df['coords'] = geo_df['geometry'].apply(lambda x: x.representative_point().coords[:])
    geo_df['coords'] = [coords[0] for coords in geo_df['coords']]
    for idx, row in df_join.iterrows():
        plt.annotate(s=row['Country Name_4'], xy=row['coords'],horizontalalignment='center', color = 'Red', fontsize=2)
for i in range(1,21):
    make_plot(df_join,i)
b = list(df_join.columns)
df_pivot_join = pd.melt(df_join.reset_index(), id_vars =['Country Name_4'], 
        value_vars =a, var_name = "Tahun", value_name="DAU")
df_join = df_geo.merge(df_geojson, how='inner', left_on="NAME_2", right_on="Country Name_4")
df_join = df_join[['Country Name_4','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','geometry']]
df_pivot_join['DAU'].max()
df_join_series = df_geo.merge(df_pivot_join, how='inner', left_on="NAME_2", right_on="Country Name_4")
df_join_series = df_join_series[['Country Name_4','Tahun','DAU','geometry']]
import json 
geojson = json.load(open('../input/peta-kabupatenkota-indonesia/county.geojson'))
# Over the time
import plotly.express as px

fig = px.choropleth_mapbox(df_join_series, 
                           geojson=geojson, 
                           locations='Country Name_4', 
                           featureidkey='properties.NAME_2',
                           color='DAU', 
                           center={'lat': -0.7893, 'lon': 113.9213}, 
                           animation_frame=df_join_series["Tahun"],
                           mapbox_style='carto-positron',
                           zoom=2,
                           hover_name='Country Name_4',
                           color_continuous_scale=px.colors.sequential.Purp,
                           range_color=(0,2000000000000)
                          )
fig.update(layout_coloraxis_showscale=True)
fig.show()
df_join_series.to_csv('join_series.csv')
