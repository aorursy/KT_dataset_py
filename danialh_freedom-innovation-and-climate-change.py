# modules we'll use

import pandas as pd #linear algebra

import numpy as np



# for Box-Cox Transformation

from scipy import stats



# for min_max scaling

from mlxtend.preprocessing import minmax_scaling



# plotting modules

import seaborn as sns

import matplotlib.pyplot as plt



# set seed for reproducibility

np.random.seed(0)



from scipy.stats import norm

import plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly.tools import FigureFactory as ff

from wordcloud import WordCloud,STOPWORDS

from PIL import Image

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")
pathy = "../input/the-human-freedom-index/hfi_cc_2018.csv"

data = pd.read_csv(pathy)

data.describe()
data_2016 = data.loc[data['year'] == 2016]

data_2016.describe()
data1 = data['pf_score']

data1.describe()
data_2016 = data_2016.loc[:, (data_2016.isnull().sum(axis=0) <= 1242)]



# Rename the columns for a better undestanding

data_2016.rename(columns={"pf_score": "Personal Freedom Score",

                     "pf_ss": "Security",

                     "pf_expression": "Freedom_of_Expression",

                     "pf_religion": "Freedom of Religion",

                     "pf_rol_civil": "Civil Justice",

                     "ef_government": "Size of Government", 

                     "ef_legal": "Legal System and Property Rights",

                     "ef_money": "Sound Money",

                     "ef_trade": "Freedom to Trade Internationally"}, inplace=True)

data_2016.head()
sns.distplot(data_2016['Personal Freedom Score'], fit = norm, color = 'blue');
sns.distplot(data_2016['hf_score'], fit = norm, color = '#2980b9');
sns.distplot(data_2016['ef_score'], fit = norm, color = '#3498db');
data2016_corr = data_2016[["Personal Freedom Score", "Civil Justice", "Security", "Freedom_of_Expression", "Freedom of Religion"]]

sns.heatmap(data2016_corr.corr(), square=True, cmap='Blues')

plt.show()

data2016_corr = data_2016[["ef_score", "Size of Government", "Legal System and Property Rights", "Sound Money", "Freedom to Trade Internationally"]]

sns.heatmap(data2016_corr.corr(), square=True, cmap='Blues')

plt.show()
import plotly.figure_factory as ff

#prepare data

dataframe = data[data.year == 2016]

dataVar = dataframe.loc[:,["ef_trade", "pf_expression"]]

dataVar["index"] = np.arange(1,len(dataVar)+1)

#scatter matrix

fig = ff.create_scatterplotmatrix(dataVar, diag='box', index ='index', colormap = 'Portland', 

                                    colormap_type='cat',

                                   height = 900, width =900)

iplot(fig)
x = data_2016['Personal Freedom Score'].values

y = data_2016['hf_score'].values

z = data_2016['ef_score'].values





trace1 = go.Scatter3d(

    x=x,

    y=y,

    z=z,

    mode='markers',

    marker=dict(

        size=12,

        color= 'blue',                # set color to an array/list of desired values

        colorscale='Jet',   # choose a colorscale

        opacity=0.5

    )

)



data = [trace1]

layout = go.Layout(

    showlegend=True,

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df = pd.read_csv("../input/insead-global-innovation-index/INSEADGlobalinnovationIndex2018.csv")

df.head(10)
pathy = "../input/the-human-freedom-index/hfi_cc_2018.csv"

data = pd.read_csv(pathy)

hdi = ['Switzerland', 'Netherlands', 'Sweden', 'United Kingdom', 'Ireland', 'Singapore', 'United States of America', 'Finland', 'Denmark','Germany']

df_innovation = df[df.Economy.isin(hdi)]

data_innovation = data[data.countries.isin(hdi)]

# import graph objects as "go"

import plotly.graph_objs as go



x = df_innovation.Economy



trace1 = {

  'x': x,

  'y': data_innovation.ef_rank,

  'name': 'Economic Freedom Score',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': data_innovation.hf_rank,

  'name': 'Human Freedom Score',

  'type': 'bar'

};

trace3 = {

  'x': x,

  'y': data_innovation['pf_rank'],

  'name': 'Personal Freedom Score',

  'type': 'bar'

};

data = [trace1, trace2, trace3];

layout = {

  'xaxis': {'title': ' Countries in 2016'},

  'barmode': 'relative',

  'title': 'Personal, Human, and Economic Freedom Rank For Top 10 Global Innovation Index Countries'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
top = ['United States of America', 'China', 'India', 'Russian Federation', 'Japan']

df_power = df[df.Economy.isin(top)]

df_power.describe()
clean = ["Iceland", "Switzerland", "Costa Rica", "Sweden", "Norway"]

df_clean = df[df.Economy.isin(clean)]

df_clean.describe()
human = pd.read_csv('../input/the-human-freedom-index/hfi_cc_2018.csv')

power = ['United States', 'China', 'India', 'Russia', 'Japan', 'Germany', 'South Korea', 'Iran', 'Canada', 'Saudi Arabia']

data_power = data_2016[data_2016.countries.isin(power)]
# import graph objects as "go"

import plotly.graph_objs as go



x = data_power.countries



trace1 = {

  'x': x,

  'y': data_power.ef_rank,

  'name': 'Economic Freedom Score',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': data_power.hf_rank,

  'name': 'Human Freedom Score',

  'type': 'bar'

};

trace3 = {

  'x': x,

  'y': data_power['pf_rank'],

  'name': 'Personal Freedom Score',

  'type': 'bar'

};

data = [trace1, trace2, trace3];

layout = {

  'xaxis': {'title': ' Countries in 2016'},

  'barmode': 'relative',

  'title': 'Personal, Human, and Economic Freedom Rank For Top 10 Green House Gas Emitting Countries'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
from mpl_toolkits.basemap import Basemap

concap = pd.read_csv('../input/world-capitals-gps/concap.csv')

data18 = pd.read_csv('../input/the-human-freedom-index/hfi_cc_2018.csv')

data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         data18,left_on='CountryName',right_on='countries')

def mapWorld():

    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,\

            llcrnrlon=-110,urcrnrlon=180,resolution='c')

    m.drawcoastlines()

    m.drawcountries()

    m.drawparallels(np.arange(-90,91.,30.))

    m.drawmeridians(np.arange(-90,90.,60.))

    lat = data_full['CapitalLatitude'].values

    lon = data_full['CapitalLongitude'].values

    a_1 = data_full['ef_regulation_business'].values

    #a_2 = data_full['Economy (GDP per Capita)'].values

    #300*a_2

    m.scatter(lon, lat, latlon=True,c=a_1,s=500,linewidth=1,edgecolors='black',cmap='Blues', alpha=1)

    

    #m.fillcontinents(color='#FFFFFF',lake_color='#FFFFFF',alpha=0.3)

    cbar = m.colorbar()

    cbar.set_label('Business Regulation',fontsize=30)

    #plt.clim(20000, 100000)

    plt.title("Business Regulation (score)", fontsize=30)

    plt.show()

plt.figure(figsize=(30,30))

mapWorld()
from mpl_toolkits.basemap import Basemap

concap = pd.read_csv('../input/world-capitals-gps/concap.csv')

data18 = pd.read_csv("../input/insead-global-innovation-index/INSEADGlobalinnovationIndex2018.csv")

data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         data18,left_on='CountryName',right_on='Economy')

def mapWorld():

    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,\

            llcrnrlon=-110,urcrnrlon=180,resolution='c')

    m.drawcoastlines()

    m.drawcountries()

    m.drawparallels(np.arange(-90,91.,30.))

    m.drawmeridians(np.arange(-90,90.,60.))

    lat = data_full['CapitalLatitude'].values

    lon = data_full['CapitalLongitude'].values

    a_1 = data_full['Score'].values

    #a_2 = data_full['Economy (GDP per Capita)'].values

    #300*a_2

    m.scatter(lon, lat, latlon=True,c=a_1,s=500,linewidth=1,edgecolors='black',cmap='Reds', alpha=1)

    

    #m.fillcontinents(color='#FFFFFF',lake_color='#FFFFFF',alpha=0.3)

    cbar = m.colorbar()

    cbar.set_label('Global Innovation Score',fontsize=30)

    #plt.clim(20000, 100000)

    plt.title("Global Innovation Index (score)", fontsize=30)

    plt.show()

plt.figure(figsize=(30,30))

mapWorld()