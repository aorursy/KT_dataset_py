import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.cm as cm
import itertools
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
%matplotlib inline
df = pd.read_csv('../input/countries of the world.csv')
df.head()
#data cleaning, replace all NaN values with 0
df.fillna(0, inplace=True)
#change commas to . so we can process those as float
for column in df.columns:
    try:
        df[column] = df[column].str.replace(",", ".").astype(float)
    except:
        print('')
df.describe()
countries = df['Country'].value_counts()
print('Countries count : {}'.format(len(countries)))
#Population per country
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Population'],
text = df['Country'], colorbar = {'title':'Population'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Population per country',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Net migration'],
text = df['Country'], colorbar = {'title':'Migration'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Popular country for migration',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Infant mortality (per 1000 births)'],
text = df['Country'], colorbar = {'title':'Infant mortality (per 1000 births)'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Best Healthcare',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['GDP ($ per capita)'],
text = df['Country'], colorbar = {'title':'GDP ($ per capita)'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Richest Countries',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Literacy (%)'],
text = df['Country'], colorbar = {'title':'Literacy (%)'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Smartest Countries',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Birthrate'],
text = df['Country'], colorbar = {'title':'Birthrate'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Is having many kids a good sign?',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Deathrate'],
text = df['Country'], colorbar = {'title':'Deathrate'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Deathrate',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Phones (per 1000)'],
text = df['Country'], colorbar = {'title':'Phones (per 1000)'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Good places to sell phones',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Agriculture'],
text = df['Country'], colorbar = {'title':'Agriculture'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Agricultural-focused countries',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Industry'],
text = df['Country'], colorbar = {'title':'Industry'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Industrial-focused countries',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
data = dict(type='choropleth',
locations = df['Country'],
locationmode = 'country names', z = df['Service'],
text = df['Country'], colorbar = {'title':'Service'},
colorscale = 'Viridis', reversescale = True)
layout = dict(title='Service-focused countries',
geo = dict(showframe=False,projection={'type':'Mercator'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)