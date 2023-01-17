import pandas as pd
import numpy as np

import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

from IPython.display import display

warnings.filterwarnings('ignore')
apps = pd.read_csv('../input/googleplaystore.csv')
reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')
apps.head(3)
len(apps)
def print_missing_values(data):
    data_null = pd.DataFrame(len(data) - data.notnull().sum(), columns = ['Count'])
    data_null = data_null[data_null['Count'] > 0].sort_values(by='Count', ascending=False)

    trace = go.Bar(x=data_null.index, y=data_null['Count'], marker=dict(color='#c0392b'),
              name = 'At least one missing value', opacity=0.9)
    layout = go.Layout(barmode='group', title='Column with missing values in Apps dataset', showlegend=True,
                   legend=dict(orientation="h"))
    fig = go.Figure([trace], layout=layout)
    py.iplot(fig)
print_missing_values(apps)
apps[apps[['Type', 'Content Rating']].isnull().any(axis=1)]
apps.at[9148, 'Type'] = 'Free'
for i in reversed(range(2,len(apps.columns))):
    apps.iat[10472, i]= apps.iloc[10472][i-1]
apps.at[10472, 'Category'] = 'UNKNOWN'
apps.loc[[9148,10472]]
reviews.head(3)
len(reviews)
print_missing_values(reviews)
app = apps['App'].value_counts()
len(app[app > 1])
apps = apps.drop_duplicates(subset=['App'])
def genres_split(genres):
    genres = genres.str.split(';', expand=True)
    return genres[0].append(genres[1].dropna(), ignore_index=True)

def update_size(size):
    size[size == 'Varies with device'] = '0k'
    size = size.apply(lambda s: float(s[:-1])*1000 if s[-1] == 'M' else s[:-1])
    size[size == '0'] = np.NAN
    return size.astype('float64')
apps["Reviews"] = pd.to_numeric(apps["Reviews"])
apps["Price"] = pd.to_numeric(apps["Price"].str.replace('$',''))
apps["Last Updated"] = pd.to_datetime(apps["Last Updated"])
apps['Size'] = update_size(apps['Size'])
apps['Installs'] = apps['Installs'].str.replace(',','')
apps.describe()
def plot_bar(column, color='#2980b9', orientation = 'h', type=None):
    data =  apps[column].dropna().value_counts(ascending=True) if column != 'Genres' else genres_split(apps[column]).value_counts(ascending=True)
    if column == 'Installs':
        data.index = data.index.str.replace('+','').astype('int64')
        data.sort_index(inplace=True)
    x, y = (data.values, data.index) if orientation == 'h' else (data.index, data.values)
    trace = go.Bar(x=x, y=y, marker=dict(color=color)
           , opacity=0.9, orientation = orientation)
    layout = go.Layout(barmode='group', title='Bar plot for the '+column+' column', xaxis=dict(type=type))
    fig = go.Figure([trace], layout=layout)
    py.iplot(fig)
    
def plot_histo(column, type=None):
    trace = go.Histogram(x=apps[column].dropna())
    layout = go.Layout(title='Histogram of the '+column+' column'
                       , yaxis= dict(type=type,autorange=True))
    fig = go.Figure([trace], layout=layout)
    py.iplot(fig)
plot_histo('Rating')
plot_histo('Reviews', type='log')
plot_histo('Size')
plot_histo('Price', type='log')
plot_bar('Category')
plot_bar('Installs', orientation='v', type='category')
plot_bar('Content Rating')
plot_bar('Type', orientation='v')
plot_bar('Genres')
plot_bar('Last Updated', orientation='v')
def top_n_plot(column, n=10):
    data = apps[column].dropna().sort_values(ascending=False)[0:n]
    data.index = apps.loc[data.index,'App']
    
    gold, silver, bronze, other = ('#FFA400','#bdc3c7','#cd7f32','#3498db')
    color = [gold if i == 1 else silver if i == 2 else bronze if i == 3 else other for i in range(1,n+1)]
    
    x, y = (data.index, data.values)
    trace = go.Bar(x=x, y=y, marker=dict(color=color)
           , opacity=0.9, orientation = 'v')
    layout = go.Layout(barmode='group', title='Top '+str(n)+' for the '+column+' column')
    fig = go.Figure([trace], layout=layout)
    py.iplot(fig)
top_n_plot('Price', n=20)
top_n_plot('Reviews', n=15)
top_n_plot('Size', n=15)
top_n_plot('Rating', n=15)