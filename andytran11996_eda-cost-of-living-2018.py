%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt # visualization
import seaborn as sns # visualization
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True) 
import numpy as np
import warnings  
warnings.filterwarnings('ignore')

from_year = 2016
to_year = 2018

dfs = []
for y in range(from_year, to_year+1):
    df = pd.read_csv('../input/cost-of-living-%d.csv' % (y))
    df['Year'] = y
    print(df.columns.values)
    dfs.append(df)
    
def extract_city_country(loc):
    s = loc.split(',')
    return s[0].strip(), s[-1].strip()

rename_rules = {
    'Cost of Living Index': 'CLI',
    'Cost of Living Plus Rent Index' : 'CLRI',
    'Rent Index': 'RI',
    'Groceries Index': 'GI',
    'Restaurant Price Index': 'RPI',
    'Local Purchasing Power Index': 'LPPI',
}

reversed_rename_rules = {v: k for k, v in rename_rules.items()}


for df in dfs:
    df.columns = [c.replace('.', ' ') for c in df.columns]
    df.rename(columns= rename_rules, inplace=True)
    if 'Country' not in df.columns:
        df['City'], df['Country'] = zip(*df['City'].apply(extract_city_country).values)

dfs = [df[['City', 'Country', 'CLI', 'RI', 'CLRI', 'GI', 'RPI', 'LPPI', 'Year']] for df in dfs]
df = pd.concat(dfs)
df['Country'] = df['Country'].apply(lambda c: c.strip() if len(c) > 2 else 'United States').values
df.head(10)
df_top20_cli_2018 = df[df['Year'] == 2018].sort_values(['CLI'], ascending=False).head(20).sort_values(['CLI'])
top20cli_2018 = [go.Bar(
            x=df_top20_cli_2018['CLI'].values,
            y=df_top20_cli_2018['City'].values,
            orientation = 'h',
            marker=dict(
            color=df_top20_cli_2018.sort_values(['CLI'], ascending=True)['CLI'].values,
            colorscale='RdBu',
            opacity=0.7
        ),
)]

iplot(top20cli_2018)
df_top20_l_2018 = df[df['Year'] == 2018].sort_values(['LPPI'], ascending=False).head(20).sort_values(['LPPI'])
top20cli_l_2018 = [go.Bar(
            x=df_top20_l_2018['LPPI'].values,
            y=df_top20_l_2018['City'].values,
            orientation = 'h',
            marker=dict(
                color=df_top20_cli_2018.sort_values(['LPPI'], ascending=False)['LPPI'].values,
                colorscale='Greens'
            ),
                opacity=0.8

)]

iplot(top20cli_l_2018)
plt.figure(figsize=(20,5))
sns.heatmap(df.corr(),cmap='RdBu_r', annot=True)
sns.pairplot(df)
layout = go.Layout(
    autosize=False,
    width=600,
    height=600,
    
    xaxis=dict(
        title='Cost of Living Index'
    ),
    yaxis=dict(
        title='Local Purchase Power Index'
    )
)
trace = go.Scatter(
    x = df['CLI'],
    y = df['LPPI'],
    mode = 'markers')
data = [trace]
fig = go.Figure(data=data, layout=layout)
# Plot and embed in ipython notebook!
iplot(fig)

g = sns.jointplot(y=df['LPPI'].values, x=df['CLI'].values,kind='kde').set_axis_labels("Cost of Living Index", "Local Purchase Power Index")
plt.show()
cols = ['CLI','RI','CLRI','GI','RPI','LPPI']
df_hamburg = df[df['City'] == 'Hamburg']
data = []
for col in cols:
    trace = go.Scatter(
        x = df_hamburg['Year'],
        y = df_hamburg[col],
        name = reversed_rename_rules[col],
        mode = 'lines+markers',    )
    data.append(trace)
layout = go.Layout(
    title="Hamburg's Cost of Living 2016-2018",
    autosize=False,
    width=800,
    height=600,
    xaxis=dict(
        title="Year",
        autorange=True,
        showgrid=False,
        zeroline=True,
        showline=True,
        tickmode='array',
        tickvals=df_hamburg['Year'].values,
        showticklabels=True,
        ticklen=4,
        tickwidth=2,
        tickcolor='#000'
    ),
    yaxis=dict(
        title="Index value",
        autorange=True,
        showgrid=False,
        zeroline=True,
        showline=True,
        tickmode='array',
        showticklabels=True,
        ticklen=4,
        tickwidth=2,
        tickcolor='#000'
    ),
)
fig = go.Figure(data=data, layout=layout)
# Plot and embed in ipython notebook!
iplot(fig)
from ipywidgets import interactive, HBox, VBox
from IPython.display import display, clear_output, Image
from plotly.widgets import GraphWidget
import ipywidgets as widgets
df_groupby_country = df.groupby(['Country','Year'], as_index=False).mean().round(2)

data = dict(type="choropleth",
           locations = df_groupby_country[df_groupby_country['Year'] == 2016]['Country'].values,
            locationmode = "country names",
           z = df_groupby_country[df_groupby_country['Year'] == 2016]['CLI'].values,
            colorscale='Reds',
           colorbar = {'title':'CLI'})

layout = dict(title="World AVG Cost of Living Index",
              width=1000,
            height=600,
             geo = dict(showframe=False,
                      projection = {'type':'mercator'}))
choromap = go.Figure(data = [data],layout = layout)
f = go.FigureWidget(choromap)
def update_z(year):
    f.data[0].z = sum(zip(df_groupby_country[df_groupby_country['Year'] == year]['CLI'].values),())
    f.data[0].locations = sum(zip(df_groupby_country[df_groupby_country['Year'] == year]['Country'].values),())
    f.layout.title = "World AVG Cost of Living Index %d" % (year)

year_slider = interactive(update_z, year=(2016, 2018, 1))
vb = VBox((f, year_slider))
vb.layout.align_items = 'center'
vb
