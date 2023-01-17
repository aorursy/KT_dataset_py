import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # Shuts down irrelevant userWarnings
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.offline as offline
import plotly.graph_objs as go
sui_sex_ratios = pd.read_csv("../input/Male-Female-Ratio-of-Suicide-Rates.csv").sort_values(by='Entity', ascending=True)
sui_both = pd.read_csv("../input/suicide-rates-by-country.csv").sort_values(by='Entity', ascending=True)
sui_all = pd.read_csv("../input/SDGSUICIDE.csv")
# Generatina a merged dataset
suicides_merge = pd.merge(sui_sex_ratios, sui_both, how='inner').sort_values(by='Entity', ascending=True)
suicides_merge.columns = ['Country', 'Code', 'Year', 'Male-Female ratio', 'Total suicide rate']
suicides_merge = suicides_merge.dropna()
sui_all.columns = sui_all.iloc[0].values
for i in range (2, 6):
    sui_all.columns.values[i] = str(int(sui_all.columns[i]))


sui_all = sui_all.drop(0)
sui_all.reset_index(drop = True, inplace = True)
suicides = pd.DataFrame(columns=['Country', 'Male suicide rate', 'Female suicide rate', 'Total suicide rate', 'Year'])
Years = ['2000', '2005', '2010', '2015']

deleted = []
                  
for year in Years:
    both_df = sui_all[sui_all['Sex'].str.contains("Both sexes")][['Country', year]]
    male_df = sui_all[sui_all['Sex'].str.contains("Male")][['Country', year]]
    female_df = sui_all[sui_all['Sex'].str.contains("Female")][['Country', year]]

    both_df.reset_index(drop = True, inplace = True)
    male_df.reset_index(drop = True, inplace = True)
    female_df.reset_index(drop = True, inplace = True)
    country_df = both_df['Country']

    year_df = pd.concat([male_df,female_df,both_df], axis=1)
    del year_df['Country'] # To get a dataframe without multiple 'Country' columns

    year_df = pd.concat([country_df,year_df], axis=1)
    
    year_df['Year'] = year # To get a dataframe that includes Country and Year
    year_df.columns = ['Country', 'Male suicide rate', 'Female suicide rate', 'Total suicide rate', 'Year']
    year_df.replace([0, np.inf], np.nan)
    year_df = year_df.dropna()
    year_df['Male-Female ratio'] = year_df['Male suicide rate'] / year_df['Female suicide rate']
    
    suicides = pd.concat([suicides,year_df])
suicides.reset_index(drop = True, inplace = True)
suicides = suicides.replace('Republic of Korea', 'South Korea')
suicides = suicides.replace('Russian Federation', 'Russia')
suicides = suicides.replace('Bolivia (Plurinational State of)', 'Bolivia')
suicides = suicides.replace('Democratic People\'s Republic of Korea', 'North Korea')
suicides = suicides.replace('Republic of Moldova', 'Moldova')
suicides = suicides.replace('United States of America', 'United States')
suicides = suicides.replace('Czechia', 'Czech Republic')
suicides = suicides.replace('Lao People\'s Democratic Republic', 'Laos')
suicides = suicides.replace('Micronesia (Federated States of)', 'Micronesia')
suicides = suicides.replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
suicides = suicides.replace('The former Yugoslav republic of Macedonia', 'Macedonia')
suicides = suicides.replace('Viet Nam', 'Vietnam')
suicides = suicides.replace('United Republic of Tanzania', 'Tanzania')
suicides = suicides.replace('Iran (Islamic Republic of)', 'Iran')
suicides = suicides.replace('Venezuela (Bolivarian Republic of)', 'Venezuela')
suicides = suicides.replace('Syrian Arab Republic', 'Syria')
suicides = suicides.replace('Brunei Darussalam', 'Brunei')
suicides = suicides.replace('Cabo Verde', 'Cape Verde')
suicides_merge = suicides_merge.replace('Democratic Republic of Congo', 'Democratic Republic of the Congo')
suicides_merge = suicides_merge.replace('Timor', 'Timor-Leste')
suicides_merge = suicides_merge.replace('Micronesia (country)', 'Micronesia')
countriesCodes = dict(zip(suicides_merge['Country'].tolist(), suicides_merge['Code'].tolist()))
countriesNames = dict(zip(suicides_merge['Code'].tolist(), suicides_merge['Country'].tolist()))
newCodes = {'South Sudan':'SSD', 'Kiribati':'KIR', 'Montenegro':'MNE', 'Côte d\'Ivoire':'CIV'}
newNames = {'SSD':'South Sudan', 'KIR':'Kiribati', 'MNE':'Montenegro', 'CIV':'Côte d\'Ivoire'}
countriesCodes = dict(countriesCodes, **newCodes)
countriesNames = dict(countriesNames, **newNames)
print("Size of the database:", suicides.shape)
suicides.head()
suicides.describe()
suicides.info()
print("Size of the merged database:", suicides_merge.shape)
suicides_merge.head()
suicides_merge.describe()
suicides_merge.info()
mask2000 = suicides['Year'].astype(int) == 2000
mask2005 = suicides['Year'].astype(int) == 2005
mask2010 = suicides['Year'].astype(int) == 2010
mask2015 = suicides['Year'].astype(int) == 2015
maskSince1990 = suicides_merge['Year'].astype(int) >= 1990
total2000 = suicides[mask2000][['Country', 'Total suicide rate']].sort_values(by='Total suicide rate', ascending=False)
total2005 = suicides[mask2005][['Country', 'Total suicide rate']].sort_values(by='Total suicide rate', ascending=False)
total2010 = suicides[mask2010][['Country', 'Total suicide rate']].sort_values(by='Total suicide rate', ascending=False)
total2015 = suicides[mask2015][['Country', 'Total suicide rate']].sort_values(by='Total suicide rate', ascending=False)
t2000 = go.Bar(
    x = total2000[total2000['Country'].isin(total2015['Country'].head(10))]['Country'].head(10),
    y = total2000[total2000['Country'].isin(total2015['Country'].head(10))]['Total suicide rate'].head(10),
    marker=dict(color='rgb(129, 67, 116)'),
    name = '2000'
)

t2005 = go.Bar(
    x = total2005[total2005['Country'].isin(total2015['Country'].head(10))]['Country'].head(10),
    y = total2005[total2005['Country'].isin(total2015['Country'].head(10))]['Total suicide rate'].head(10),
    marker=dict(color='rgb(81, 163, 157)'),
    name = '2005'
)

t2010 = go.Bar(
    x = total2010[total2010['Country'].isin(total2015['Country'].head(10))]['Country'].head(10),
    y = total2010[total2010['Country'].isin(total2015['Country'].head(10))]['Total suicide rate'].head(10),
    marker=dict(color='rgb( 183, 105, 92)'),
    name = '2010'
)

t2015 = go.Bar(
    x = total2015['Country'].head(10),
    y = total2015['Total suicide rate'].head(10),
    marker=dict(color='rgb(205, 187, 121)',
    ),
    name = '2015'
)


data = [t2000, t2005, t2010, t2015]

layout = go.Layout(
    title='Top countries with the highest suicide rates for both sexes',
    xaxis={'title':'Countries'},
    yaxis={'title':'Suicide rate (per 100,000 people)'},
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
male2000 = suicides[mask2000][['Country', 'Male suicide rate']].sort_values(by='Male suicide rate', ascending=False)
male2005 = suicides[mask2005][['Country', 'Male suicide rate']].sort_values(by='Male suicide rate', ascending=False)
male2010 = suicides[mask2010][['Country', 'Male suicide rate']].sort_values(by='Male suicide rate', ascending=False)
male2015 = suicides[mask2015][['Country', 'Male suicide rate']].sort_values(by='Male suicide rate', ascending=False)
t2000 = go.Bar(
    x = male2000[male2000['Country'].isin(male2015['Country'].head(10))]['Country'].head(10),
    y = male2000[male2000['Country'].isin(male2015['Country'].head(10))]['Male suicide rate'].head(10),
    marker=dict(color='rgb(55, 83, 109)'),
    name = '2000'
)

t2005 = go.Bar(
    x = male2005[male2005['Country'].isin(male2015['Country'].head(10))]['Country'].head(10),
    y = male2005[male2005['Country'].isin(male2015['Country'].head(10))]['Male suicide rate'].head(10),
    marker=dict(color='rgb(133, 133, 173)'),
    name = '2005'
)

t2010 = go.Bar(
    x = male2010[male2010['Country'].isin(male2015['Country'].head(10))]['Country'].head(10),
    y = male2010[male2010['Country'].isin(male2015['Country'].head(10))]['Male suicide rate'].head(10),
    marker=dict(color='rgb(101, 190, 190)'),
    name = '2010'
)

t2015 = go.Bar(
    x = male2015['Country'].head(10),
    y = male2015['Male suicide rate'].head(10),
    marker=dict(color='rgb(26, 118, 255)'),
    name = '2015'
)

data = [t2000, t2005, t2010, t2015]

layout = go.Layout(
    title='Top countries with the highest suicide rates for males',
    xaxis={'title':'Countries'},
    yaxis={'title':'Suicide rate (per 100,000 people)'},
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
female2000 = suicides[mask2000][['Country', 'Female suicide rate']].sort_values(by='Female suicide rate', ascending=False)
female2005 = suicides[mask2005][['Country', 'Female suicide rate']].sort_values(by='Female suicide rate', ascending=False)
female2010 = suicides[mask2010][['Country', 'Female suicide rate']].sort_values(by='Female suicide rate', ascending=False)
female2015 = suicides[mask2015][['Country', 'Female suicide rate']].sort_values(by='Female suicide rate', ascending=False)
t2000 = go.Bar(
    x = female2000[female2000['Country'].isin(female2015['Country'].head(10))]['Country'].head(10),
    y = female2000[female2000['Country'].isin(female2015['Country'].head(10))]['Female suicide rate'].head(10),
    marker=dict(color='rgb(170, 0, 120)'),
    name = '2000'
)

t2005 = go.Bar(
    x = female2005[female2005['Country'].isin(female2015['Country'].head(10))]['Country'].head(10),
    y = female2005[female2005['Country'].isin(female2015['Country'].head(10))]['Female suicide rate'].head(10),
    marker=dict(color='rgb(198, 141, 227)'),
    name = '2005'
)

t2010 = go.Bar(
    x = female2010[female2010['Country'].isin(female2015['Country'].head(10))]['Country'].head(10),
    y = female2010[female2010['Country'].isin(female2015['Country'].head(10))]['Female suicide rate'].head(10),
    marker=dict(color='rgb(94, 154, 242)'),
    name = '2010'
)

t2015 = go.Bar(
    x = female2015['Country'].head(10),
    y = female2015['Female suicide rate'].head(10),
    marker=dict(color='rgb(0, 0, 120)'),
    name = '2015'
)

data = [t2000, t2005, t2010, t2015]

layout = go.Layout(
    title='Top countries with the highest suicide rates for females',
    xaxis={'title':'Countries'},
    yaxis={'title':'Suicide rate (per 100,000 people)'},
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
rate2000 = suicides[mask2000].sort_values(by='Total suicide rate', ascending=False)
rate2005 = suicides[mask2005].sort_values(by='Total suicide rate', ascending=False)
rate2010 = suicides[mask2010].sort_values(by='Total suicide rate', ascending=False)
rate2015 = suicides[mask2015].sort_values(by='Total suicide rate', ascending=False)
total = go.Bar(
    x = rate2015['Country'].head(15),
    y = rate2015['Total suicide rate'].head(15),
    marker=dict(color='rgb(53, 183, 85)'),
    name = 'Both Sexes'
)

male = go.Bar(
    x = rate2015['Country'].head(15),
    y = rate2015['Male suicide rate'].head(15),
    marker=dict(color='rgb(26, 118, 255)'),
    name = 'Male'
)

female = go.Bar(
    x = rate2015['Country'].head(15),
    y = rate2015['Female suicide rate'].head(15),
    marker=dict(color='rgb(255, 25, 64)'),
    name = 'Female'
)

data = [total, male, female]

layout = go.Layout(
    title='Top countries with the highest suicide rates in 2015',
    xaxis={'title':'Countries'},
    yaxis={'title':'Suicide rate (per 100,000 people)'},
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
total = go.Bar(
    x = rate2010['Country'].head(15),
    y = rate2010['Total suicide rate'].head(15),
    marker=dict(color='rgb(53, 183, 85)'),
    name = 'Both Sexes'
)

male = go.Bar(
    x = rate2010['Country'].head(15),
    y = rate2010['Male suicide rate'].head(15),
    marker=dict(color='rgb(26, 118, 255)'),
    name = 'Male'
)

female = go.Bar(
    x = rate2010['Country'].head(15),
    y = rate2010['Female suicide rate'].head(15),
    marker=dict(color='rgb(255, 25, 64)'),
    name = 'Female'
)

data = [total, male, female]

layout = go.Layout(
    title='Top countries with the highest suicide rates in 2010',
    xaxis={'title':'Countries'},
    yaxis={'title':'Suicide rate (per 100,000 people)'},
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
total = go.Bar(
    x = rate2005['Country'].head(15),
    y = rate2005['Total suicide rate'].head(15),
    marker=dict(color='rgb(53, 183, 85)'),
    name = 'Both Sexes'
)

male = go.Bar(
    x = rate2005['Country'].head(15),
    y = rate2005['Male suicide rate'].head(15),
    marker=dict(color='rgb(26, 118, 255)'),
    name = 'Male'
)

female = go.Bar(
    x = rate2005['Country'].head(15),
    y = rate2005['Female suicide rate'].head(15),
    marker=dict(color='rgb(255, 25, 64)'),
    name = 'Female'
)

data = [total, male, female]

layout = go.Layout(
    title='Top countries with the highest suicide rates in 2005',
    xaxis={'title':'Countries'},
    yaxis={'title':'Suicide rate (per 100,000 people)'},
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
total = go.Bar(
    x = rate2000['Country'].head(15),
    y = rate2000['Total suicide rate'].head(15),
    marker=dict(color='rgb(53, 183, 85)'),
    name = 'Both Sexes'
)

male = go.Bar(
    x = rate2000['Country'].head(15),
    y = rate2000['Male suicide rate'].head(15),
    marker=dict(color='rgb(26, 118, 255)'),
    name = 'Male'
)

female = go.Bar(
    x = rate2000['Country'].head(15),
    y = rate2000['Female suicide rate'].head(15),
    marker=dict(color='rgb(255, 25, 64)'),
    name = 'Female'
)

data = [total, male, female]

layout = go.Layout(
    title='Top countries with the highest suicide rates in 2000',
    xaxis={'title':'Countries'},
    yaxis={'title':'Suicide rate (per 100,000 people)'},
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
total2015['Code'] = ""
for i in range(len(total2015)):
    country = total2015.iloc[i]['Country']
    if country in countriesCodes.keys():
        total2015['Code'].iloc[i] = countriesCodes[country]
    else:
        total2015['Code'].iloc[i] = np.nan
total2015 = total2015.dropna()
male2015['Code'] = ""
for i in range(len(male2015)):
    country = male2015.iloc[i]['Country']
    if country in countriesCodes.keys():
        male2015['Code'].iloc[i] = countriesCodes[country]
    else:
        male2015['Code'].iloc[i] = np.nan
male2015 = male2015.dropna()
female2015['Code'] = ""
for i in range(len(male2015)):
    country = female2015.iloc[i]['Country']
    if country in countriesCodes.keys():
        female2015['Code'].iloc[i] = countriesCodes[country]
    else:
        female2015['Code'].iloc[i] = np.nan
female2015 = female2015.dropna()
data = [go.Bar(
    x = total2015['Country'].head(30),
    y = total2015['Total suicide rate'].head(30),
    marker=dict(color='rgb(255, 210, 119)', line=dict(color='rgb(206, 142, 12)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the highest suicide rates in 2015 for both sexes',
    xaxis={'title':'Countries'},
    yaxis={'title':'Total suicide rate (per 100,000 people)'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [go.Bar(
    x = total2015['Country'].tail(30),
    y = total2015['Total suicide rate'].tail(30),
    marker=dict(color='rgb(198, 141, 227)', line=dict(color='rgb(127, 20, 181)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the lowest suicide rates in 2015 for both sexes',
    xaxis={'title':'Countries'},
    yaxis={'title':'Total suicide rate (per 100,000 people)'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [ dict(
        type = 'choropleth',
        locations = total2015['Code'],
        z = total2015['Total suicide rate'],
        text = total2015['Country'],
        colorscale = [[0,"rgb(255, 60, 20)"],[0.25,"rgb(255, 100, 20)"],[0.5,"rgb(255, 140, 20)"],[0.75,"rgb(255, 200, 20)"],[1,"rgb(225, 255, 150)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(240, 240, 240)',
                width = 1.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Suicide rate (per 100K)'),
      ) ]

layout = dict(
    title = 'Suicide rates in 2015 - Both sexes',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
    
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='world-countries.json')
data = [go.Bar(
    x = male2015['Country'].head(30),
    y = male2015['Male suicide rate'].head(30),
    marker=dict(color='rgb(126, 229, 212)', line=dict(color='rgb(12, 104, 89)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the highest suicide rates in 2015 for males',
    xaxis={'title':'Countries'},
    yaxis={'title':'Male suicide rate (per 100,000 people)'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [ dict(
        type = 'choropleth',
        locations = male2015['Code'],
        z = male2015['Male suicide rate'],
        text = male2015['Country'],
        colorscale = [[0,"rgb(255, 60, 20)"],[0.25,"rgb(255, 100, 20)"],[0.5,"rgb(255, 140, 20)"],[0.75,"rgb(255, 200, 20)"],[1,"rgb(225, 255, 150)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(240, 240, 240)',
                width = 1.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Suicide rate (per 100K)'),
      ) ]

layout = dict(
    title = 'Suicide rates in 2015 - Males',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
    
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='world-countries.json')
data = [go.Bar(
    x = female2015['Country'].head(30),
    y = female2015['Female suicide rate'].head(30),
    marker=dict(color='rgb(255, 165, 210)', line=dict(color='rgb(150, 24, 87)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the highest suicide rates in 2015 for females',
    xaxis={'title':'Countries'},
    yaxis={'title':'Female suicide rate (per 100,000 people)'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [ dict(
        type = 'choropleth',
        locations = female2015['Code'],
        z = female2015['Female suicide rate'],
        text = female2015['Country'],
        colorscale = [[0,"rgb(255, 60, 20)"],[0.25,"rgb(255, 100, 20)"],[0.5,"rgb(255, 140, 20)"],[0.75,"rgb(255, 200, 20)"],[1,"rgb(225, 255, 150)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(240, 240, 240)',
                width = 1.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Suicide rate (per 100K)'),
      ) ]

layout = dict(
    title = 'Suicide rates in 2015 - Females',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
    
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='world-countries.json')
totalSince1990 = suicides_merge[maskSince1990][['Country', 'Total suicide rate']].groupby('Country', as_index=False).mean().sort_values(by='Total suicide rate', ascending=False)
mapSince1990 = suicides_merge[maskSince1990][['Code', 'Total suicide rate']].groupby('Code', as_index=False).mean()
mapSince1990['Country'] = ""
for i in range(len(mapSince1990)):
    mapSince1990['Country'].iloc[i] = countriesNames[mapSince1990.iloc[i]['Code']]
data = [go.Bar(
    x = totalSince1990['Country'].head(30),
    y = totalSince1990['Total suicide rate'].head(30),
    marker=dict(color='rgb(255, 210, 119)', line=dict(color='rgb(206, 142, 12)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the highest suicide rates average since the 90\'s for both sexes',
    xaxis={'title':'Countries'},
    yaxis={'title':'Total suicide rate (per 100,000 people)'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [go.Bar(
    x = totalSince1990['Country'].tail(30),
    y = totalSince1990['Total suicide rate'].tail(30),
    marker=dict(color='rgb(198, 141, 227)', line=dict(color='rgb(127, 20, 181)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the lowest suicide rates average since the 90\'s for both sexes',
    xaxis={'title':'Countries'},
    yaxis={'title':'Total suicide rate (per 100,000 people)'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [ dict(
        type = 'choropleth',
        locations = mapSince1990['Code'],
        z = mapSince1990['Total suicide rate'],
        text = mapSince1990['Country'],
        colorscale = [[0,"rgb(255, 60, 20)"],[0.25,"rgb(255, 100, 20)"],[0.5,"rgb(255, 140, 20)"],[0.75,"rgb(255, 200, 20)"],[1,"rgb(225, 255, 150)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(240, 240, 240)',
                width = 1.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Suicide rate (per 100K)'),
      ) ]

layout = dict(
    title = 'Suicide rates average since the 90\'s',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
    
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='world-countries.json')
ratio2000 = suicides[mask2000][['Country', 'Male-Female ratio']].sort_values(by='Male-Female ratio', ascending=False)
ratio2005 = suicides[mask2005][['Country', 'Male-Female ratio']].sort_values(by='Male-Female ratio', ascending=False)
ratio2010 = suicides[mask2010][['Country', 'Male-Female ratio']].sort_values(by='Male-Female ratio', ascending=False)
ratio2015 = suicides[mask2015][['Country', 'Male-Female ratio']].sort_values(by='Male-Female ratio', ascending=False)
ratio2000 = ratio2000.replace([0, np.inf], np.nan)
ratio2005 = ratio2005.replace([0, np.inf], np.nan)
ratio2010 = ratio2010.replace([0, np.inf], np.nan)
ratio2015 = ratio2015.replace([0, np.inf], np.nan)
ratio2000 = ratio2000.dropna()
ratio2005 = ratio2005.dropna()
ratio2010 = ratio2010.dropna()
ratio2015 = ratio2015.dropna()
data = [go.Bar(
    x = ratio2015['Country'].head(30),
    y = ratio2015['Male-Female ratio'].head(30),
    marker=dict(color='rgb(255, 210, 119)', line=dict(color='rgb(206, 142, 12)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the highest male-female suicides ratio in 2015',
    xaxis={'title':'Countries'},
    yaxis={'title':'Male-Female suicide rate ratio'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [go.Bar(
    x = ratio2015['Country'].tail(30),
    y = ratio2015['Male-Female ratio'].tail(30),
    marker=dict(color='rgb(198, 141, 227)', line=dict(color='rgb(127, 20, 181)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the lowest male-female suicides ratio in 2015',
    xaxis={'title':'Countries'},
    yaxis={'title':'Male-Female suicide rate ratio'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [go.Bar(
    x = ratio2010['Country'].head(30),
    y = ratio2010['Male-Female ratio'].head(30),
    marker=dict(color='rgb(255, 210, 119)', line=dict(color='rgb(206, 142, 12)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the highest male-female suicides ratio in 2010',
    xaxis={'title':'Countries'},
    yaxis={'title':'Male-Female suicide rate ratio'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [go.Bar(
    x = ratio2010['Country'].tail(30),
    y = ratio2010['Male-Female ratio'].tail(30),
    marker=dict(color='rgb(198, 141, 227)', line=dict(color='rgb(127, 20, 181)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the lowest male-female suicides ratio in 2010',
    xaxis={'title':'Countries'},
    yaxis={'title':'Male-Female suicide rate ratio'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [go.Bar(
    x = ratio2005['Country'].head(30),
    y = ratio2005['Male-Female ratio'].head(30),
    marker=dict(color='rgb(255, 210, 119)', line=dict(color='rgb(206, 142, 12)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the highest male-female suicides ratio in 2005',
    xaxis={'title':'Countries'},
    yaxis={'title':'Male-Female suicide rate ratio'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [go.Bar(
    x = ratio2005['Country'].tail(30),
    y = ratio2005['Male-Female ratio'].tail(30),
    marker=dict(color='rgb(198, 141, 227)', line=dict(color='rgb(127, 20, 181)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the lowest male-female suicides ratio in 2005',
    xaxis={'title':'Countries'},
    yaxis={'title':'Male-Female suicide rate ratio'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [go.Bar(
    x = ratio2000['Country'].head(30),
    y = ratio2000['Male-Female ratio'].head(30),
    marker=dict(color='rgb(255, 210, 119)', line=dict(color='rgb(206, 142, 12)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the highest male-female suicides ratio in 2000',
    xaxis={'title':'Countries'},
    yaxis={'title':'Male-Female suicide rate ratio'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
data = [go.Bar(
    x = ratio2000['Country'].tail(30),
    y = ratio2000['Male-Female ratio'].tail(30),
    marker=dict(color='rgb(198, 141, 227)', line=dict(color='rgb(127, 20, 181)', width=1.5)),opacity=0.6 
)]

layout = go.Layout(
    title='Top countries with the lowest male-female suicides ratio in 2000',
    xaxis={'title':'Countries'},
    yaxis={'title':'Male-Female suicide rate ratio'}
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
selected = ['Lithuania', 'Sri Lanka', 'Belarus', 'North Korea', 'Japan', 'South Korea', 'Poland', 'Belguim']
selectedMasks = []
for country in selected:
    mask = suicides['Country'].str.contains(country)
    selectedMasks.append(mask)
data = []
for i in range(len(selected)):
    
    trace = go.Scatter(
        x = ['2000', '2005', '2010', '2015'],
        y = suicides[selectedMasks[i]]['Total suicide rate'].values.tolist(),
        name = selected[i],
        line=dict(
            shape='spline'
        )
    )
    data.append(trace)
    
layout = go.Layout(
    title='Total Suicide Rates of the Selected Countries',
    xaxis=dict(
        title='Year',
        autotick=False,
        ticks='outside',
        tick0=0,
        dtick=5
    ),
    yaxis=dict(
        title='Total Suicide Rate',
        rangemode='tozero',
        dtick=5
    )
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='line-mode')
data = []
for i in range(len(selected)):
    
    trace = go.Scatter(
        x = ['2000', '2005', '2010', '2015'],
        y = suicides[selectedMasks[i]]['Male suicide rate'].values.tolist(),
        name = selected[i],
        line=dict(
            shape='spline'
        )
    )
    data.append(trace)
    
layout = go.Layout(
    title='Males Suicide Rates of the Selected Countries',
    xaxis=dict(
        title='Year',
        autotick=False,
        ticks='outside',
        tick0=0,
        dtick=5
    ),
    yaxis=dict(
        title='Males Suicide Rate',
        rangemode='tozero'
    )
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='line-mode')
data = []
for i in range(len(selected)):
    
    trace = go.Scatter(
        x = ['2000', '2005', '2010', '2015'],
        y = suicides[selectedMasks[i]]['Female suicide rate'].values.tolist(),
        name = selected[i],
        line=dict(
            shape='spline'
        )
    )  
    data.append(trace)
  
layout = go.Layout(
    title='Females Suicide Rates of the Selected Countries',
    xaxis=dict(
        title='Year',
        autotick=False,
        ticks='outside',
        tick0=0,
        dtick=5
    ),
    yaxis=dict(
        title='Females Suicide Rate',
        rangemode='tozero'
    )
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='line-mode')
data = []
for i in range(len(selected)):
    
    trace = go.Scatter(
        x = ['2000', '2005', '2010', '2015'],
        y = suicides[selectedMasks[i]]['Male-Female ratio'].values.tolist(),
        name = selected[i],
        line=dict(
            shape='spline'
        )
    )
    data.append(trace)

layout = go.Layout(
    title='Male-Female Suicide Rates Ratio in the Selected Countries',
    xaxis=dict(
        title='Year',
        autotick=False,
        ticks='outside',
        tick0=0,
        dtick=5
    ),
    yaxis=dict(
        title='Male-Female Ratio',
        rangemode='tozero'
    )
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='line-mode')