import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)



pd.set_option('display.max_colwidth', -1)
pax = pd.read_csv('../input/pax_20_02_2018_1_CSV.csv')

pax.head()
pax['Countries involved'] = pax['Con'].apply(str.split, args = '/')
pax['Year'] = pax['Dat'].apply(lambda date : int(date.split('/')[2]))

pax['Month'] = pax['Dat'].apply(lambda date : int(date.split('/')[1]))

pax['Day'] = pax['Dat'].apply(lambda date : int(date.split('/')[0]))
pax['isCor'] = pax['Cor'] >= 1

pax['isTerr'] = pax['Terr'] >= 1

pax['isCrOcr'] = pax['SsrCrOcr'] >= 1

pax['isDrugs'] = pax['SsrDrugs'] >= 1

pax['isCrime'] = pax['isDrugs'] + pax['isCrOcr'] + pax['isTerr'] + pax['isCor']
country_list = {}

for num, agreement in pax.iterrows():

    for country in agreement['Countries involved']:

        if country.strip('()') not in country_list: #What do the parenthesis mean in the DB?

            country_list[country.strip('()')] = 1

        else:

            country_list[country.strip('()')] += 1

countries_df = pd.DataFrame.from_dict(country_list, orient = 'index', columns=['Peace Treaties Signed'])



code_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')



def return_country_code(con):

    if con in code_df['COUNTRY'].values:

        return code_df[code_df['COUNTRY'] == con]['CODE'].values[0]



countries_df['Country'] = countries_df.index

countries_df['Code'] = countries_df['Country'].apply(return_country_code)

countries = countries_df.dropna()
data=dict(

    type = 'choropleth',

    locations = countries['Code'],

    z = countries['Peace Treaties Signed'],

    text = countries['Country'],

    colorscale = 'YlOrRd',

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'Peace Treaties Signed',

)



layout = dict(title_text='Peace Treaties since 1990',

    geo=dict(

        showframe=False,

        showcoastlines=True,

        projection_type='equirectangular'

    ))



fig = go.Figure(data = [data], layout = layout)

iplot(fig)
def return_stage(stage):

    return pax[pax['Stage'] == stage].groupby('Reg').count()['Con']

data = []

names = ['Implementation', 'Pre-negotiation', 'Partial', 'Comprehensive', 'Ceasefire', 'Renewal', 'Others']

abb = ['Imp', 'Pre', 'SubPar', 'SubComp', 'Cea', 'Ren', 'Oth']

for i in range(len(names)):

    data.append(go.Bar(name = names[i], x = return_stage(abb[i]).index, y = return_stage(abb[i]).values))



layout = dict(barmode = 'stack')



fig = go.Figure(data=data, layout = layout)

fig.show()
layout = dict(title_text='Mentions of Crimes in Peace Treaties by region',

             barmode = 'stack')

data = [

    go.Bar(name='Corruption', x=pax[pax['isCor'] == 1].groupby('Reg').count()['Con'].index, y=pax[pax['isCor'] == 1].groupby('Reg').count()['Con'].values),

    go.Bar(name='Terrorism', x=pax[pax['isTerr'] == 1].groupby('Reg').count()['Con'].index, y=pax[pax['isTerr'] == 1].groupby('Reg').count()['Con'].values),

    go.Bar(name='Organised Crime', x=pax[pax['isCrOcr'] == 1].groupby('Reg').count()['Con'].index, y=pax[pax['isCrOcr'] == 1].groupby('Reg').count()['Con'].values),

    go.Bar(name='Drugs', x=pax[pax['isDrugs'] == 1].groupby('Reg').count()['Con'].index, y=pax[pax['isDrugs'] == 1].groupby('Reg').count()['Con'].values),

]

fig = go.Figure(data=data, layout = layout)

fig.show()
layout = dict(title_text='Length of Peace Treaties by region',

             barmode = 'stack')

data = [

    go.Bar(name='Less than 2 pages', x=pax[pax['Lgt'] <= 2].groupby('Reg').count()['Con'].index, y=pax[pax['Lgt'] <= 2].groupby('Reg').count()['Con'].values),

    go.Bar(name='Between 2 and 6 pages', x=pax[(pax['Lgt'] > 2) & (pax['Lgt'] <= 6)].groupby('Reg').count()['Con'].index, y=pax[(pax['Lgt'] > 2) & (pax['Lgt'] <= 6)].groupby('Reg').count()['Con'].values),

    go.Bar(name='More than 6 pages', x=pax[pax['Lgt'] > 6].groupby('Reg').count()['Con'].index, y=pax[pax['Lgt'] > 6].groupby('Reg').count()['Con'].values)

]

fig = go.Figure(data=data, layout = layout)

fig.show()
pax[pax['Lgt'] > 6]

country_list = {}

for num, agreement in pax[pax['Lgt'] > 6].iterrows():

    for country in agreement['Countries involved']:

        if country.strip('()') not in country_list: 

            country_list[country.strip('()')] = 1

        else:

            country_list[country.strip('()')] += 1

countries_df = pd.DataFrame.from_dict(country_list, orient = 'index', columns=['Long Peace Treaties Signed'])



countries_df['Country'] = countries_df.index

countries_df['Code'] = countries_df['Country'].apply(return_country_code)

countries = countries_df.dropna()

data=dict(

    type = 'choropleth',

    locations = countries['Code'],

    z = countries['Long Peace Treaties Signed'],

    text = countries['Country'],

    colorscale = 'YlOrRd',

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'Long Peace Treaties Signed',

)



layout = dict(title_text='Countries that have signed long Peace Treaties since 1990',

              geo=dict(

        showframe=False,

        showcoastlines=True,

        projection_type='equirectangular'

    ))



fig = go.Figure(data = [data], layout = layout)

iplot(fig)
plt.figure(figsize = (12,4))

plt.subplot(121)

sns.distplot(pax['Year'], kde = False, bins = 25).set_title('Peace agreements per year')

plt.subplot(122)

sns.distplot(pax['Month'], kde = False).set_title('Peace agreements per month')
regions_evolution = pd.crosstab(pax.Year,pax.Reg)

regions_evolution.plot(color=sns.color_palette('Set2',12), title='Peace Agreements per region over time')

fig=plt.gcf()

fig.set_size_inches(18,6)

plt.show()
layout = dict(title_text='Mentions of Crime in Peace Treaties over time',

             barmode = 'stack')

data = [

    go.Bar(name='Corruption', x=pax[pax['isCor'] == 1].groupby('Year').count()['Con'].index, y=pax[pax['isCor'] == 1].groupby('Year').count()['Con'].values),

    go.Bar(name='Terrorism', x=pax[pax['isTerr'] == 1].groupby('Year').count()['Con'].index, y=pax[pax['isTerr'] == 1].groupby('Year').count()['Con'].values),

    go.Bar(name='Organised Crime', x=pax[pax['isCrOcr'] == 1].groupby('Year').count()['Con'].index, y=pax[pax['isCrOcr'] == 1].groupby('Year').count()['Con'].values),

    go.Bar(name='Drugs', x=pax[pax['isDrugs'] == 1].groupby('Year').count()['Con'].index, y=pax[pax['isDrugs'] == 1].groupby('Year').count()['Con'].values),

]

fig = go.Figure(data=data, layout = layout)

fig.show()
# Agreements with the largest number of countries involved

max = 0

for num, agreement in pax.iterrows():

    if len(agreement['Countries involved']) >= max:

        max = len(agreement['Countries involved'])

pax[pax['Countries involved'].apply(len) == max]
#Longest agreement in number of pages

max = 0

for num, agreement in pax.iterrows():

    if agreement['Lgt'] >= max:

        max = agreement['Lgt']

pax[pax['Lgt'] == max]