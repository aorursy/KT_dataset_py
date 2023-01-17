import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from datetime import datetime
import pandas_datareader.data as web

import seaborn as sns
import matplotlib.pyplot as plt

import pycountry
import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#UserInfo.tsv
business_demography = pd.read_csv('../input/european-business-demography.tsv', delimiter='\t', encoding='utf-8')
#print(list(business_by_demography.columns.values)) #file header
business_demography.columns
business_demography = business_demography.rename(columns={'GEO,LEG_FORM,INDIC_SB,NACE_R2\TIME':'Countries'})
#Fill empty cells with zeros
business_demography = business_demography.fillna(0)
#Set countries as new dataframe index
business_demography.set_index('Countries',inplace=True)
#Transpose the table
business_demography = business_demography.transpose()
flags = ['b', 'c', 'd','e','f','i','n','p','r','s','u','z']
for column in business_demography.columns:  
    column = str(column)
    for flag in flags:
        business_demography[column] = business_demography[column].str.strip(flag)
for column in business_demography.columns:  
    column = str(column)
    column_content = column.split(',')[0]
    if 'Germany' in column_content:
        old_german_name = 'Germany (until 1990 former territory of the FRG)'
        new_german_name = 'Germany'
        new_column = column.replace(old_german_name,new_german_name)
        business_demography = business_demography.rename(index=str,columns={column:new_column})
business_demography = business_demography.replace(':',0)
business_demography =  business_demography.apply(pd.to_numeric, args=('coerce',))
business_demography = business_demography.fillna(0)
business_demography.drop(list(business_demography.filter(regex = 'European Union')), axis = 1, inplace = True)
business_demography.drop('2006').drop('2007').drop('2008')
countries_unique = []
for column in business_demography.columns:
    column_content = str(column).split(",")
    countries_unique.append(column_content[0])
countries_unique = list(set(countries_unique))
sole = business_demography['Germany,Sole proprietorship,Net business population growth - percentage,Business economy except activities of holding companies']    
total = business_demography['Germany,Total,Net business population growth - percentage,Business economy except activities of holding companies']
partnership = business_demography['Germany,Partnership, co-operatives, associations, etc.,Net business population growth - percentage,Business economy except activities of holding companies']
limiteds = business_demography['Germany,Limited liability enterprise,Net business population growth - percentage,Business economy except activities of holding companies']



sole_proprietorship = go.Scatter(x=sole.index, y=sole, name='Sole proprietorship')
total_growth = go.Scatter(x=total.index, y=total, name='Total growth')
partnerships = go.Scatter(x=partnership.index, y=partnership, name='Partnerships & Associates')
limiteds = go.Scatter(x=limiteds.index, y=limiteds, name='Limited liability')

data = [sole_proprietorship,total_growth,partnerships,limiteds]

layout = dict(
    title = "Growth comparison of different legal forms in Germany",
    xaxis = dict(
        range = ['2009','2015'])
)

fig = dict(data=data, layout=layout)
py.iplot(fig)
#Addin new column to dataframe which contains the average of the particular row's values
business_demography['Sole proprietorship growth'] = business_demography.filter(regex='Sole proprietorship,Net business population growth', axis=1).mean(axis=1)
business_demography['Total growth in the EU'] = business_demography.filter(regex='Total,Net business population growth', axis=1).mean(axis=1)
business_demography['Partnership growth'] = business_demography.filter(regex='Partnership, co-operatives, associations, etc.,Net business population growth', axis=1).mean(axis=1)
business_demography['Limiteds growth'] = business_demography.filter(regex='Limited liability enterprise,Net business population growth', axis=1).mean(axis=1)



sole_proprietorship = go.Scatter(x=business_demography.index, y=business_demography['Sole proprietorship growth'], name='Sole proprietorship')
total_growth = go.Scatter(x=business_demography.index, y=business_demography['Total growth in the EU'], name='Total growth')
partnerships = go.Scatter(x=business_demography.index, y=business_demography['Partnership growth'], name='Partnerships & Associates')
limiteds = go.Scatter(x=business_demography.index, y=business_demography['Limiteds growth'], name='Limited liability')

data = [sole_proprietorship,total_growth,partnerships,limiteds]

layout = dict(
    title = "Growth comparison of different legal forms across the EU",
    xaxis = dict(
        range = ['2009','2015'])
)

fig = dict(data=data, layout=layout)
py.iplot(fig)
import seaborn as sns
import matplotlib.pyplot as plt

eu_growth_rates = business_demography.iloc[:,-4:]

plt.figure(figsize=(10,10))
plt.title('Pearson correlation legal forms across the EU', y=1.05, size=15)
plot = sns.heatmap(eu_growth_rates.corr(), xticklabels=eu_growth_rates.columns, yticklabels=eu_growth_rates.columns,  linewidths=0.1,vmax=1.0,vmin=0, square=True, linecolor='white')
#plot.set_xticklabels(plot.get_xticklabels(),rotation=30)
plt.setp(plot.get_xticklabels(), rotation=45)
plt.setp(plot.get_yticklabels(), rotation=0)
plt.show()
from pycountry import * 

mean_growth = []
country_names = []
country_codes = []
# Mapping function to add the corresponding country (alpha3) country code
mapping = {country.name: country.alpha_3 for country in pycountry.countries}

for column in business_demography.columns:
    if 'Total,Net business population growth' in column:
        new_column_name = column.split(',')[0]
        #country_names.append(new_column_name)
        mean_growth.append(business_demography[column].mean())
        country_codes.append((new_column_name, mapping.get(new_column_name, 'No country found')))

mean_growth_country = pd.DataFrame(np.column_stack([country_codes,mean_growth]), columns=['Country', 'Code', 'Mean growth'])
data = [go.Bar(
            x = mean_growth_country['Country'],
            y = mean_growth_country['Mean growth']
        
    )]
layout = go.Layout(
    title='Business growth between 2009 and 2015',
)

fig = go.Figure(layout=layout)
py.iplot(data, filename='basic-bar')

data = [ dict(
        type = 'choropleth',
        locations = mean_growth_country['Code'],
        z = mean_growth_country['Mean growth'],
        text = mean_growth_country['Country'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '%',
            title = 'Average growth of business population'),
      ) ]

layout = dict(
    title = 'Average growth of business population in the years 2009-2015',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )

