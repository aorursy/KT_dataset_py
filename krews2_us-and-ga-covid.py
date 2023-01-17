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
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly as py
import plotly.graph_objects as go
import ipywidgets as widgets
import plotly.express as px
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
cc_data = pd.read_csv('../input/uscovidcases/us-counties.csv' , parse_dates =['date'],
                     dtype={'fips': object})
pop_data = pd.read_csv('/kaggle/input/covid19-for-georgia/county population 2019.csv', encoding='latin-1')
cc_data.tail()
pop_data['CTYNAME'] = pop_data['CTYNAME'].map(lambda x: x.strip('County'))
pop_data['CTYNAME'] = pop_data['CTYNAME'].map(lambda x: x.strip())
pop_data.head()
df = cc_data.merge(pop_data,left_on=('county', 'state') ,right_on=('CTYNAME', 'STNAME'))
df = df.drop(columns=['county','state'])
df.tail()
df['datedata'] = df['date']

df.set_index(['date'], inplace=True)

df['cases'].plot()
df['Formatted POP 2019'] = df['POPESTIMATE2019'].apply(lambda x: "{:,}".format(x))
df.dtypes
cases_data = df[(df['STNAME']=='Georgia')]
cases_data = cases_data.sort_values(by=['cases','CTYNAME'], ascending=False)
fig =px.line(cases_data, x="datedata", y="cases", color ="CTYNAME", title ='''Georgia's Corona Cases by County''', 
        labels={'datedata':'Date','cases':'Cases','Formatted POP 2019':'Pop 2019 est.', 'CTYNAME':'County'}, hover_name='STNAME', hover_data=['Formatted POP 2019'])
fig.update_layout(
    title={
        'y':0.9,
        'x':0.5,
         'xanchor': 'center',
        })

fig.show()
death_data = df[(df['STNAME']=='Georgia')]
death_data = death_data.sort_values(by=['deaths','CTYNAME'], ascending=False)
fig2 =px.line(death_data, x="datedata", y="deaths", color ="CTYNAME", title ='''Georgia's Corona Death by County''',
        labels={'datedata':'Date','deaths':'Deaths','Formatted POP 2019':'Pop 2019 est.'}, hover_name='STNAME', hover_data=['Formatted POP 2019'])
fig2.update_layout(
    title={
        'y':0.9,
        'x':0.5,
         'xanchor': 'center',
        })
fig2.show()
GA = df[(df['STNAME']=='Georgia')]
table = pd.pivot_table(GA, values =['POPESTIMATE2019','cases'], index =['STNAME', 'CTYNAME'], 
                          aggfunc = np.max) 
table
table['Per 1000'] = table['cases']/(table['POPESTIMATE2019']/1000)

table = table.reset_index(level='CTYNAME')
table = table.sort_values(by='Per 1000', ascending=False)
fig3 = px.bar(table, x="CTYNAME", y="Per 1000",  title ='''Georgia's Cases by County''',
       width = 3000, color='cases', color_continuous_scale='Jet',
             labels={'CTYNAME':'County'})

fig3.update_layout(
    title={
        'y':0.9,
        'x':0.5,
         'xanchor': 'center',
        })

fig3.show()
tbldeaths = pd.pivot_table(GA, values =['deaths','cases'], index =['STNAME', 'CTYNAME'], 
                          aggfunc =np.max) 

tbldeaths['deaths%'] = (tbldeaths['deaths']/(tbldeaths['cases'])*100).round(1)

tbldeaths = tbldeaths[(tbldeaths['deaths']>0) & (tbldeaths['cases'] >10)]
tbldeaths = tbldeaths.reset_index(level='CTYNAME')
tbldeaths = tbldeaths.sort_values(by='deaths%', ascending=False)
fig4 = px.bar(tbldeaths, x="CTYNAME", y="deaths%",  title ='''Georgia's Deaths % by County''',
       width = 900, hover_data=['cases'], color='deaths', color_continuous_scale='Jet', 
             labels={'CTYNAME':'County'})

fig4.update_layout(
    title={
        'y':0.9,
        'x':0.5,
         'xanchor': 'center',
        }, yaxis=dict(tickformat=".%"),
          )
fig4.show()
cc_data['date'] = pd.to_datetime(cc_data['date'])
enddate = cc_data['date'].max()
cc_data = cc_data[cc_data['date']==enddate]
UStable = pd.pivot_table(cc_data, values =['deaths','cases'], index =['state'], 
                          aggfunc =np.sum) 

UStable['deaths%'] = (UStable['deaths']/(UStable['cases'])*100).round(1)

UStable = UStable[(UStable['deaths']>0) & (UStable['cases'] >50)]
UStable = UStable.reset_index(level='state')
UStable = UStable.sort_values(by='deaths%', ascending=False)


UStable.index = np.arange(1, len(UStable) + 1)
UStable.index.name = 'Rank'
UStable
fig5 = px.bar(UStable, x="state", y="deaths%",  title ='''US's Deaths % by State''',
       width = 900, hover_data=['cases'], color='deaths', color_continuous_scale='Jet', 
             labels={'state':'State'})

fig5.update_layout(
    title={
        'y':0.9,
        'x':0.5,
         'xanchor': 'center',
        }, yaxis=dict(tickformat=".%"),
          )
fig5.show()
state_pop = pd.read_csv('/kaggle/input/covid19-for-georgia/state pop 2019.csv')
state_pop.head()
fig6 = px.bar(UStable, x="state", y="cases",  title ='''US's Cases by State''',
       width = 1500, hover_data=['cases'], color='cases', color_continuous_scale='Jet', 
             labels={'state':'State'})

fig6.update_layout(
    title={
        'y':0.9,
        'x':0.5,
         'xanchor': 'center',
        }, yaxis=dict(tickformat=".%"),
          )
fig6.show()
dfstates = UStable.merge(state_pop,left_on=('state') ,right_on=('NAME'))
dfstates.drop(columns=['NAME'])
dfstates['Per 100k'] = (dfstates['cases']/(dfstates['POPESTIMATE2019']/100000)).round(0)
dfstates = dfstates.sort_values(by='Per 100k', ascending=False)

dfstates.index = np.arange(1, len(dfstates) + 1)
dfstates.index.name = 'Rank'
dfstates
fig7 = px.bar(dfstates, x="state", y="Per 100k",  title ='''US's Cases by State''',
       width = 1500, hover_data=['cases'], color='cases', color_continuous_scale='Jet', 
             labels={'state':'State'})

fig7.update_layout(
    title={
        'y':0.9,
        'x':0.5,
         'xanchor': 'center',
        }, yaxis=dict(tickformat=".%"),
          )

fig7.show()
from urllib.request import urlopen
import json
with open('/kaggle/input/covid19-for-georgia/geojson-counties-fips.json') as c:
    counties = json.load(c)

# Select which states you would like to see using the state name separated by commas
state_list = ['Georgia']


df_map = cc_data[cc_data['state'].isin(state_list)]

maxval = df_map['cases'].max()
if maxval <= 2500:
    maxval = maxval
else:
    maxval = 2500
    
import plotly.express as px

figmap = px.choropleth(df_map, geojson=counties, locations='fips' ,color='cases',
                           color_continuous_scale="OrRd",
                           range_color=(0, maxval),
                           scope="usa",
                           hover_name='county',
                           hover_data=['deaths']
                          )
figmap.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
figmap.show()