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



import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')



import seaborn as sns

import matplotlib as p

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.graph_objs as gobj

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)



import plotly.express as px       

import plotly.offline as py       

import plotly.graph_objects as go 

from plotly.subplots import make_subplots
pd.set_option('display.max_columns', 100)

measures = pd.read_csv('/kaggle/input/uncover/UNCOVER/HDE_update/acaps-covid-19-government-measures-dataset.csv')

measures.head(10)
pd.set_option('display.max_rows', 1800)

measures_gr = measures.groupby(['country','entry_date', 'category'])['category'].count().to_frame(name = 'count')

measures_gr = measures_gr.sort_values(by=['country', 'count'], ascending = True).reset_index()

measures_gr
pd.set_option('display.max_rows', 3000)

measures_gr2 = measures.groupby(['country','entry_date', 'category', 'measure'])['measure'].count().to_frame(name = 'count')

measures_gr2 = measures_gr2.sort_values(by=['country', 'count'], ascending = True).reset_index()

measures_gr2
pd.set_option('display.max_rows', 200)

measures_gr3 = measures.groupby(['entry_date', 'category'])['category'].count().to_frame(name = 'count')

measures_gr3 = measures_gr3.sort_values(by=['entry_date', 'count'], ascending = True).reset_index()

measures_gr3
pd.set_option('display.max_rows', 600)

measures_gr4 = measures.groupby(['entry_date', 'category', 'measure'])['measure'].count().to_frame(name = 'count')

measures_gr4 = measures_gr4.sort_values(by=['entry_date', 'count'], ascending = True).reset_index()

measures_gr4
pd.set_option('display.max_columns', 300)

# df2 = pd.read_csv('/kaggle/input/uncover/UNCOVER/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-deaths.csv')

status_df = pd.read_csv('/kaggle/input/uncover/UNCOVER/WHO/who-situation-reports-covid-19.csv')

status_df.head()
status_df_1 = status_df.groupby(['reporting_country_territory'])['total_deaths'].sum().to_frame(name = 'sum_total_deaths')

status_df_1 = status_df_1.sort_values(by=['sum_total_deaths'], ascending = False).reset_index()

status_df_1.head(30)
list_1 = status_df_1.reporting_country_territory.to_list()
status_df_2 = status_df.groupby(['reporting_country_territory'])['new_total_deaths'].sum().to_frame(name = 'sum_new_total_deaths')

status_df_2 = status_df_2.sort_values(by=['sum_new_total_deaths'], ascending = False).reset_index()

status_df_2.head(30)
list_2 = status_df_2.reporting_country_territory.to_list()
status_df_gr = status_df.groupby(['reporting_country_territory', 'reported_date'])['total_deaths'].sum().to_frame(name = 'sum_total_deaths')

status_df_gr = status_df_gr.sort_values(by=['reporting_country_territory', 'reported_date'], ascending = True).reset_index()

status_df_gr.head(50)
status_df_gr2 = status_df.groupby(['reporting_country_territory', 'reported_date'])['new_total_deaths'].sum().to_frame(name = 'sum_new_total_deaths')

status_df_gr2 = status_df_gr2.sort_values(by=['reporting_country_territory', 'reported_date'], ascending = True).reset_index()

status_df_gr2.head(50)
status_df_gr3 = status_df.groupby(['reported_date'])['total_deaths'].sum().to_frame(name = 'sum_total_deaths')

status_df_gr3 = status_df_gr3.sort_values(by=['reported_date'], ascending = True).reset_index()

status_df_gr3.head(50)
status_df_gr4 = status_df.groupby(['reported_date'])['new_total_deaths'].sum().to_frame(name = 'sum_new_total_deaths')

status_df_gr4 = status_df_gr4.sort_values(by=['reported_date'], ascending = True).reset_index()

status_df_gr4.head(50)
measures_gr5 = measures_gr3.groupby('entry_date')['count'].sum().to_frame(name = 'sum')

measures_gr5 = measures_gr5.sort_values(by=['entry_date']).reset_index()
# Added secondary axis

fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=measures_gr5['entry_date'], y=measures_gr5['sum'], name='Sum of measures implemented', marker_color='rgb(60, 83, 109)'), secondary_y=True)

fig.add_trace(go.Bar(x=status_df_gr3['reported_date'],y=status_df_gr3['sum_total_deaths'],name='Total Deaths by Reported Date',marker_color='rgb(70, 118, 255)'))



fig.update_layout(title='Measures Implemented and Total Deaths',xaxis_tickfont_size=14,

                  yaxis=dict(title='Sum',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)

fig.show()
# Added secondary axis

fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=measures_gr5['entry_date'], y=measures_gr5['sum'], name='Sum of measures implemented', marker_color='rgb(60, 83, 109)'), secondary_y=True)

fig.add_trace(go.Bar(x=status_df_gr4['reported_date'],y=status_df_gr4['sum_new_total_deaths'],name='New Total Deaths by Reported Date',marker_color='rgb(70, 118, 255)'))



fig.update_layout(title='Measures Implemented and New Total Deaths',xaxis_tickfont_size=14,

                  yaxis=dict(title='Sum',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)

fig.show()
measures_china = measures_gr[measures_gr['country'] == 'China'] .sort_values('entry_date').reset_index()

measures_china2 = measures_gr2[measures_gr2['country'] == 'China'] .sort_values('entry_date').reset_index()

display(measures_china)

display(measures_china2)



status_china = status_df_gr[status_df_gr['reporting_country_territory'] == 'China']

status_china2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'China']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_china, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('China - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_china2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('China - New Total Deaths', size = 14)

plt.show()
measures_italy = measures_gr[measures_gr['country'] == 'Italy'] .sort_values('entry_date').reset_index()

measures_italy2 = measures_gr2[measures_gr2['country'] == 'Italy'] .sort_values('entry_date').reset_index()

display(measures_italy)

display(measures_italy2)



status_italy = status_df_gr[status_df_gr['reporting_country_territory'] == 'Italy']

status_italy2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Italy']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_italy, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Italy - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_italy2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Italy - New Total Deaths', size = 14)

plt.show()
measures_spain = measures_gr[measures_gr['country'] == 'Spain'] .sort_values('entry_date').reset_index()

measures_spain2 = measures_gr2[measures_gr2['country'] == 'Spain'] .sort_values('entry_date').reset_index()

display(measures_spain)

display(measures_spain2)



status_spain = status_df_gr[status_df_gr['reporting_country_territory'] == 'Spain']

status_spain2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Spain']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_spain, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Spain - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_spain2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Spain - New Total Deaths', size = 14)

plt.show()
measures_iran = measures_gr[measures_gr['country'] == 'Iran'] .sort_values('entry_date').reset_index()

measures_iran2 = measures_gr2[measures_gr2['country'] == 'Iran'] .sort_values('entry_date').reset_index()

display(measures_iran)

display(measures_iran2)



status_iran = status_df_gr[status_df_gr['reporting_country_territory'] == 'Iran (Islamic Republic of)']

status_iran2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Iran (Islamic Republic of)']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_iran, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Iran - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_iran2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Iran - New Total Deaths', size = 14)

plt.show()
measures_france = measures_gr[measures_gr['country'] == 'France'] .sort_values('entry_date').reset_index()

measures_france2 = measures_gr2[measures_gr2['country'] == 'France'] .sort_values('entry_date').reset_index()

display(measures_france)

display(measures_france2)



status_france = status_df_gr[status_df_gr['reporting_country_territory'] == 'France']

status_france2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'France']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_france, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('France - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_france2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('France - New Total Deaths', size = 14)

plt.show()
measures_usa = measures_gr[measures_gr['country'] == 'United States of America'] .sort_values('entry_date').reset_index()

measures_usa2 = measures_gr2[measures_gr2['country'] == 'United States of America'] .sort_values('entry_date').reset_index()

display(measures_usa)

display(measures_usa2)



status_usa = status_df_gr[status_df_gr['reporting_country_territory'] == 'United States of America']

status_usa2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'United States of America']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_usa, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('United States of America - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_usa2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('United States of America - New Total Deaths', size = 14)

plt.show()

measures_uk = measures_gr[measures_gr['country'] == 'United Kingdom'].sort_values('entry_date').reset_index()

measures_uk2 = measures_gr2[measures_gr2['country'] == 'United Kingdom'] .sort_values('entry_date').reset_index()

display(measures_uk)

display(measures_uk2)



status_uk = status_df_gr[status_df_gr['reporting_country_territory'] == 'United Kingdom']

status_uk2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'United Kingdom']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_uk, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('United Kingdom - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_uk2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('United Kingdom  - New Total Deaths', size = 14)

plt.show()
measures_net = measures_gr[measures_gr['country'] == 'Netherlands'].sort_values('entry_date').reset_index()

measures_net2 = measures_gr2[measures_gr2['country'] == 'Netherlands'] .sort_values('entry_date').reset_index()

display(measures_net)

display(measures_net2)



status_net = status_df_gr[status_df_gr['reporting_country_territory'] == 'Netherlands']

status_net2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Netherlands']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_net, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Netherlands - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_net2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Netherlands  - New Total Deaths', size = 14)

plt.show()
measures_ger = measures_gr[measures_gr['country'] == 'Germany'].sort_values('entry_date').reset_index()

measures_ger2 = measures_gr2[measures_gr2['country'] == 'Germany'] .sort_values('entry_date').reset_index()

display(measures_ger)

display(measures_ger2)



status_ger = status_df_gr[status_df_gr['reporting_country_territory'] == 'Germany']

status_ger2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Germany']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_ger, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Germany - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_ger2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Germany  - New Total Deaths', size = 14)

plt.show()
measures_bel = measures_gr[measures_gr['country'] == 'Belgium'].sort_values('entry_date').reset_index()

measures_bel2 = measures_gr2[measures_gr2['country'] == 'Belgium'] .sort_values('entry_date').reset_index()

display(measures_bel)

display(measures_bel2)



status_bel = status_df_gr[status_df_gr['reporting_country_territory'] == 'Belgium']

status_bel2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Belgium']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_bel, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Belgium - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_bel2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Belgium  - New Total Deaths', size = 14)

plt.show()
measures_korea = measurmeasures_korea = measures_gr[measures_gr['country'] == 'Korea Republic of'].sort_values('entry_date').reset_index()

measures_korea2 = measures_gr2[measures_gr2['country'] == 'Korea Republic of'] .sort_values('entry_date').reset_index()

display(measures_korea)

display(measures_korea2)



status_korea = status_df_gr[status_df_gr['reporting_country_territory'] == 'Republic of Korea']

status_korea2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Republic of Korea']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_korea, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Republic of Korea - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_korea2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Republic of Korea  - New Total Deaths', size = 14)

plt.show()
measures_swit = measures_gr[measures_gr['country'] == 'Switzerland'].sort_values('entry_date').reset_index()

measures_swit2 = measures_gr2[measures_gr2['country'] == 'Switzerland'] .sort_values('entry_date').reset_index()

display(measures_swit)

display(measures_swit2)



status_swit = status_df_gr[status_df_gr['reporting_country_territory'] == 'Switzerland']

status_swit2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Switzerland']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_swit, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Switzerland - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_swit2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Switzerland  - New Total Deaths', size = 14)

plt.show()
measures_turkey = measures_gr[measures_gr['country'] == 'Turkey'].sort_values('entry_date').reset_index()

measures_turkey2 = measures_gr2[measures_gr2['country'] == 'Turkey'].sort_values('entry_date').reset_index()

display(measures_turkey)

display(measures_turkey2)



status_turkey = status_df_gr[status_df_gr['reporting_country_territory'] == 'Turkey']

status_turkey2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Turkey']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_turkey, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Turkey - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_turkey2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Turkey - New Total Deaths', size = 14)

plt.show()
measures_brazil = measures_gr[measures_gr['country'] == 'Brazil'].sort_values('entry_date').reset_index()

measures_brazil2 = measures_gr2[measures_gr2['country'] == 'Brazil'].sort_values('entry_date').reset_index()

display(measures_brazil)

display(measures_brazil2)



status_brazil = status_df_gr[status_df_gr['reporting_country_territory'] == 'Brazil']

status_brazil2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Brazil']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_brazil, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Brazil - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_brazil2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Brazil - New Total Deaths', size = 14)

plt.show()
measures_sweden = measures_gr[measures_gr['country'] == 'Sweden'].sort_values('entry_date').reset_index()

measures_sweden2 = measures_gr2[measures_gr2['country'] == 'Sweden'].sort_values('entry_date').reset_index()

display(measures_sweden)

display(measures_sweden2)



status_sweden = status_df_gr[status_df_gr['reporting_country_territory'] == 'Sweden']

status_sweden2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Sweden']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_sweden, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Sweden - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_sweden2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Sweden - New Total Deaths', size = 14)

plt.show()
measures_indonesia = measures_gr[measures_gr['country'] == 'Indonesia'].sort_values('entry_date').reset_index()

measures_indonesia2 = measures_gr2[measures_gr2['country'] == 'Indonesia'].sort_values('entry_date').reset_index()

display(measures_indonesia)

display(measures_indonesia2)



status_indonesia = status_df_gr[status_df_gr['reporting_country_territory'] == 'Indonesia']

status_indonesia2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Indonesia']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_indonesia, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Indonesia - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_indonesia2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Indonesia - New Total Deaths', size = 14)

plt.show()
measures_afghanistan = measures_gr[measures_gr['country'] == 'Afghanistan'].sort_values('entry_date').reset_index()

measures_afghanistan2 = measures_gr2[measures_gr2['country'] == 'Afghanistan'].sort_values('entry_date').reset_index()

display(measures_afghanistan)

display(measures_afghanistan2)



status_afghanistan = status_df_gr[status_df_gr['reporting_country_territory'] == 'Afghanistan']

status_afghanistan2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Afghanistan']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_afghanistan, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Afghanistan - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_afghanistan2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Afghanistan - New Total Deaths', size = 14)

plt.show()
measures_albania = measures_gr[measures_gr['country'] == 'Albania'].sort_values('entry_date').reset_index()

measures_albania2 = measures_gr2[measures_gr2['country'] == 'Albania'].sort_values('entry_date').reset_index()

display(measures_albania)

display(measures_albania2)



status_albania = status_df_gr[status_df_gr['reporting_country_territory'] == 'Albania']

status_albania2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Albania']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_albania, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Albania - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_albania2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Albania - New Total Deaths', size = 14)

plt.show()
measures_algeria = measures_gr[measures_gr['country'] == 'Algeria'].sort_values('entry_date').reset_index()

measures_algeria2 = measures_gr2[measures_gr2['country'] == 'Algeria'].sort_values('entry_date').reset_index()

display(measures_algeria)

display(measures_algeria2)



status_algeria = status_df_gr[status_df_gr['reporting_country_territory'] == 'Algeria']

status_algeria2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Algeria']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_algeria, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Algeria - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_algeria2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Algeria - New Total Deaths', size = 14)

plt.show()
measures_angola = measures_gr[measures_gr['country'] == 'Angola'] .sort_values('entry_date').reset_index()

measures_angola2 = measures_gr2[measures_gr2['country'] == 'Angola'] .sort_values('entry_date').reset_index()

display(measures_angola)

display(measures_angola2)



status_angola = status_df_gr[status_df_gr['reporting_country_territory'] == 'Angola']

status_angola2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Angola']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_angola, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Angola - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_angola2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 45,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Angola - New Total Deaths', size = 14)

plt.show()
measures_portugal = measures_gr[measures_gr['country'] == 'Portugal'].sort_values('entry_date').reset_index()

measures_portugal2 = measures_gr2[measures_gr2['country'] == 'Portugal'].sort_values('entry_date').reset_index()

display(measures_portugal)

display(measures_portugal2)



status_portugal = status_df_gr[status_df_gr['reporting_country_territory'] == 'Portugal']

status_portugal2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Portugal']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_portugal, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Portugal - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_portugal2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Portugal - New Total Deaths', size = 14)

plt.show()
measures_argentina = measures_gr[measures_gr['country'] == 'Argentina'] .sort_values('entry_date').reset_index()

measures_argentina2 = measures_gr2[measures_gr2['country'] == 'Argentina'] .sort_values('entry_date').reset_index()

display(measures_argentina)

display(measures_argentina2)



status_argentina = status_df_gr[status_df_gr['reporting_country_territory'] == 'Argentina']

status_argentina2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Argentina']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_argentina, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Argentina - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_argentina2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Argentina - New Total Deaths', size = 14)

plt.show()
measures_armenia = measures_gr[measures_gr['country'] == 'Armenia'] .sort_values('entry_date').reset_index()

measures_armenia2 = measures_gr2[measures_gr2['country'] == 'Armenia'] .sort_values('entry_date').reset_index()

display(measures_armenia)

display(measures_armenia2)



status_armenia = status_df_gr[status_df_gr['reporting_country_territory'] == 'Armenia']

status_armenia2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Armenia']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_armenia, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Armenia - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_armenia2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Armenia - New Total Deaths', size = 14)

plt.show()
measures_australia = measures_gr[measures_gr['country'] == 'Australia'] .sort_values('entry_date').reset_index()

measures_australia2 = measures_gr2[measures_gr2['country'] == 'Australia'] .sort_values('entry_date').reset_index()

display(measures_australia)

display(measures_australia2)



status_australia = status_df_gr[status_df_gr['reporting_country_territory'] == 'Australia']

status_australia2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Australia']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_australia, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Australia - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_australia2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Australia - New Total Deaths', size = 14)

plt.show()
measures_austria = measures_gr[measures_gr['country'] == 'Austria'] .sort_values('entry_date').reset_index()

measures_austria2 = measures_gr2[measures_gr2['country'] == 'Austria'] .sort_values('entry_date').reset_index()

display(measures_austria)

display(measures_austria2)



status_austria = status_df_gr[status_df_gr['reporting_country_territory'] == 'Austria']

status_austria2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Austria']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_austria, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Austria - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_austria2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Austria - New Total Deaths', size = 14)

plt.show()
measures_japan = measures_gr[measures_gr['country'] == 'Japan'].sort_values('entry_date').reset_index()

measures_japan2 = measures_gr2[measures_gr2['country'] == 'Japan'].sort_values('entry_date').reset_index()

display(measures_japan)

display(measures_japan2)



status_japan = status_df_gr[status_df_gr['reporting_country_territory'] == 'Japan']

status_japan2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Japan']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_japan, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Japan - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_japan2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Japan - New Total Deaths', size = 14)

plt.show()
measures_azerbaijan = measures_gr[measures_gr['country'] == 'Azerbaijan'] .sort_values('entry_date').reset_index()

measures_azerbaijan2 = measures_gr2[measures_gr2['country'] == 'Azerbaijan'] .sort_values('entry_date').reset_index()

display(measures_azerbaijan)

display(measures_azerbaijan2)



status_azerbaijan = status_df_gr[status_df_gr['reporting_country_territory'] == 'Azerbaijan']

status_azerbaijan2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Azerbaijan']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_azerbaijan, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Azerbaijan - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_azerbaijan2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Azerbaijan - New Total Deaths', size = 14)

plt.show()
measures_bahamas = measures_gr[measures_gr['country'] == 'Bahamas'] .sort_values('entry_date').reset_index()

measures_bahamas2 = measures_gr2[measures_gr2['country'] == 'Bahamas'] .sort_values('entry_date').reset_index()

display(measures_bahamas)

display(measures_bahamas2)



status_bahamas = status_df_gr[status_df_gr['reporting_country_territory'] == 'Bahamas']

status_bahamas2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Bahamas']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_bahamas, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Bahamas - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_bahamas2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Bahamas - New Total Deaths', size = 14)

plt.show()
measures_bahrain = measures_gr[measures_gr['country'] == 'Bahrain'] .sort_values('entry_date').reset_index()

measures_bahrain2 = measures_gr2[measures_gr2['country'] == 'Bahrain'] .sort_values('entry_date').reset_index()

display(measures_bahrain)

display(measures_bahrain2)



status_bahrain = status_df_gr[status_df_gr['reporting_country_territory'] == 'Bahrain']

status_bahrain2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Bahrain']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_bahrain, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Bahrain - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_bahrain2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Bahrain - New Total Deaths', size = 14)

plt.show()
measures_bangladesh = measures_gr[measures_gr['country'] == 'Bangladesh'] .sort_values('entry_date').reset_index()

measures_bangladesh2 = measures_gr2[measures_gr2['country'] == 'Bangladesh'] .sort_values('entry_date').reset_index()

display(measures_bangladesh)

display(measures_bangladesh2)



status_bangladesh = status_df_gr[status_df_gr['reporting_country_territory'] == 'Bangladesh']

status_bangladesh2 = status_df_gr2[status_df_gr2['reporting_country_territory'] == 'Bangladesh']
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_total_deaths", data=status_bangladesh, c = 'red', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Bangladesh - Total Deaths', size = 14)

plt.show()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x="reported_date", y="sum_new_total_deaths", data=status_bangladesh2, c = 'orange', linewidth = 3)

plt.xlabel("reported_date", size=14)

plt.ylabel("sum_new_total_deaths", size=14)

plt.tick_params(axis='x', rotation = 90,  labelsize = 12)

plt.tick_params(axis='y', labelsize = 12) 

plt.title('Bangladesh - New Total Deaths', size = 14)

plt.show()
status_df_1
status_df_2