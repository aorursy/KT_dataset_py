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
full_table = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

temp = full_table.groupby('ObservationDate')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
temp = temp[temp['ObservationDate']==max(temp['ObservationDate'])].reset_index(drop=True)
temp.style.background_gradient(cmap='Pastel1')
import plotly.express as px

temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
temp = temp[temp['Country/Region'] == 'France']

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan
temp['Deaths_per_mln'] = temp['Deaths']/67000000*1000000

temp = temp[temp['ObservationDate']>'03/08/2020']

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Deaths_per_mln", color='Country/Region',title='New Deaths_per_mln France',)
fig.update_layout(showlegend=False)
fig.show()


temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
temp = temp[temp['Country/Region'] == 'Sweden']
temp['Deaths_per_mln'] = temp['Deaths']/10230000*1000000

temp = temp[temp['ObservationDate']>'03/08/2020']


mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Deaths_per_mln", color='Country/Region',title='New Deaths_per_mln Sweden',)
fig.update_layout(showlegend=False)
fig.show()

###
temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
temp = temp[temp['Country/Region'] == 'US']
temp['Deaths_per_mln'] = temp['Deaths']/322000000*1000000

temp = temp[temp['ObservationDate']>'03/08/2020']


mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Deaths_per_mln", color='Country/Region',title='New Deaths_per_mln US',)
fig.update_layout(showlegend=False)
fig.show()

temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
temp = temp[temp['Country/Region'] == 'Ukraine']
temp['Deaths_per_mln'] = temp['Deaths']/37000000*1000000

temp = temp[temp['ObservationDate']>'03/08/2020']


mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Deaths_per_mln", color='Country/Region',title='New Deaths_per_mln Ukraine',)
fig.update_layout(showlegend=False)
fig.show()

temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
temp = temp[temp['Country/Region'] == 'Czech Republic']
temp['Deaths_per_mln'] = temp['Deaths']/10650000*1000000

temp = temp[temp['ObservationDate']>'03/08/2020']


mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Deaths_per_mln", color='Country/Region',title='New Deaths_per_mln Czech Republic',)
fig.update_layout(showlegend=False)
fig.show()
import plotly.express as px
temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Confirmed", color='Country/Region',title='New cases',)
fig.update_layout(showlegend=False)
fig.show()

###
temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
temp = temp[temp['Country/Region'] != 'US']

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Confirmed", color='Country/Region',title='New cases WO USA',)
fig.update_layout(showlegend=False)
fig.show()
###

###
temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
temp = temp[temp['Country/Region'] != 'US']
temp = temp[temp['Country/Region'] != 'France']

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Confirmed", color='Country/Region',title='New cases WO USA and France',)
fig.update_layout(showlegend=False)
fig.show()
###

temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
temp = temp[temp['Country/Region'] == 'Ukraine']

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Confirmed", color='Country/Region',title='New cases Ukraine',
            )
fig.update_layout(showlegend=False)
fig.show()

##
temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']
temp = temp.sum().diff().reset_index()
temp = temp[temp['Country/Region'] == 'Sweden']

mask = temp['Country/Region'] != temp['Country/Region'].shift(1)

temp.loc[mask, 'Confirmed'] = np.nan
temp.loc[mask, 'Deaths'] = np.nan
temp.loc[mask, 'Recovered'] = np.nan

fig = px.bar(temp.sort_values('ObservationDate'), 
             x="ObservationDate", y="Confirmed", color='Country/Region',title='New cases Sweden',
            )
fig.update_layout(showlegend=False)
fig.show()
temp = full_table.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Confirmed', ascending=False)

fig = px.line(temp.sort_values('ObservationDate'), x="ObservationDate", y="Confirmed", color='Country/Region', title='Cases Spread',)
fig.update_layout(showlegend=False)
fig.show()

temp = full_table.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Confirmed', ascending=False)
temp = temp[temp['Country/Region'] == 'Ukraine']
fig = px.line(temp.sort_values('ObservationDate'), x="ObservationDate", y="Confirmed", color='Country/Region', title='Cases Spread Ukraine',)
fig.update_layout(showlegend=False)
fig.show()
gdf = full_table.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths','Recovered'].max()
gdf = gdf.reset_index()

temp = gdf[gdf['Country/Region']=='Mainland China'].reset_index()
temp = temp.melt(id_vars='ObservationDate', value_vars=['Confirmed', 'Deaths','Recovered'],
                var_name='Case', value_name='Count')
fig = px.bar(temp, x="ObservationDate", y="Count", color='Case', facet_col="Case",
            title='China'#, color_discrete_sequence=[cnf, dth, rec]
            )
fig.show()

temp = gdf[gdf['Country/Region']!='Mainland China'].groupby('ObservationDate').sum().reset_index()
temp = temp.melt(id_vars='ObservationDate', value_vars=['Confirmed', 'Deaths','Recovered'],
                var_name='Case', value_name='Count')
fig = px.bar(temp, x="ObservationDate", y="Count", color='Case', facet_col="Case",
             title='ROW'#, color_discrete_sequence=[cnf, dth, rec]
            )
fig.show()

temp = gdf[gdf['Country/Region']=='Ukraine'].groupby('ObservationDate').sum().reset_index()
temp = temp.melt(id_vars='ObservationDate', value_vars=['Confirmed', 'Deaths','Recovered'],
                var_name='Case', value_name='Count')
fig = px.bar(temp, x="ObservationDate", y="Count", color='Case', facet_col="Case",
             title='Ukraine'#, color_discrete_sequence=[cnf, dth, rec]
            )
fig.show()
gdf[gdf['Country/Region']=='Sweden']
gdf['Active'] = gdf['Confirmed'] - gdf['Deaths'] - gdf['Recovered']

gdfU = gdf[gdf['Country/Region']=='Sweden']
gdfU.loc[:,'Active_new'] = (gdfU.loc[:,'Active'] - gdfU.loc[:,'Active'].shift(1).replace(np.nan,0))

fig = px.bar(gdfU.sort_values('ObservationDate'), 
             x="ObservationDate", y="Active_new", color='Country/Region',title='New Active Sweden',
            )
fig.update_layout(showlegend=False)
fig.show()
gdf[gdf['Country/Region']=='Ukraine']
gdf['Active'] = gdf['Confirmed'] - gdf['Deaths'] - gdf['Recovered']

gdfU = gdf[gdf['Country/Region']=='Ukraine']
gdfU.loc[:,'Active_new'] = (gdfU.loc[:,'Active'] - gdfU.loc[:,'Active'].shift(1).replace(np.nan,0))

fig = px.bar(gdfU.sort_values('ObservationDate'), 
             x="ObservationDate", y="Active_new", color='Country/Region',title='New Active Ukraine',
            )
fig.update_layout(showlegend=False)
fig.show()
def add_colomns(table):
    table.loc[:,'New cases'] = (table.loc[:,'Confirmed'] 
                                - table.loc[:,'Confirmed'].shift(1).replace(np.nan,0))
    
    table.loc[:,'avg_back'] = (table.loc[:,'New cases']
                               + table.loc[:,'New cases'].shift(1)
                               .replace(np.nan,table.loc[:,'New cases']))/2
    
    table.loc[:,'avg_forward'] = (table.loc[:,'New cases'] 
                                  + table.loc[:,'New cases'].shift(-1)
                                  .replace(np.nan,table.loc[:,'New cases'])
                                  )/2
                              
    table.loc[:,'Smooth increase'] = (table.loc[:,'avg_back'] + table.loc[:,'avg_forward'])/2
    return table.drop(['avg_back', 'avg_forward'], axis=1)
counries_group = full_table.groupby(['ObservationDate','Country/Region']
                                   )['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
counries_group_new_cases = pd.DataFrame()
for country in counries_group['Country/Region'].unique():
    temp_table = counries_group[counries_group['Country/Region'] == country]
    temp_table = add_colomns(temp_table)
    counries_group_new_cases = pd.concat([counries_group_new_cases, temp_table])
counries_group_new_cases.head()#['Country/Region'].unique()
import matplotlib.pyplot as plt

contry = counries_group_new_cases.loc[counries_group_new_cases['Country/Region'] == 'Ukraine']

fig, ax = plt.subplots(figsize =(15, 8))

index = contry.ObservationDate
bar_width = 0.2

rects1 = plt.bar(index, contry['Smooth increase'],
label='Smooth increase',
)

rects2 = plt.bar(index, contry['New cases'], bar_width,
label='New cases')

plt.legend(title='Ukraine')
plt.xticks(index, fontsize=10, rotation=30)

plt.tight_layout()
plt.show()
cg = counries_group_new_cases
cg['Death_rate'] = cg['Deaths'] / cg['Confirmed']
cg100 = pd.DataFrame()

for country in cg['Country/Region'].unique():
    temp_table = cg[(cg['Country/Region'] == country) & (cg['Confirmed'] > 1000)]
    temp_table['day_after100'] = range(0,len(temp_table))
    cg100 = pd.concat([cg100, temp_table])
cg100c = cg100.loc[cg100['Country/Region'] != 'Mainland China']

temp = cg100c.groupby(['day_after100', 'Country/Region'])['Death_rate'].sum().reset_index().sort_values('day_after100', ascending=False)

fig = px.line(temp.sort_values('day_after100'), x="day_after100", y='Death_rate', color='Country/Region', title='Death_rate',)
fig.update_layout(showlegend=False)
fig.show()

cg100c = cg100.loc[cg100['Country/Region'] != 'Mainland China']
cg100c = cg100c.loc[cg100['Country/Region'] != 'Italy']
cg100c = cg100c.loc[cg100['Country/Region'] != 'Spain']

temp = cg100c.groupby(['day_after100', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Confirmed', ascending=False)

fig = px.line(temp.sort_values('day_after100'), x="day_after100", y="Confirmed", color='Country/Region', title='Cases Spread after 100 cases',)
fig.update_layout(showlegend=False)
fig.show()

cg100['Country/Region'].unique()
temp
Ukraine = cg100c.loc[(cg100['Country/Region'] == 'Ukraine')]

values=['Confirmed', 'Deaths', 'Recovered', 'New cases', 'Smooth increase', 'Death_rate']
table = pd.pivot_table(Ukraine, values=values, 
                       index=['Country/Region'],
                       columns=['day_after100'],
                       #aggfunc=np.min, 
                       )
table.reset_index()


temp = cg100c.groupby(['day_after100'])['Confirmed'].mean().reset_index()#.sort_values('Confirmed', ascending=False)
temp['Country/Region'] = 'Average_world'
Ukraine = Ukraine[['day_after100','Confirmed', 'Country/Region']]
temp = pd.concat([temp, Ukraine])
temp= temp.loc[temp.day_after100 < len(Ukraine)+10]

fig = px.line(temp.sort_values('day_after100'), x="day_after100", y="Confirmed", color='Country/Region', 
              title='Cases Spread after 100. Ukraine vs World_average',)
fig.update_layout(showlegend=False)
fig.show()

###
temp = cg100c.groupby(['day_after100'])['Death_rate'].mean().reset_index()
temp['Country/Region'] = 'Average_world'
Ukraine = cg100c.loc[(cg100c['Country/Region'] == 'Czech Republic')]
Ukraine = Ukraine[['day_after100','Death_rate', 'Country/Region']]
temp = pd.concat([temp, Ukraine])
temp= temp.loc[temp.day_after100 < len(Ukraine)+10]

fig = px.line(temp.sort_values('day_after100'), x="day_after100", y="Death_rate", color='Country/Region', 
              title='Death_rate. Ukraine vs World_average WO Italy and Spain',)
fig.update_layout(showlegend=False)
fig.show()

df = pd.read_csv('/kaggle/input/covid19-tests-conducted-by-country/Tests_conducted_15April2020.csv')
df['Tests'] = round(df['Positive'] / df['%'] *100)

temp = df#[['Country or region','Tests','Positive','Tests /millionpeople']]
temp['Country or region'].unique()
short_list = ['Albania', 'Argentina', 'Armenia', 'Australia',
       'Austria', 'Azerbaijan','Belarus', 'Belgium', 'Brazil', 'Bulgaria',
       'Canada', 'Croatia', 'Czechia', 'Denmark','Egypt', 'Estonia',
       'Finland', 'France', 'Germany', 'Greece', 'Grenada', 'Hungary',
       'Iceland', 'India', 'Indonesia', 'Iran', 'Ireland', 'Israel',
       'Italy', 'Japan',   'Kazakhstan','Kyrgyzstan', 'Latvia', 'Lithuania',
       'Mexico', 'Montenegro', 'Nepal',
       'Netherlands', 'New Zealand', 'North Macedonia', 'Norway',
       'Pakistan', 'Palestine', 'Panama', 'Peru', 'Philippines', 'Poland',
       'Portugal', 'Romania', 'Russia',  'Serbia',
       'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea',
       'Spain', 'Sweden', 'Switzerland', 'Taiwan', 'Thailand',
        'Turkey', 'Ukraine',  'United Kingdom', 'Scotland', 'United States','Vietnam']
temp_short = temp[temp['Country or region'].isin(short_list)]
temp_short['Country or region'].replace('United States (unofficial)','US', inplace=True)
temp_short.head()

fig = px.bar(temp_short.sort_values('Tests', ascending=True), 
             x='Country or region', y="Tests", color='Country or region', 
            text='Tests',
            # log_y=True,
             title='Number of tests',
            )
fig.update_layout(showlegend=False)
fig.update_yaxes(range=[0, 1000000])
#fig.update_yaxes(showticklabels=False)
fig.update_xaxes(tickangle=90, tickfont=dict(size=8))

t_s = temp_short.sort_values('Tests', ascending=True).reset_index()
fig.add_shape(
        # unfilled Rectangle
            type="rect",
            x0=t_s.loc[t_s['Country or region']=='Ukraine'].index[0]-.4,
            y0=0,
            x1=t_s.loc[t_s['Country or region']=='Ukraine'].index[0]+.4,
            y1=t_s.loc[t_s['Country or region']=='Ukraine']['Tests'].values[0]+.1,
            line=dict(
                color="RED"
                ),
            line_width=3,
    
        )

fig.show()

fig = px.bar(temp_short.sort_values('Tests /millionpeople', ascending=True), 
             x='Country or region', y='Tests /millionpeople', color='Country or region', 
            text='Tests /millionpeople',
            # log_y=True,
             title='Tests /millionpeople',
            )
fig.update_layout(showlegend=False)
fig.update_yaxes(range=[0, 10000])
#fig.update_yaxes(showticklabels=False)
fig.update_xaxes(tickangle=90, tickfont=dict(size=8))

t_s = temp_short.sort_values('Tests /millionpeople', ascending=True).reset_index()
fig.add_shape(
        # unfilled Rectangle
            type="rect",
            x0=t_s.loc[t_s['Country or region']=='Ukraine'].index[0]-.4,
            y0=0,
            x1=t_s.loc[t_s['Country or region']=='Ukraine'].index[0]+.4,
            y1=t_s.loc[t_s['Country or region']=='Ukraine']['Tests /millionpeople'].values[0]+.1,
            line=dict(
                color="RED"
                ),
            line_width=3,
    
        )

fig.show()
import matplotlib.patches as patches

fig = px.bar(temp_short.sort_values('%', ascending=True), 
             x='Country or region', y='%', color='Country or region', 
            text='%',
            # log_y=True,
             title='%Number of positive cases identified',
            )
fig.update_layout(showlegend=False)
#fig.update_yaxes(range=[0, 40])
#fig.update_yaxes(showticklabels=False)
fig.update_xaxes(tickangle=90, tickfont=dict(size=8))

t_s = temp_short.sort_values('%', ascending=True).reset_index()

# Add patches to color the X axis labels
fig.add_shape(
        # unfilled Rectangle
            type="rect",
            x0=t_s.loc[t_s['Country or region']=='Ukraine'].index[0]-.4,
            y0=0,
            x1=t_s.loc[t_s['Country or region']=='Ukraine'].index[0]+.4,
            y1=t_s.loc[t_s['Country or region']=='Ukraine']['%'].values[0]+.1,
            line=dict(
                color="RED"
            ),line_width=3
            #fillcolor="LightSkyBlue"
        )


fig.show()
temp_short = temp_short.sort_values('Tests', ascending=False)
temp_short['Ukraine'] = ""
temp_short.loc[temp_short['Country or region'] == 'Ukraine', 'Ukraine'] = 'Ukraine'

for i in list(temp_short.head(10)['Country or region']):
    temp_short.loc[temp_short['Country or region'] == i, 'Ukraine'] = i
temp_short.head()
temp_short['Positive_rate'] = temp_short['Positive'] / temp_short['Tests']
fig = px.scatter(temp_short.sort_values('Tests', ascending=False).iloc[:, :], 
                 x='Tests', y='Positive', color='Country or region', size='Positive_rate', #height=700,
                 text='Ukraine', #log_x=True, log_y=True, 
                 title='Tested vs Confirmed (Scale is in log10) ALL')
fig.update_traces(textposition='top center')
fig.update_layout(showlegend=False)
#fig.update_layout(xaxis_rangeslider_visible=True)
#fig.update_yaxes(range=[0, 400000])
#fig.update_xaxes(range=[0, 2000000])
fig.show()


fig = px.scatter(temp_short.sort_values('Tests', ascending=True).iloc[:20, :], 
                 x='Tests', y='Positive', color='Country or region', size='Positive_rate', #height=700,
                 text='Ukraine', #log_x=True, log_y=True, 
                 title='Tested vs Confirmed (Scale is in log10) 20bottom')
fig.update_traces(textposition='top center')
fig.update_layout(showlegend=False)
#fig.update_layout(xaxis_rangeslider_visible=True)
#fig.update_yaxes(range=[0, 40000])
#fig.update_xaxes(range=[0, 200000])
fig.show()
