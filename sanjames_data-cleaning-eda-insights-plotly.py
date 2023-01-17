#Import all required libraries for reading data, analysing and visualizing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
shootings = pd.read_csv('../input/Mass Shootings Dataset Ver 5.csv',encoding='latin-1')
shootings.shape
shootings.head(2)
shootings.info()
#Which features are available in the dataset?
shootings.columns.values  
shootings.isnull().sum()
#Create new columns for year and month based on the shooting date
from datetime import datetime
shootings['Date'] = pd.to_datetime(shootings['Date'])
shootings['Shooting_Year'] = shootings['Date'].dt.year
shootings['Shooting_Month'] = shootings['Date'].dt.month
#My logic is based on the assumption that some titles have city and state separated by comma and some titles just have the state.
shootings['loc1'] = shootings.Title.str.extract('([A-Za-z]+, [A-Za-z]+)', expand=False)
shootings['loc2']= shootings.Title.str.extract('([A-Za-z]+ )', expand=False)
# For null values of the location column, update the new column location1 with values from loc1. If location1 is still null, update with loc2
for i in range(len(shootings)):
    if pd.isnull(shootings.loc[i,'Location']):
        shootings.loc[i,'Location1'] = shootings.loc[i,'loc1']
        if pd.isnull(shootings['Location1'][i]):
            shootings.loc[i,'Location1'] = shootings.loc[i,'loc2']
    else:
        shootings.loc[i,'Location1'] = shootings.loc[i,'Location']
#Drop all unwanted columns and just retain location1 - rename it to Location
shootings = shootings.drop(['Location','loc1','loc2'], axis=1)
shootings.rename(columns={'Location1': 'Location'}, inplace=True)
shootings['State'] = shootings['Location'].str.rpartition(',')[2]
shootings['State'][shootings.State.str.len()==3].value_counts().head(2)

shootings['State'].replace([' CA',' VA',' WA',' NM',' LA',' GA',' MD',' FL',' CO',' TX',' PA',' TN',' AZ',' NV',' AL',' DE',' NJ'],
                           ['California','Virginia','Washington','New Mexico','Louisiana','Georgia','Maryland','Florida',
                            'Colorado','Texas','Pennsylvania','Tennessee','Arizona','Nevada','Alabama','Delaware','New Jersey'], inplace=True)
shootings['State'].value_counts().head()
shootings.Gender.value_counts()
shootings['Gender'].replace(['M', 'M/F'], ['Male', 'Male/Female'], inplace=True)
shootings.Gender.value_counts()
shootings['Race'].value_counts()
shootings['Race'].replace(['White American or European American', 'white', 'White American or European American/Some other Race'], ['White', 'White', 'White'], inplace=True)
shootings['Race'].replace(['Black American or African American', 'black', 'Black American or African American/Unknown'], ['Black', 'Black', 'Black'], inplace=True)
shootings['Race'].replace(['Asian', 'Asian American/Some other race'], ['Asian American', 'Asian American'], inplace=True)
shootings['Race'].replace(['Unknown', 'Some other race', 'Two or more races'], ['Other', 'Other' ,'Other'], inplace=True)
shootings['Race'].replace(['Native American or Alaska Native'], ['Native American'], inplace=True)
shootings['Race'].value_counts()
shootings['Mental Health Issues'].value_counts()
shootings['Mental Health Issues'].replace(['Unclear' ,'unknown'], ['Unknown', 'Unknown'], inplace=True)
shootings['Mental Health Issues'].value_counts()
shootings['Incident Area'].value_counts().tail()
#Update the Incident area to common groups.
shootings['Area'] = np.nan
shootings.loc[shootings['Incident Area'].str.contains("school",case=False, na=False), 'Area'] = 'School'
shootings.loc[shootings['Incident Area'].str.contains("University",case=False, na=False), 'Area'] = 'School'
shootings.loc[shootings['Incident Area'].str.contains("college",case=False, na=False), 'Area'] = 'School'
shootings.loc[shootings['Incident Area'].str.contains("dormitory",case=False, na=False), 'Area'] = 'School'
shootings.loc[shootings['Incident Area'].str.contains("lecture",case=False, na=False), 'Area'] = 'School'
shootings.loc[shootings['Incident Area'].str.contains("scool",case=False, na=False), 'Area'] = 'School'
shootings.loc[shootings['Incident Area'].str.contains("academy",case=False, na=False), 'Area'] = 'School'
shootings.loc[shootings['Incident Area'].str.contains("classroom",case=False, na=False), 'Area'] = 'School'
shootings.loc[shootings['Incident Area'].str.contains("apartment",case=False, na=False), 'Area'] = 'Home'
shootings.loc[shootings['Incident Area'].str.contains("house",case=False, na=False), 'Area'] = 'Home'
shootings.loc[shootings['Incident Area'].str.contains("home",case=False, na=False), 'Area'] = 'Home'
shootings.loc[shootings['Incident Area'].str.contains("shop",case=False, na=False), 'Area'] = 'Shop'
shootings.loc[shootings['Incident Area'].str.contains("Store",case=False, na=False), 'Area'] = 'Shop'
shootings.loc[shootings['Incident Area'].str.contains("Restaurant",case=False, na=False), 'Area'] = 'Shop'
shootings.loc[shootings['Incident Area'].str.contains("salon",case=False, na=False), 'Area'] = 'Shop'
shootings.loc[shootings['Incident Area'].str.contains("spa",case=False, na=False), 'Area'] = 'Shop'
shootings.loc[shootings['Incident Area'].str.contains("Cafe",case=False, na=False), 'Area'] = 'Shop'
shootings.loc[shootings['Incident Area'].str.contains("grocery",case=False, na=False), 'Area'] = 'Shop'
shootings.loc[shootings['Incident Area'].str.contains("mart",case=False, na=False), 'Area'] = 'Shop'
shootings.loc[shootings['Incident Area'].str.contains("Office",case=False, na=False), 'Area'] = 'Work'
shootings.loc[shootings['Incident Area'].str.contains("Company",case=False, na=False), 'Area'] = 'Work'
shootings.loc[shootings['Incident Area'].str.contains("work",case=False, na=False), 'Area'] = 'Work'
shootings.loc[shootings['Incident Area'].str.contains("conference",case=False, na=False), 'Area'] = 'Work'
shootings.loc[shootings['Incident Area'].str.contains("firm",case=False, na=False), 'Area'] = 'Work'
shootings.loc[shootings['Incident Area'].str.contains("street",case=False, na=False), 'Area'] = 'Road'
shootings.loc[shootings['Incident Area'].str.contains("lot",case=False, na=False), 'Area'] = 'Road'
shootings.loc[shootings['Incident Area'].str.contains("walk",case=False, na=False), 'Area'] = 'Road'
shootings.loc[shootings['Incident Area'].str.contains("highway",case=False, na=False), 'Area'] = 'Road'
shootings.loc[shootings['Incident Area'].str.contains("interstate",case=False, na=False), 'Area'] = 'Road'
shootings.loc[shootings['Incident Area'].str.contains("club",case=False, na=False), 'Area'] = 'Pub-Club'
shootings.loc[shootings['Incident Area'].str.contains("pub",case=False, na=False), 'Area'] = 'Pub-Club'
shootings.loc[shootings['Incident Area'].str.contains("gas",case=False, na=False), 'Area'] = 'Gas-Station'
shootings.loc[shootings['Incident Area'].str.contains("party",case=False, na=False), 'Area'] = 'Party'
shootings.loc[shootings['Incident Area'].str.contains("airport",case=False, na=False), 'Area'] = 'Airport'
shootings.loc[shootings['Incident Area'].str.contains("clinic",case=False, na=False), 'Area'] = 'Hospital'
shootings.loc[shootings['Incident Area'].str.contains("Hospital",case=False, na=False), 'Area'] = 'Hospital'
shootings.loc[shootings['Incident Area'].str.contains("church",case=False, na=False), 'Area'] = 'Religious'
shootings.loc[shootings['Incident Area'].str.contains("temple",case=False, na=False), 'Area'] = 'Religious'
shootings.loc[shootings['Incident Area'].str.contains("monastery",case=False, na=False), 'Area'] = 'Religious'
shootings.Area.fillna(shootings['Incident Area'],inplace=True)
shootings['Area'].value_counts().head(10)
shootings['Target'].value_counts().tail()
shootings['FTarget'] = np.nan
shootings.loc[shootings['Target'].str.contains("family",case=False, na=False), 'FTarget'] = 'Family'
shootings.loc[shootings['Target'].str.contains("Ex-",case=False, na=False), 'FTarget'] = 'Family'
shootings.loc[shootings['Target'].str.contains("coworker",case=False, na=False), 'FTarget'] = 'Coworkers'
shootings.loc[shootings['Target'].str.contains("employee",case=False, na=False), 'FTarget'] = 'Coworkers'
shootings.loc[shootings['Target'].str.contains("Student",case=False, na=False), 'FTarget'] = 'School'
shootings.loc[shootings['Target'].str.contains("player",case=False, na=False), 'FTarget'] = 'School'
shootings.loc[shootings['Target'].str.contains("teacher",case=False, na=False), 'FTarget'] = 'School'
shootings.loc[shootings['Target'].str.contains("school",case=False, na=False), 'FTarget'] = 'School'
shootings.loc[shootings['Target'].str.contains("random",case=False, na=False), 'FTarget'] = 'Random'
shootings['FTarget'].fillna(shootings['Target'],inplace=True)
shootings['FTarget'].value_counts().head()
shootings['Cause'].value_counts()
shootings['Cause'].replace(['domestic disputer'], ['domestic dispute'], inplace=True)
shootings['Open/Close Location'].value_counts()
shootings['Open/Close Location'].replace(['Open+CLose'], ['Open+Close'], inplace=True)
#Some age has the ages of both shooters. I'm going with the assumption of picking the first Age.
shootings['Age1'] = np.nan
shootings.loc[shootings['Age'].str.contains(",",case=False, na=False), 'Age1'] = shootings['Age'].str.rpartition(',')[0]
shootings = shootings.drop(['Age'], axis=1)
shootings.rename(columns={'Age1': 'Age'}, inplace=True)
#create a new Mass Shooting data frame with only required columns
ms = pd.DataFrame(shootings[['Date', 'Shooting_Year', 'Shooting_Month', 'State', 'Fatalities','Injured', 'Total victims', 'Policeman Killed', 'Age', 
                             'Race', 'Gender','Cause', 'Area', 'FTarget', 'Latitude', 'Longitude','Mental Health Issues','Employeed (Y/N)']])
ms.head(2)
data = [dict(
  x = ms['Date'],
  autobinx = False,
  autobiny = True,
  marker = dict(color = 'rgb(255, 0, 0)'),
  name = 'date',
  type = 'histogram',
  xbins = dict(
    end = '2017-12-31 12:00',
    size = 'M1',
    start = '1966-01-01 12:00'
  )
)]

layout = dict(
  paper_bgcolor = 'rgb(240, 240, 240)',
  plot_bgcolor = 'rgb(240, 240, 240)',
  title = '<b>US Mass Shootings 1966-2017 / Monthly, Yearly, Quarterly details</b>',
  xaxis = dict(
    title = '',
    type = 'date'
  ),
  yaxis = dict(
    title = 'US Mass Shootings Count',
    type = 'linear'
  ),
  updatemenus = [dict(
        x = 0.1,
        y = 1.15,
        xref = 'paper',
        yref = 'paper',
        yanchor = 'top',
        active = 1,
        showactive = True,
        buttons = [
        dict(
            args = ['xbins.size', 'M12'],
            label = 'Year',
            method = 'restyle',
        ), dict(
            args = ['xbins.size', 'D1'],
            label = 'Day',
            method = 'restyle',
        ), dict(
            args = ['xbins.size', 'M1'],
            label = 'Month',
            method = 'restyle',
        ), dict(
            args = ['xbins.size', 'M3'],
            label = 'Quarter',
            method = 'restyle',
        )]
  )]
)
py.iplot({'data': data,'layout': layout}, validate=False)
#Mass Shooting over the years
ms_year_stats = shootings[['Shooting_Year', 'Total victims' ]].groupby(['Shooting_Year'], as_index=False).sum().sort_values(by='Shooting_Year', ascending=False)
ms_year_stats1 = shootings[['Shooting_Year', 'Fatalities' ]].groupby(['Shooting_Year'], as_index=False).sum().sort_values(by='Shooting_Year', ascending=False)
ms_year_stats2 = shootings[['Shooting_Year', 'Injured' ]].groupby(['Shooting_Year'], as_index=False).sum().sort_values(by='Shooting_Year', ascending=False)
ms_year_stats3 = shootings[['Shooting_Year', 'Total victims' ]].groupby(['Shooting_Year'], as_index=False).count().sort_values(by='Shooting_Year', ascending=False)
# Create traces
trace0 = go.Scatter(
    x = ms_year_stats['Shooting_Year'],
    y = ms_year_stats['Total victims'],
    mode = 'lines',
    name = 'Total Victims'
)
trace1 = go.Scatter(
    x = ms_year_stats1['Shooting_Year'],
    y = ms_year_stats1['Fatalities'],
    mode = 'lines+markers',
    name = 'Fatalities'
)
trace2 = go.Scatter(
    x = ms_year_stats2['Shooting_Year'],
    y = ms_year_stats2['Injured'],
    mode = 'lines+markers',
    name = 'Injured'
)

trace3 = go.Scatter(
    x = ms_year_stats3['Shooting_Year'],
    y = ms_year_stats3['Total victims'],
    mode = 'lines+markers',
    name = 'Count of Mass Shootings'
)
data = [trace0, trace1, trace2, trace3]
py.iplot(data, filename='Mass shooting Fatalities Vs Total Victims')
ms_cause_cnt = ms[['Cause', 'Total victims' ]].groupby(['Cause'], as_index=False).count().sort_values(by='Total victims', ascending=False)
ms_cause_cnt[ms_cause_cnt['Total victims'] > 9]
ms_cause_sum = ms[['Cause', 'Total victims' ]].groupby(['Cause'], as_index=False).sum().sort_values(by='Total victims', ascending=False)
ms_cause_sum[ms_cause_sum['Total victims'] > 9]
values1 = ms_cause_cnt['Total victims']
labels1 = ms_cause_cnt['Cause']
values2 = ms_cause_sum['Total victims']
labels2 = ms_cause_sum['Cause']
fig = {
  "data": [
    {
      "values": values1,
      "labels": labels1,
      "domain": {"x": [0, .48]},
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": values2,
      "labels": labels2,
      "domain": {"x": [.52, 1]},
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Mass Shootings Cause that resulted in more shootings and more victims",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "More shootings",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "More victims",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='Mass Shootings Cause')
ms_race_cnt = ms[['Race', 'Total victims' ]].groupby(['Race'], as_index=False).count().sort_values(by='Total victims', ascending=False)
ms_race_cnt
ms_race_sum = ms[['Race', 'Total victims' ]].groupby(['Race'], as_index=False).sum().sort_values(by='Total victims', ascending=False)
ms_race_sum
values1 = ms_race_cnt['Total victims']
labels1 = ms_race_cnt['Race']
values2 = ms_race_sum['Total victims']
labels2 = ms_race_sum['Race']
fig = {
  "data": [
    {
      "values": values1,
      "labels": labels1,
      "domain": {"x": [0, .48]},
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": values2,
      "labels": labels2,
      "domain": {"x": [.52, 1]},
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Shooters race that resulted in more shootings and more victims",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "More Shootings",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "More Victims",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='Mass Shootings Race')
ms_health_cnt = pd.DataFrame(ms[['Cause','Race', 'Mental Health Issues','Total victims']].groupby(['Cause','Race','Mental Health Issues'], as_index=False).count().sort_values(by='Total victims', ascending=False))
ms_health_sum = pd.DataFrame(ms[['Cause','Race', 'Mental Health Issues','Total victims']].groupby(['Cause','Race','Mental Health Issues'], as_index=False).sum().sort_values(by='Total victims', ascending=False))
ms_health_cnt.head(3), ms_health_sum.head(3)
g = sns.factorplot(x='Total victims', y='Cause', col='Mental Health Issues', hue= 'Race', kind='bar', 
                   data=ms_health_cnt[ms_health_cnt['Total victims'] > 2], saturation=.8, size=5,
               ci=None, aspect=.8, col_wrap=3, palette='Set1')
g.set_xticklabels(step=1)
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Number of Mass Shootings based on Race, Cause and Mental Health')
g = sns.factorplot(x='Total victims', y='Cause', col='Mental Health Issues', hue= 'Race', kind='bar', 
                   data=ms_health_sum[ms_health_sum['Total victims'] > 2], saturation=.8, size=5,
               ci=None, aspect=.8, col_wrap=3, palette='Set1')
g.set_xticklabels(step=1)
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Victims impacted during Mass Shootings based on Race, Cause and Mental Health')
impute_grps = shootings.pivot_table(values=["Total victims"], index=["Shooting_Year","Race","Cause"], aggfunc=np.sum)
print(impute_grps)
ms_state = pd.DataFrame(ms[['State','Shooting_Year', 'Latitude', 'Longitude', 'Fatalities']][ms['Fatalities']>0].sort_values(by='Fatalities', ascending=False))
ms_state['Desc'] = ms_state['State'] + '<br>Year ' + (ms_state['Shooting_Year']).astype(str) + '<br>Fatalities ' + (ms_state['Fatalities']).astype(str)
ms_state.head()
limits = [(0,1),(2,10),(11,50),(51,236),(237,350)]
colors = ["rgb(255,0,0)","rgb(128,0,128)","rgb(0,255,255)","rgb(173,255,47)", "rgb(0,0,255)"]
states = []

for i in range(len(limits)):
    lim = limits[i]
    ms_state_df = ms_state[lim[0]:lim[1]]
    state = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = ms_state_df[ms_state_df['Longitude'].notnull()]['Longitude'],
        lat = ms_state_df[ms_state_df['Longitude'].notnull()]['Latitude'],
        text = ms_state_df[ms_state_df['Longitude'].notnull()]['Desc'],
        marker = dict(
            size = ms_state_df[ms_state_df['Longitude'].notnull()]['Fatalities']**2 ,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = 'Fatalities >='+(ms_state.loc[lim[0],'Fatalities']).astype(str)  )
    states.append(state)
        
layout = dict(
        title = 'Mass shootings in US 1966-2017',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=states, layout=layout )
py.iplot( fig, validate=False, filename='US Mass Shootings' )
state = ms_state['State']
totvictims = ms_state['Fatalities']

layout = dict(
  title = 'US States that had most fatalities during Mass Shootings'
)    

data = [dict(
  type = 'bar',
  x = state,
  y = totvictims,
  mode = 'markers',
  transforms = [dict(
    type = 'aggregate',
    groups = state,
    aggregations = [dict(
        target = 'y', func = 'sum', ascending=False, enabled = True),
    ]
  )]
)]
py.iplot({'data': data, 'layout': layout}, validate=False)
ms_school = ms[['Shooting_Year', 'Fatalities', 'Injured', 'Total victims','State','Race','Cause','Mental Health Issues',
                'FTarget', 'Area']][ms['Area'] == 'School']
ms_school.head()
ms_school0 = ms_school[['Shooting_Year', 'Total victims' ]].groupby(['Shooting_Year'], as_index=False).sum().sort_values(by='Shooting_Year', ascending=False)
ms_school1 = ms_school[['Shooting_Year', 'Fatalities' ]].groupby(['Shooting_Year'], as_index=False).sum().sort_values(by='Shooting_Year', ascending=False)
ms_school2 = ms_school[['Shooting_Year', 'Injured' ]].groupby(['Shooting_Year'], as_index=False).sum().sort_values(by='Shooting_Year', ascending=False)
ms_school3 = ms_school[['Shooting_Year', 'Total victims' ]].groupby(['Shooting_Year'], as_index=False).count().sort_values(by='Shooting_Year', ascending=False)

# Create traces
trace0 = go.Scatter(
    x = ms_school0['Shooting_Year'],
    y = ms_school0['Total victims'],
    mode = 'lines',
    name = 'Total Victims'
)
trace1 = go.Scatter(
    x = ms_school1['Shooting_Year'],
    y = ms_school1['Fatalities'],
    mode = 'lines+markers',
    name = 'Fatalities'
)
trace2 = go.Scatter(
    x = ms_school2['Shooting_Year'],
    y = ms_school2['Injured'],
    mode = 'lines+markers',
    name = 'Injured'
)

trace3 = go.Scatter(
    x = ms_school3['Shooting_Year'],
    y = ms_school3['Total victims'],
    mode = 'lines+markers',
    name = 'Count of Mass Shootings'
)

layout = dict(
  title = 'Mass Shootings Stats in Schools'
)    
data = [trace0, trace1, trace2, trace3]
py.iplot({'data': data, 'layout': layout}, filename='Mass shooting Fatalities in Schools')
ms_sch_target_cnt = ms_school[['FTarget', 'Total victims' ]].groupby(['FTarget'], as_index=False).count().sort_values(by='Total victims', ascending=False)
ms_sch_target_sum = ms_school[['FTarget', 'Total victims' ]].groupby(['FTarget'], as_index=False).sum().sort_values(by='Total victims', ascending=False)



values1 = ms_sch_target_cnt['Total victims']
labels1 = ms_sch_target_cnt['FTarget']
values2 = ms_sch_target_sum['Total victims']
labels2 = ms_sch_target_sum['FTarget']
fig = {
  "data": [
    {
      "values": values1,
      "labels": labels1,
      "domain": {"x": [0, .48]},
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": values2,
      "labels": labels2,
      "domain": {"x": [.52, 1]},
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"School shooters and their Targets",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "More Shootings Target",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "More Victims Target",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
py.iplot(fig, filename='School Shootings')
g = sns.factorplot(x='FTarget', y='Total victims', col='Race', row= 'Cause', kind='bar', 
                   data=ms_school[ms_school['Total victims']>5], saturation=.8, size=5,
               ci=None, aspect=.8,  palette='Set1')
g.set_xticklabels(step=1)
ms_gender = ms[['Shooting_Year','Gender','Fatalities','Injured', 'Total victims']].sort_values(ascending=False, by='Total victims')
ms_gender.head()
gender = ms_gender['Gender']
totvictims = ms_gender['Total victims']

layout = dict(
  title = 'Mass Shootings and Gender'
)    

data = [dict(
  type = 'scatter',
  x = gender,
  y = totvictims,
  mode = 'markers',
  transforms = [dict(
    type = 'aggregate',
    groups = gender,
    aggregations = [dict(
        target = 'y', func = 'sum', ascending=False, enabled = True),
    ]
  )]
)]
py.iplot({'data': data, 'layout': layout}, validate=False)
