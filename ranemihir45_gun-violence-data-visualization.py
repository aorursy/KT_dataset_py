# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px



# Input data files are available in the "../input/"directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Standard plotly imports

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode

import cufflinks

import datetime

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)
violence_df = pd.read_csv("/kaggle/input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv")

violence_df.head()
fig = px.histogram(violence_df,'n_killed')

fig.show()
# source : https://gist.github.com/rogerallen/1583593



states = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'American Samoa': 'AS',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Guam': 'GU',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY'

}
violence_df['state code'] = violence_df['state'].apply(lambda x : states[x])



violence_df.head()
violence_df.groupby('state code')['n_killed'].sum().sort_values(ascending = False).reset_index().head()


fig = px.choropleth(violence_df.groupby('state code')['n_killed'].sum().reset_index(), locations='state code', locationmode="USA-states", color='n_killed', scope="usa", color_continuous_scale="Viridis")

fig.show()
violence_df.groupby('state code')['n_injured'].sum().sort_values(ascending = False).reset_index().head()
import plotly.express as px



fig = px.choropleth(violence_df.groupby('state code')['n_injured'].sum().reset_index(), locations='state code', locationmode="USA-states", color='n_injured', scope="usa", color_continuous_scale="Viridis")

fig.show()
illinois = violence_df[violence_df['state code'] == "IL"].copy()

illinois['text'] = illinois['n_injured'].astype(str)



fig = go.Figure(data=go.Scattergeo(

        lon = illinois['longitude'],

        lat = illinois['latitude'],

        text = illinois['text'],

        mode = 'markers',

        marker_color = illinois['n_injured'],

        ))



fig.update_layout(

        title = 'Distribution of injured people in Illinois',

        geo_scope='usa',

    )

fig.show()
import plotly.express as px



fig = px.choropleth(violence_df.groupby('state code')['n_guns_involved'].sum().reset_index(), locations='state code', locationmode="USA-states", color='n_guns_involved', scope="usa", color_continuous_scale="Viridis")

fig.show()
violence_df['toddler'] = 0

violence_df['child'] = 0

violence_df['adult'] = 0

violence_df['elderly'] = 0
def age_group(row):

    age_col = row['participant_age']

    if str(age_col) == 'nan':

        return row

    

    ages = age_col.split("||")

    if len(ages) == 1:

        ages = ages[0].split("|")

    for age in ages:

        try:

            age_value = int(age.split('::')[1])

        except:

            age_value = int(age.split(':')[1])

        if age_value >= 0 and age_value <= 2:

            row['toddler'] += 1

        elif age_value >= 3 and age_value <= 17:

            row['child'] += 1

        elif age_value >= 18 and age_value <= 65:

            row['adult'] += 1

        else:

            row['elderly'] += 1

    return row
x = violence_df.apply(age_group, axis = 1)
x.head()
x.groupby('state')['toddler'].sum().sort_values(ascending = False).reset_index().head()
import plotly.express as px



fig = px.choropleth(x.groupby('state code')['toddler'].sum().reset_index(), locations='state code', locationmode="USA-states", color='toddler', scope="usa", color_continuous_scale="Viridis")

fig.show()
import plotly.express as px



fig = px.choropleth(x.groupby('state code')['child'].sum().reset_index(), locations='state code', locationmode="USA-states", color='child', scope="usa", color_continuous_scale="Viridis")

fig.show()
import plotly.express as px



fig = px.choropleth(x.groupby('state code')['adult'].sum().reset_index(), locations='state code', locationmode="USA-states", color='adult', scope="usa", color_continuous_scale="Viridis")

fig.show()
import plotly.express as px

fig = px.choropleth(x.groupby('state code')['elderly'].sum().reset_index(), locations='state code', locationmode="USA-states", color='elderly', scope="usa", color_continuous_scale="Viridis")

fig.show()
trace = go.Scatter(x=list(violence_df.date),

                   y=list(violence_df.n_killed))



data = [trace]

layout = dict(

    title='Number of people killed over time',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(count=1,

                    label='YTD',

                    step='year',

                    stepmode='todate'),

                dict(count=1,

                    label='1y',

                    step='year',

                    stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(

            visible = True

        ),

        type='date'

    )

)



fig = dict(data=data, layout=layout)

iplot(fig)
fig = go.Figure()

fig.add_trace(go.Scatter(x=x['date'], y=x['toddler'], name="Toddler",

                         line_color='deepskyblue'))



fig.add_trace(go.Scatter(x=x['date'], y=x['child'], name="Child",

                         line_color='rgb(168, 50, 151)'))



fig.add_trace(go.Scatter(x=x['date'], y=x['adult'], name="Adult",

                         line_color='rgb(50, 168, 52)'))



fig.add_trace(go.Scatter(x=x['date'], y=x['elderly'], name="Elderly",

                         line_color='yellow'))



fig.update_layout(title_text='Time Series with Age Groups',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label="1m",

                     step="month",

                     stepmode="backward"),

                dict(count=6,

                     label="6m",

                     step="month",

                     stepmode="backward"),

                dict(count=1,

                     label="YTD",

                     step="year",

                     stepmode="todate"),

                dict(count=1,

                     label="1y",

                     step="year",

                     stepmode="backward"),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            visible=True

        ),

        type="date"

    ))



fig.show()
gun_types = {}



def gun_count(gun_type):

    if str(gun_type) != 'nan':

        guns = gun_type.split("||")

        if len(guns) == 1:

            guns = guns[0].split("|")

        for gun in guns:

            try:

                if gun.split('::')[1] in gun_types.keys():

                    gun_types[gun.split('::')[1]] += 1

                else:

                    gun_types[gun.split('::')[1]] = 0

            except:

                if gun.split(':')[1] in gun_types.keys():

                    gun_types[gun.split(':')[1]] += 1

                else:

                    gun_types[gun.split(':')[1]] = 0



violence_df['gun_type'].apply(gun_count);
gun_types
gun_df = pd.DataFrame(gun_types.items(), columns = ['Gun Type', 'Count'])
fig = px.bar(gun_df, x='Gun Type', y='Count')

fig.show()
import plotly.express as px



fig = px.bar(violence_df.groupby('congressional_district')['n_killed'].sum().reset_index(), x='congressional_district', y='n_killed')

fig.show()
df = violence_df.groupby(['state','congressional_district'])['n_killed'].sum()



df
df.index.levels[0]
l = []

for i in df.index.levels[1]:

    l.append(go.Bar(name=i, x=df.xs(i, level = 1).index.tolist(), y=df.xs(i, level = 1).values.tolist()))
fig = go.Figure(data=l)

# Change the bar mode

fig.update_layout(barmode='stack',xaxis_tickangle=-45)

fig.show()
words = ' '.join(i for i in violence_df['notes'].str.split(expand=True).stack().values.tolist())
from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 

stopwords = set(STOPWORDS) 

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
incident_types = {}



def incident_count(incident_type):

    if str(incident_type) != 'nan':

        incidents = incident_type.split("||")

        if len(incidents) == 1:

            incidents = incidents[0].split("|")

        for incident in incidents:

            if incident in incident_types.keys():

                incident_types[incident] += 1

            else:

                incident_types[incident] = 0



violence_df['incident_characteristics'].apply(incident_count) 
incident_types
incident_types = {k: v for k, v in sorted(incident_types.items(), key=lambda item: -item[1])}

incident_df = pd.DataFrame(incident_types.items(), columns = ['Incident Type', 'Count'])

fig = px.bar(incident_df.head(10), x='Incident Type', y='Count')

# fig.update_layout(xaxis_tickangle=45)

fig.show()
age_types = {}



def age_count(age_type):

    if str(age_type) != 'nan':

        ages = age_type.split("||")

        if len(ages) == 1:

            ages = ages[0].split("|")

        for age in ages:

            try:

                if age.split('::')[1] in age_types.keys():

                    age_types[age.split('::')[1]] += 1

                else:

                    age_types[age.split('::')[1]] = 0

            except:

                if age.split(':')[1] in age_types.keys():

                    age_types[age.split(':')[1]] += 1

                else:

                    age_types[age.split(':')[1]] = 0





violence_df['participant_age_group'].apply(age_count) 
age_types = {k: v for k, v in sorted(age_types.items(), key=lambda item: -item[1])}

age_df = pd.DataFrame(age_types.items(), columns = ['Age Type', 'Count'])

fig = px.bar(age_df, x='Age Type', y='Count')

# fig.update_layout(xaxis_tickangle=45)

fig.show()
participant_types = {}



def participant_type_count(age_type):

    if str(age_type) != 'nan':

        ages = age_type.split("||")

        if len(ages) == 1:

            ages = ages[0].split("|")

        for age in ages:

            try:

                if age.split('::')[1] in participant_types.keys():

                    participant_types[age.split('::')[1]] += 1

                else:

                    participant_types[age.split('::')[1]] = 0

            except:

                if age.split(':')[1] in participant_types.keys():

                    participant_types[age.split(':')[1]] += 1

                else:

                    participant_types[age.split(':')[1]] = 0





violence_df['participant_type'].apply(participant_type_count);
participant_types
participant_df = pd.DataFrame(participant_types.items(), columns = [' Type', 'Count'])

fig = px.bar(participant_df, x=' Type', y='Count')

# fig.update_layout(xaxis_tickangle=45)

fig.show()
participant_statuses  = {}



def participant_type_count(age_type):

    if str(age_type) != 'nan':

        ages = age_type.split("||")

        if len(ages) == 1:

            ages = ages[0].split("|")

        for age in ages:

            try:

                if age.split('::')[1] in participant_statuses.keys():

                    participant_statuses[age.split('::')[1]] += 1

                else:

                    participant_statuses[age.split('::')[1]] = 0

            except:

                if age.split(':')[1] in participant_statuses.keys():

                    participant_statuses[age.split(':')[1]] += 1

                else:

                    participant_statuses[age.split(':')[1]] = 0





violence_df['participant_status'].apply(participant_type_count);
participant_statuses
participant_statuses = {k: v for k, v in sorted(participant_statuses.items(), key=lambda item: -item[1])}

age_df = pd.DataFrame(participant_statuses.items(), columns = ['participant statuses', 'Count'])

fig = px.bar(age_df, x='participant statuses', y='Count')

# fig.update_layout(xaxis_tickangle=45)

fig.show()
d = set()

def unique_items(col):

    if str(col) != 'nan':

        col = col.split("||")

        for i in col:

            try:

                d.add(i.split('::')[1])

            except:

                d.add(i.split(':')[1])
violence_df['participant_relationship'].apply(unique_items);



for i in d:

    violence_df[i] = 0
def participant_group(row, col_name):

    res_col = row[col_name]

    if str(res_col) == 'nan':

        return row

    

    results = res_col.split("||")

    if len(results) == 1:

        results = results[0].split("|")

    for res in results:

        try:

            res_value = res.split('::')[1]

        except:

            res_value = res.split(':')[1]

        row[res_value] += 1

    return row
violence_relations = violence_df.apply(lambda x : participant_group(x, 'participant_relationship'), axis = 1);
# fig = go.Figure()



# from random import randint



# for i in d:

#     color = '#%06X' % randint(0, 0xFFFFFF)

    

#     fig.add_trace(go.Scatter(x=violence_relations['date'], y=violence_relations[i], name=i,

#                          line_color= color))



# fig.update_layout(title_text='Time Series with Participant Relations',

#     xaxis=dict(

#         rangeselector=dict(

#             buttons=list([

#                 dict(count=1,

#                      label="1m",

#                      step="month",

#                      stepmode="backward"),

#                 dict(count=6,

#                      label="6m",

#                      step="month",

#                      stepmode="backward"),

#                 dict(count=1,

#                      label="YTD",

#                      step="year",

#                      stepmode="todate"),

#                 dict(count=1,

#                      label="1y",

#                      step="year",

#                      stepmode="backward"),

#                 dict(step="all")

#             ])

#         ),

#         rangeslider=dict(

#             visible=True

#         ),

#         type="date"

#     ))



# fig.show()
df = violence_relations[d].sum().sort_values(ascending = False).reset_index(name = 'count')

fig = px.bar(df, x = 'index', y = 'count')

fig.show()
import plotly.express as px

for i in df['index'].head(5).tolist():

    fig = px.choropleth(violence_relations.groupby('state code')[i].sum().reset_index(), locations='state code', locationmode="USA-states", color=i, scope="usa", color_continuous_scale="Viridis")

    fig.update_layout(title_text=i,

                  title_font_size=30)

    fig.show()