import pandas as pd

df = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
df = df.dropna()
df.head()
killed = df.groupby('state', as_index=False).agg({'n_killed':'count', 'latitude':'mean', 'longitude':'mean'})
killed.head()
state_to_code = {'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}
killed['code'] = killed['state'].apply(lambda x : state_to_code[x])
killed.head()
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import pandas as pd


df = killed
for col in df.columns:
    df[col] = df[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df['code'],
        z = df['n_killed'].astype(float),
        locationmode = 'USA-states',
        text = df['state'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Kill number")
        ) ]

layout = dict(
        title = 'Number of kills for the month of March of 2013',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot(fig)
import pandas as pd
df = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
df.dropna()
df['participant_gender'].head()
def separate(df):
    df=df.split("||")
    df=[(x.split("::")) for x in df]
    y = []
    for  i in range (0, len(df)):
        y.append(df[i][-1])
    return(y) 
df['participant_gender'] = df['participant_gender'].fillna("0::Zero")
df['gender'] = df['participant_gender'].apply(lambda x: separate(x))
df['Males'] = df['gender'].apply(lambda x: x.count('Male'))
df['Females'] = df['gender'].apply(lambda x: x.count('Female'))
genders = df[['state', 'Males', 'Females']].groupby('state', as_index = False).sum()
import  plotly.plotly as py
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

trace1 = go.Bar(
    x=genders['state'],
    y=genders['Males'],
    name='Males'
)
trace2 = go.Bar(
    x=genders['state'],
    y=genders['Females'],
    name='Females'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
import pandas as pd
df = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
df = df.dropna()
df['participant_age'].head(10)
import numpy as np
def separate_mean(df):
    df=df.split("||")
    df=[(x.split("::")) for x in df]
    y = []
    for  i in range (0, len(df)):
        age = int(df[i][-1].split(':')[-1])
        y.append(age)
    return np.mean(y)
df['age'] = df['participant_age'].apply(lambda x: separate_mean(x))
ages = df[['state', 'age']].groupby('state', as_index=False).mean()
ages.head()
import altair as alt
alt.renderers.enable('notebook')
mean = str(np.mean(ages['age']))
bar = alt.Chart(ages).mark_bar().encode(
    x = 'state:N',
    y = 'age:Q',
)
bar2 = alt.Chart(ages).mark_bar(color="#e45755").encode(
    x = 'state:N',
    y = 'baseline:Q',
    y2='age:Q'
).transform_filter(
    "datum.age >= "+mean
).transform_calculate(
    "baseline", mean
)
line = alt.Chart(ages).mark_rule(size = 2).encode(
    y = 'mean(age):Q'
)
text = alt.Chart(ages).mark_text(
    align='left', dx=1160, dy=-1
).encode(
    alt.Y('mean(age):Q', axis=alt.Axis(title='Age')),
    text=alt.value('Mean')
)
(bar + bar2 + line+text).properties(
    width = 1200
)