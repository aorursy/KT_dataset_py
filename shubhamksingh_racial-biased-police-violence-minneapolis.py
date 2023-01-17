import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sb

import matplotlib.pyplot as plt

from wordcloud import WordCloud, ImageColorGenerator

from matplotlib import rcParams

rcParams['figure.figsize'] = (15, 8)



#ignoring all the warnings because we don't need them

import warnings

warnings.filterwarnings('ignore')







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
stop_data = pd.read_csv('../input/minneapolis-police-stops-and-police-violence/police_stop_data.csv')

force_data = pd.read_csv('../input/minneapolis-police-stops-and-police-violence/police_use_of_force.csv')
stop_data.head(3)
stop_data.tail(3)
stop_data.shape
stop_data.info()
stop_data.describe()
print("Missing Data")

miss_me = stop_data.isnull().sum()

miss_me[miss_me>0]
numerical_cols = [col for col in stop_data.columns 

                  if stop_data[col].dtype!='object']



print(f"Numerical Columns:\n{numerical_cols}")
categorical_cols = [col for col in stop_data.columns

                    if stop_data[col].dtype=='object']



print(f"Categorical Columns:\n{categorical_cols}")
census_2010 = {'white': ['63.8'], 

        'black': ['18.6'], 

        'hispanic': ['10.5'],

        'asian': ['5.6'],

        'other': ['5.6'],

        'american_indian': ['2.0']}



census_2010_df = pd.DataFrame(census_2010, columns = ['white','black','hispanic','asian','other','american_indian']).transpose()

census_2010_df.columns = ['Percentage of Population']

census_2010_df['Percentage of Population'] = census_2010_df['Percentage of Population'].astype(float)
census_2010_df.style.background_gradient(cmap='Purples', subset=['Percentage of Population'])
# Bar Plot

fig1 = px.bar(census_2010_df, x=census_2010_df.index, 

              y=census_2010_df['Percentage of Population'], color=census_2010_df.index, 

              color_discrete_sequence=px.colors.qualitative.Pastel)



fig1.update_layout(title={

                  'text': "Demographics of Minneapolis City",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                  template='ggplot2')



# -----------------------------------------------------------



# Pie Chart

fig2 = px.pie(census_2010_df, census_2010_df.index, 

              census_2010_df['Percentage of Population'], 

              color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)



fig2.update_layout(title={

                  'text': "Demographics of Minneapolis City (Pie Chart)",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                  template='plotly_white')



fig2.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.2])



fig2.data[0].marker.line.width = 1

fig2.data[0].marker.line.color = "black"



fig1.show()

fig2.show()
stop_data['race'].value_counts()
# Bar Plot

fig1 = px.bar(stop_data['race'].value_counts(), color_discrete_sequence=[px.colors.qualitative.Pastel])



fig1.update_layout(title={

                  'text': "Minneapolis Police Stops by Race",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Race", 

                  yaxis_title="Total Number",

                  showlegend=False

                 )



# -------------------------------------------------------------



# Funnel Plot

fig2 = px.funnel(stop_data['race'].value_counts(), color_discrete_sequence=[px.colors.qualitative.Pastel])

fig2.update_layout(template='ggplot2', showlegend=False)



fig1.show()

fig2.show()
# Fill missing values with 'None'

stop_data['reason'] = stop_data['reason'].fillna('None')



stop_reason = stop_data['reason'].unique()



print("Different Reasons for Stopping the vehicle:")

for x in stop_reason:

    print(x)
print("Total Number of Cases:")

stop_data['reason'].value_counts()
# Bar Chart

fig1 = px.bar(stop_data['reason'].value_counts(), color_discrete_sequence=[px.colors.qualitative.Pastel])



fig1.update_layout(title={

                  'text': "Reason for Police Stops",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Reason", 

                  yaxis_title="Total Number",

                  showlegend=False

                 )

# ----------------------------------------------------



# Pie Chart

fig2 = px.pie(stop_data, stop_data['reason'], 

              color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3)



fig2.update_layout(title={

                  'text': "Reasons for Police Stops (Pie Chart)",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                  template='plotly_white')



fig2.update_traces(textposition='inside', textinfo='percent+label', pull=(0.05))



fig2.data[0].marker.line.width = 1

fig2.data[0].marker.line.color = "black"



fig1.show()

fig2.show()
fig = px.bar(stop_data['problem'].value_counts(), color_discrete_sequence=[px.colors.qualitative.Pastel])



fig.update_layout(title={

                  'text': "Problem Stated by Police",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Problem", 

                  yaxis_title="Total Number",

                  showlegend=False

                 )



fig.show()
problem_stop = stop_data.groupby('race')[['problem']].count().reset_index()

problem_stop.sort_values(by='problem',  ascending=False).style.background_gradient(cmap='Reds', subset=['problem'])
fig1 = px.bar(problem_stop, problem_stop['race'], 

              problem_stop['problem'], 

              color_discrete_sequence = [px.colors.qualitative.Plotly])



fig1.update_layout(title={

                  'text': "Problem based on Race",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Race", 

                  yaxis_title="Problem",

                 )



# ----------------------------------------------------



fig2 = go.Figure(data=[go.Scatter(

    x=problem_stop['race'], y=problem_stop['problem'],

    mode='markers',

    marker=dict(

        color= px.colors.qualitative.Plotly,

        size=[20, 140, 60, 50, 40, 30, 70, 90],

    )

    )])



fig2.update_layout(template='ggplot2',

                   xaxis_title='Race',

                   yaxis_title='Problem')



fig1.show()

fig2.show()
word = stop_data['callDisposition']



text = " ".join(str(each) for each in word.unique())



wordcloud = WordCloud(max_words=200, colormap='Set3', background_color="white").generate(text)



plt.figure(figsize=(17,10))



plt.imshow(wordcloud, interpolation='Bilinear')



plt.axis("off")



plt.figure(1,figsize=(12, 12))



plt.title("Word Cloud of Call Disposition", size=20)



plt.show()
personSearch_df = stop_data[stop_data['personSearch'] == 'YES']



search_person = personSearch_df.groupby('race')['personSearch'].count().reset_index()



search_person.sort_values(by='personSearch', ascending=False).style.background_gradient(cmap='Oranges', subset=['personSearch'])
fig = px.pie(search_person, search_person['race'], search_person['personSearch'], color_discrete_sequence=px.colors.qualitative.Pastel)



fig.update_layout(title={

                  'text': "Person Search based on Race",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                  template='plotly_white')



fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.2])



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"



fig.show()
vehicleSearch_df = stop_data[stop_data['vehicleSearch'] == 'YES']



search_vehicle = vehicleSearch_df.groupby('race')['vehicleSearch'].count().reset_index()



search_vehicle.sort_values(by='vehicleSearch', ascending=False).style.background_gradient(cmap='Oranges', subset=['vehicleSearch'])
fig = px.pie(search_vehicle, search_vehicle['race'], search_vehicle['vehicleSearch'], color_discrete_sequence=px.colors.qualitative.Pastel)



fig.update_layout(title={

                  'text': "Vehicle Search based on Race",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                  template='plotly_white')



fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.2])



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"



fig.show()
stop_data['neighborhood'].nunique()
stop_data['neighborhood'].value_counts().to_frame(name='# Police Stops').head(15).style.background_gradient(cmap='Blues', subset=['# Police Stops'])
fig = px.bar(stop_data['neighborhood'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)

fig.update_layout(title={

                  'text': "# Police Stops in Different Neighbourhoods",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Neighbourhood", 

                  yaxis_title="# Police Stops",

                  height=700,

                  showlegend=False

                 )
fig = px.bar(stop_data['gender'].value_counts(), color_discrete_sequence = [px.colors.qualitative.Pastel])

fig.update_layout(title={

                  'text': "Police Stops Based on Gender",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Gender", 

                  yaxis_title="# Police Stops",

                  showlegend=False

                 )

fig.show()
force_data.head(3)
force_data.columns
force_data.info()
force_data.describe()
fig = px.bar(force_data['Race'].value_counts(), color_discrete_sequence = [px.colors.qualitative.Pastel])



fig.update_layout(title={

                  'text': "Police Violence by Race",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Race", 

                  yaxis_title="# Police Violence",

                  showlegend=False

                 )





fig.show()
force_data['ForceType'].unique()
fig = px.pie(force_data, force_data['ForceType'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)



fig.update_layout(title={

                  'text': "Type of Force by Police",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                  template='plotly_white')



fig.update_traces(textposition='inside', textinfo='percent+label')



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"



fig.show()
force_race = force_data.groupby(['Race'])[['ForceType']].count().reset_index()

force_race.sort_values(by='ForceType', ascending=False).style.background_gradient(cmap='summer', subset=['ForceType'])
fig = px.pie(force_race, force_race['Race'], force_race['ForceType'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)



fig.update_layout(title={

                  'text': "Force Type Based on Race",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                  template='plotly_white')



fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.1])



fig.data[0].marker.line.width = 2

fig.data[0].marker.line.color = "black"



fig.show()
force_data['ForceTypeAction'].unique()
fig = px.bar(force_data['ForceTypeAction'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)



fig.update_layout(title={

                  'text': "Action of Force Type",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title={'text': "Action"}, 

                  yaxis_title="# Police Violence",

                  height=1000,

                  showlegend=False

                 )



fig.show()
white_action = force_data[force_data['Race'] == 'White']

fig = px.bar(white_action['ForceTypeAction'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)



fig.update_layout(title={

                  'text': "Action of Force Type on White People",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title={'text': "Action"}, 

                  yaxis_title="# Police Violence",

                  height=1000,

                  showlegend=False

                 )



fig.show()
white_action['ForceTypeAction'].value_counts()
black_action = force_data[force_data['Race'] == 'Black']

fig = px.bar(black_action['ForceTypeAction'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)



fig.update_layout(title={

                  'text': "Action of Force Type on Black People",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title={'text': "Action"}, 

                  yaxis_title="# Police Violence",

                  height=1000,

                  showlegend=False

                 )



fig.show()
black_action['ForceTypeAction'].value_counts()
print(f"Total number of unique Problems: {force_data['Problem'].nunique()}")
fig = px.bar(force_data['Problem'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)



fig.update_layout(title={

                  'text': "Problem Reported by Police",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Problem", 

                  yaxis_title="Count",

                  height=700,

                  showlegend=False

                 )



fig.show()
fig = px.pie(force_race, force_data['SubjectInjury'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)



fig.update_layout(title={

                  'text': "# Times Subject Was Injured",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                  template='plotly_white')



fig.update_traces(textposition='inside', textinfo='percent+label')



fig.data[0].marker.line.width = 2

fig.data[0].marker.line.color = "black"



fig.show()
fig = px.pie(force_data, force_data['Sex'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)



fig.update_layout(title={

                  'text': "Gender of Subject",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                  template='plotly_white')



fig.update_traces(textposition='inside', textinfo='percent+label')



fig.data[0].marker.line.width = 2

fig.data[0].marker.line.color = "black"



fig.show()
fig = px.pie(force_data, force_data['EventAge'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.6)



fig.update_layout(title={

                  'text': "Age of Subject",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'},

                   height=600,

                  template='plotly_white', 

                  showlegend=False)



fig.update_traces(textposition='inside', textinfo='percent+label')



fig.data[0].marker.line.width = 2

fig.data[0].marker.line.color = "black"



fig.show()
fig = px.bar(force_data['TypeOfResistance'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)



fig.update_layout(title={

                  'text': "Type of Resistance by Subject",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Type", 

                  yaxis_title="Count",

                  height=700,

                  showlegend=False

                 )



fig.show()
fig = px.bar(force_data['Neighborhood'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)



fig.update_layout(title={

                  'text': "Neighbourhood of Incident",

                  'y':0.98,

                  'x':0.5,

                  'xanchor': 'center',

                  'yanchor': 'top'}, 

                  template='ggplot2', 

                  xaxis_title="Neighbourhood", 

                  yaxis_title="# of Incidents",

                  height=700,

                  showlegend=False

                 )



fig.show()