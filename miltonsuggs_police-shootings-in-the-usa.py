# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np 

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import missingno as msno

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')

df.columns
df.head()
df.info()
df.describe(include='all')
missing_values = df.isnull()

missing_values.head()
for column in missing_values.columns.tolist():

    print(column)

    print(missing_values[column].value_counts())

    print('')
missing_percentage = (missing_values.sum()*100)/df.shape[0]

missing_percentage
msno.matrix(df)
#DROP NULL VALUES



df.dropna(inplace=True)
cardinality={}

for col in df.columns:

    cardinality[col] = df[col].nunique()



cardinality
print('MANNER OF DEATH')

print(df['manner_of_death'].unique())

print('-'*40)

print('RACE')

print(df['race'].unique())

print('-'*40)

print('THREAT LEVEL')

print(df['threat_level'].unique())

print('-'*40)

print('FLEE')

print(df['flee'].unique())



# SEPARATE DAY, MONTH, YEAR INTO INDIVIDUAL COLUMNS

df['date']=pd.to_datetime(df['date'])

df['year']=pd.to_datetime(df['date']).dt.year

df['month']=pd.to_datetime(df['date']).dt.month

df['month_name']=df['date'].dt.strftime('%B')

df['month_num']=df['date'].dt.strftime('%m')

df['weekday']=df['date'].dt.strftime('%A')  

df['date_num']=df['date'].dt.strftime('%d').astype(int)

df['year_month']=df.date.dt.to_period("M")



# CLASSIFY VICTIM AGES INTO AGE RANGE GROUPS

df['age_range']=np.where(df['age']<18,'<18',np.where((df['age']>=18)&(df['age']<=35),'18-35',

np.where((df['age']>=36)&(df['age']<=50),'36-50',np.where(df['age']>65,'65+',

np.where((df['age']>=51)&(df['age']<=65),'51-65',"Not Specified")))))



# CHANGE ORDER OF COLUMNS

cols = ['id', 'name', 'age', 'age_range', 'gender', 'race', 'manner_of_death', 'armed', 'flee', 

        'signs_of_mental_illness', 'threat_level', 'body_camera', 'city', 'state',

        'date', 'date_num', 'year', 'year_month', 'month', 'month_name', 'month_num', 'weekday']

df=df[cols]

df.head(3)
df.info()
fig = ff.create_distplot([df['age']], ['age'], bin_size=5, colors=['blue'])

fig.update_layout(title_text="Distribution of Age", title_x=0.5)

fig.show()
fig = px.pie(df, values='age', names='age_range')

fig.update_layout(title_text="Distribution of Age Ranges", title_x=0.5)

fig.show()

fig = px.histogram(df, x='gender', color='gender')

fig.update_layout(title_text='Value Count of Gender', title_x=0.5)

fig.show()
fig = px.histogram(df, x=df['race'], color='race')

fig.update_layout(title_text='Distribution of Race', title_x=0.5)

fig.show()
fig = px.histogram(df, x='manner_of_death', color='manner_of_death')

fig.update_layout(title_text='Manner of Death', title_x=0.5)

fig.show()
top_armed = df['armed'].value_counts().to_frame()

top_armed.reset_index(inplace=True)

top_armed = top_armed.rename(columns={'index':'armed', 'armed':'count'})



fig = px.histogram(top_armed[0:15], x='armed', y='count', color='armed')



fig.update_layout(title_text='Weapon of Victim', title_x=0.5)

fig.show()
fig = px.histogram(df, x='flee', color='flee')

fig.update_layout(title_text='Was Victim Fleeing?', title_x=0.5)

fig.show()
mental_illness = df['signs_of_mental_illness'].value_counts().to_frame().reset_index().rename(columns={'index':'mental_illness','signs_of_mental_illness':'count'})



fig = px.histogram(df, x='signs_of_mental_illness', color='signs_of_mental_illness')



fig.update_layout(title_text='Signs of Mental Illness', title_x=0.5)

fig.show()
#THREAT LEVEL

fig = px.histogram(df, x='threat_level', color='threat_level')

fig.update_layout(title_text='Threat Level of Victim', title_x=0.5)

fig.show()
#BODY CAMERA

fig = px.histogram(df, x='body_camera', color='body_camera')

fig.update_layout(title_text="Was Officer's Body Camera On?", title_x=0.5)

fig.show()
#STATE WHERE SHOOTINGS TOOK PLACE

states = df['state'].value_counts().to_frame().reset_index()

states.rename(columns={'index':'state', 'state':'count'}, inplace=True)

# states = states.sort_values(by='count', ascending=False)

states





fig = go.Figure(go.Bar(y=states['state'].sort_index(ascending=False), 

                       x=states['count'].sort_index(ascending=False),

                       orientation='h', text=states['count'].sort_index(ascending=False),

                       textposition='outside', marker_color=states['count'].sort_index(ascending=False)))





fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='Police Killings, Organized by States',yaxis_title='States',

                 xaxis_title='Total number of victims', title_x=0.5, height=1000)



fig.show()
df_years = df['year'].value_counts().to_frame().reset_index()

df_years.rename(columns={'index':'year', 'year':'count'}, inplace=True)

df_years = df_years.sort_values(by='year')

df_years
fig = go.Figure()



fig.add_trace(go.Scatter(x=df_years['year'], y=df_years['count'],

                mode='lines+markers',

                marker_color="red"))



fig.update_layout(title_text='Police Killings by Year',xaxis_title='Years',

                 yaxis_title='Total number of kills', title_x=0.5)



fig.show()

df_monthly = df['date'].groupby(df.date.dt.to_period("M")).agg('count').to_frame(name="count").reset_index()

df_monthly = df_monthly.sort_values(by='date')



year_month=[]

for i in df_monthly['date']:

    year_month.append(str(i))

    

df_monthly.head()
fig = make_subplots(rows=2, cols=1, subplot_titles=("Monthly series", "Distribution of monthly count"))



fig.add_trace(go.Scatter(x=year_month, y=df_monthly['count'], 

                         name="Monthly Deaths", mode='lines+markers'),row=1,col=1)



fig.add_trace(go.Box(y=df_monthly['count'], name='Count',

                marker_color = 'indianred',boxmean='sd'),row=2,col=1)



fig.update_xaxes(title_text="Year", row=1, col=1,showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_xaxes(title_text=" ", row=2, col=1,showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(title_text="Number of Victims", row=1, col=1,showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(title_text="Number of Victims", row=2, col=1,showline=True, linewidth=2, linecolor='black', mirror=True)



fig.update_layout(title_text='Fatal Killing Monthly Count 2015 - 2020', title_x=0.5,showlegend=False,height=1000)

fig.show()
df.head(1)
df_monthly['year'] = df_monthly['date'].dt.strftime('%Y')



def plot_month(year, color):

    temp_month = []

    for i in df_monthly.loc[df_monthly['year']==year]['date']:

        temp_month.append(str(i))

    trace=go.Bar(x=temp_month, y=df_monthly.loc[df_monthly['year']==year]['count'], 

                 name=year, marker_color=color)

    return trace
fig = make_subplots(rows=3, cols=2, subplot_titles=('2015', '2016', '2017', '2018', '2019', '2020'))



fig.add_trace(plot_month('2015', 'blue'), row=1, col=1)

fig.add_trace(plot_month('2016', 'red'), row=1, col=2)

fig.add_trace(plot_month('2017', 'green'), row=2, col=1)

fig.add_trace(plot_month('2018', 'orange'), row=2, col=2)

fig.add_trace(plot_month('2019', 'purple'), row=3, col=1)

fig.add_trace(plot_month('2020', 'teal'), row=3, col=2)



fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title_text='Distribution of Monthly Killings by Year', title_x=0.5, showlegend=False)

fig.show()
only_month = df.groupby(['month_name','month'])[['month_name']].agg('count')

only_month.rename(columns={'month_name':'count'}, inplace=True)

only_month.reset_index(inplace=True)

only_month.sort_values(by='month', inplace=True)

only_month
fig = go.Figure(data=[go.Bar(x=only_month['month_name'], y=only_month['count'], 

                             name='Months', marker_color='blue')])



fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_layout(title_text='Deaths - All Months',xaxis_title='Months',

                 yaxis_title='Total number of kills', title_x=0.5,barmode='stack')



fig.show()
year_count = df.groupby(['year'])[['id']].agg('count')

year_count.reset_index(inplace=True)

year_count.rename(columns={'id':'count'}, inplace=True)

year_count.head()
fig = go.Figure(data=[go.Bar(x=year_count['year'], y=year_count['count'], 

                             name='Months', marker_color='blue')])



fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_layout(title_text='Deaths - All Years',xaxis_title='Months',

                 yaxis_title='Total number of kills', title_x=0.5,barmode='stack')

fig.show()
df.head(1)
weekday_count = df.groupby(['weekday'])[['id']].agg('count')

weekday_count = weekday_count.reindex(['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])

weekday_count.reset_index(inplace=True)

weekday_count.rename(columns={'id':'count'}, inplace=True)

weekday_count.head(7)

fig = go.Figure(data=[go.Bar(x=weekday_count['weekday'], y=weekday_count['count'],

                            name='Weekdays', marker_color='blue')])



fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

fig.update_layout(title_text='Deaths - Days of the Week',xaxis_title='Weekdays',

                 yaxis_title='Total number of kills', title_x=0.5,barmode='stack')

fig.show()
pd.pivot_table(df, index = 'body_camera', columns = 'flee', values = 'id',aggfunc ='count')
df.head(1)
pd.pivot_table(df, index = 'race', columns = 'age_range', values = 'id',aggfunc ='count')
df_race_age = df.groupby(['race', 'age_range']).agg('count')['id'].to_frame('count').reset_index()

df_black = df_race_age.loc[df_race_age['race'] == 'B']

df_white = df_race_age.loc[df_race_age['race'] == 'W']

df_hispanic = df_race_age.loc[df_race_age['race'] == 'H']

df_native = df_race_age.loc[df_race_age['race'] == 'N']

df_asian = df_race_age.loc[df_race_age['race'] == 'A']

df_other = df_race_age.loc[df_race_age['race'] == 'O']
black = go.Bar(x = df_black['age_range'], y = df_black['count'], 

             marker=dict(color='black'),name="black")

white = go.Bar(x=df_white['age_range'],y=df_white['count'],

               marker=dict(color='pink'),name="white")

hispanic = go.Bar(x=df_hispanic['age_range'],y=df_hispanic['count'],

               marker=dict(color='tan'),name="hispanic")

asian = go.Bar(x=df_asian['age_range'],y=df_asian['count'],

               marker=dict(color='yellow'),name="asian")

native = go.Bar(x=df_native['age_range'],y=df_native['count'],

               marker=dict(color='red'),name="native")

other = go.Bar(x=df_other['age_range'],y=df_other['count'],

               marker=dict(color='teal'),name="other")



data=[white,black,hispanic,asian,native,other]



fig = go.Figure(data)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title="Death Toll - Race & Age Range",title_x=0.5,xaxis=dict(title="Year"),yaxis=dict(title="Number of Victims"),

                   barmode="group")

fig.show()
df_race_gender = df.groupby(['race', 'gender']).agg('count')['id'].to_frame('count').reset_index()



df_black_gender = df_race_gender.loc[df_race_gender['race'] == 'B']

df_black_gender = df_black_gender.sort_values(by='count', ascending=False)



df_white_gender = df_race_gender.loc[df_race_gender['race'] == 'W']

df_white_gender = df_white_gender.sort_values(by='count', ascending=False)



df_hispanic_gender = df_race_gender.loc[df_race_gender['race'] == 'H']

df_hispanic_gender = df_hispanic_gender.sort_values(by='count', ascending=False)



df_asian_gender = df_race_gender.loc[df_race_gender['race'] == 'A']

df_asian_gender = df_asian_gender.sort_values(by='count', ascending=False)



df_native_gender = df_race_gender.loc[df_race_gender['race'] == 'N']

df_native_gender = df_native_gender.sort_values(by='count', ascending=False)



df_other_gender = df_race_gender.loc[df_race_gender['race'] == 'O']

df_other_gender = df_other_gender.sort_values(by='count', ascending=False)





black = go.Bar(x=df_black_gender['gender'], y=df_black_gender['count'], 

              marker=dict(color='black'),name="black", 

              text=df_black_gender['count'], textposition='outside')



white = go.Bar(x=df_white_gender['gender'], y=df_white_gender['count'], 

              marker=dict(color='pink'),name="white",

              text=df_white_gender['count'], textposition='outside')



hispanic = go.Bar(x=df_hispanic_gender['gender'], y=df_hispanic_gender['count'], 

              marker=dict(color='tan'),name="hispanic",

            text=df_hispanic_gender['count'], textposition='outside')



asian = go.Bar(x=df_asian_gender['gender'], y=df_asian_gender['count'], 

              marker=dict(color='yellow'),name="asian",

              text=df_asian_gender['count'], textposition='outside')



native = go.Bar(x=df_native_gender['gender'], y=df_native_gender['count'], 

              marker=dict(color='red'),name="native",

               text=df_native_gender['count'], textposition='outside')



other = go.Bar(x=df_other_gender['gender'], y=df_other_gender['count'], 

              marker=dict(color='teal'),name="other",

              text=df_other_gender['count'], textposition='outside')



data=[white,black,hispanic,asian,native,other]



fig = go.Figure(data)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title="Death Toll - Race & Gender",title_x=0.5,xaxis=dict(title="Year"),yaxis=dict(title="Number of Victims"),

                   barmode="group")

fig.show()



df_race_loc = df.groupby(['race','state']).agg('count')['id'].to_frame('count').reset_index()

df_race_loc = df_race_loc.sort_values(by='state').reset_index()

df_race_loc
race_mental = df.groupby(['race','signs_of_mental_illness']).agg('count')['id'].to_frame('count').reset_index()



black_mental = race_mental.loc[race_mental['race'] == 'B']

black_mental = black_mental.sort_values(by='count', ascending=False)



white_mental = race_mental.loc[race_mental['race'] == 'W']

white_mental = white_mental.sort_values(by='count', ascending=False)



hispanic_mental = race_mental.loc[race_mental['race'] == 'H']

hispanic_mental = hispanic_mental.sort_values(by='count', ascending=False)



asian_mental = race_mental.loc[race_mental['race'] == 'A']

asian_mental = asian_mental.sort_values(by='count', ascending=False)



native_mental = race_mental.loc[race_mental['race'] == 'N']

native_mental = native_mental.sort_values(by='count', ascending=False)



other_mental = race_mental.loc[race_mental['race'] == 'O']

other_mental = other_mental.sort_values(by='count', ascending=False)
black = go.Bar(x=black_mental['signs_of_mental_illness'], y=black_mental['count'],

              marker=dict(color='black'),name="black")

white = go.Bar(x=white_mental['signs_of_mental_illness'], y=white_mental['count'],

              marker=dict(color='pink'),name="white")

hispanic = go.Bar(x=hispanic_mental['signs_of_mental_illness'], y=hispanic_mental['count'],

              marker=dict(color='tan'),name="hispanic")

asian = go.Bar(x=asian_mental['signs_of_mental_illness'], y=asian_mental['count'],

              marker=dict(color='yellow'),name="asian")

native = go.Bar(x=native_mental['signs_of_mental_illness'], y=native_mental['count'],

              marker=dict(color='red'),name="native")

other = go.Bar(x=other_mental['signs_of_mental_illness'], y=other_mental['count'],

              marker=dict(color='teal'),name="other")



data = [white,black,hispanic,asian,native,other]



fig = go.Figure(data)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title="Death Toll - Race & Mental Illness", title_x=0.5,

                  xaxis=dict(title="Signs of Mental Illness"),

                  yaxis=dict(title="Number of Victims"),

                   barmode="group")

fig.show()
race_threat = df.groupby(['race','threat_level','flee']).agg('count')['id'].to_frame('count').reset_index()



black_threat = race_threat.loc[race_threat['race'] == 'B']

black_threat = black_threat.sort_values(by='count', ascending=False)



white_threat = race_threat.loc[race_threat['race'] == 'W']

white_threat = white_threat.sort_values(by='count', ascending=False)



hispanic_threat = race_threat.loc[race_threat['race'] == 'H']

hispanic_threat = hispanic_threat.sort_values(by='count', ascending=False)



asian_threat = race_threat.loc[race_threat['race'] == 'A']

asian_threat = asian_threat.sort_values(by='count', ascending=False)



native_threat = race_threat.loc[race_threat['race'] == 'N']

native_threat = native_threat.sort_values(by='count', ascending=False)



other_threat = race_threat.loc[race_threat['race'] == 'O']

other_threat = other_threat.sort_values(by='count', ascending=False)
black = go.Bar(x=black_threat['threat_level'], y=black_threat['count'],

              marker=dict(color='black'),name="black")

white = go.Bar(x=white_threat['threat_level'], y=white_threat['count'],

              marker=dict(color='pink'),name="white")

hispanic = go.Bar(x=hispanic_threat['threat_level'], y=hispanic_threat['count'],

              marker=dict(color='tan'),name="hispanic")

asian = go.Bar(x=asian_threat['threat_level'], y=asian_threat['count'],

              marker=dict(color='yellow'),name="asian")

native = go.Bar(x=native_threat['threat_level'], y=native_threat['count'],

              marker=dict(color='red'),name="native")

other = go.Bar(x=other_threat['threat_level'], y=other_threat['count'],

              marker=dict(color='teal'),name="other")



data = [white, black, hispanic, asian, native, other]



fig = go.Figure(data)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title="Death Toll - Race & Threat Level",title_x=0.5,

                  xaxis=dict(title="Threat Level"),

                  yaxis=dict(title="Number of Victims"),

                   barmode="group")

fig.show()