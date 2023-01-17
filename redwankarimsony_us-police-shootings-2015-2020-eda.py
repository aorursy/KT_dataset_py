import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

sns.set(rc={'figure.figsize':(11.7,8.27)})

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots



#Calendar Heatmap

!pip install calmap

import calmap





data=pd.read_csv('../input/us-police-shootings/shootings.csv')
data.info()
data.head(10)
df=data['gender'].value_counts().reset_index().rename(columns={'index':'gender','gender':'count'})

fig = go.Figure([go.Pie(labels=['Male', 'Female'],values=df['count'], hole = 0.5)])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial')

fig.update_layout(title="Male to Female Ratio in Shootings",title_x=0.5)

fig.show()
# data generation code

data['date']=pd.to_datetime(data['date'])

data['year']=pd.to_datetime(data['date']).dt.year



shoot_gender=data.groupby(['year','gender']).agg('count')['id'].to_frame(name='count').reset_index()

shoot_gender_male=shoot_gender.loc[shoot_gender['gender']=='M']

shoot_gender_female=shoot_gender.loc[shoot_gender['gender']=='F']



# plotting part

male=go.Bar(x=shoot_gender_male['year'],y=shoot_gender_male['count'],marker=dict(color='brown'),name="male")

female=go.Bar(x=shoot_gender_female['year'],y=shoot_gender_female['count'],marker=dict(color='orange'),name="female")

data_genderwise =[male,female]



fig = go.Figure(data_genderwise)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title="Gender Ratio vs Years",title_x=0.5,xaxis=dict(title="Year"),yaxis=dict(title="Number of Shootings"), barmode="group")

fig.show()
df=data['race'].value_counts().reset_index().rename(columns={'index':'race','race':'count'})



fig = go.Figure(go.Bar(x=df['race'],y=df['count'],

                       marker={'color': df['count'], 'colorscale': 'Viridis'},  

))

fig.update_layout(title_text='Distribution of Races in Shoootings',xaxis_title="Race",yaxis_title="Number of Shootings")

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.show()
fig = go.Figure([go.Pie(labels=df['race'],values=df['count'], hole = 0.4)])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial')

fig.update_layout(title="Racial Propotions in Shootings",title_x=0.5)

fig.show()
df=data.groupby('date')['manner_of_death'].count().reset_index()

df['date']=pd.to_datetime(df['date'])

df['year-month'] = df['date'].apply(lambda x: str(x.year) + '-' + str(x.month))

df_ym=df.groupby('year-month')[['manner_of_death']].sum().reset_index()

df_ym['year-month']=pd.to_datetime(df_ym['year-month'])

df_ym=df_ym.sort_values('year-month')







fig = go.Figure(go.Bar(

    x=df_ym['year-month'],y=df_ym['manner_of_death'],

    marker={'color': df_ym['manner_of_death'], 'colorscale': 'Viridis'},  

    text=df_ym['manner_of_death'],

    textposition = "outside",

))

fig.update_layout(title_text='No of deaths (2015-2020)',yaxis_title="no. of deaths", xaxis_title = 'Time')

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.show()
hist_data = [data['age'].values]

group_labels = ['distribution'] # name of the dataset



fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(title_text='Distribution of age of all',xaxis_title="Age",yaxis_title="Probability")

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.show()
import plotly.figure_factory as ff



# Add histogram data

age_white = data[data['race'] =='White'].age.values

age_black = data[data['race'] =='Black'].age.values

age_hispanic = data[data['race'] =='Hispanic'].age.values

age_asian = data[data['race'] =='Asian'].age.values

age_native = data[data['race'] =='Native'].age.values

age_other = data[data['race'] =='Other'].age.values



# Group data together

hist_data = [age_white, age_black, age_hispanic, age_asian, age_native, age_other]



group_labels = ['White', 'Black', 'Hispanic', 'Asian', 'Native', 'Other']



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.show()
fig = go.Figure()

fig.add_trace(go.Box(y=data[data['race'] =='White'].age.values , name='White', marker_color = 'gray',boxmean=True))

fig.add_trace(go.Box(y=data[data['race'] =='Black'].age.values , name='Black', marker_color = 'brown',boxmean=True))

fig.add_trace(go.Box(y=data[data['race'] =='Hispanic'].age.values , name='Hispanic', marker_color = 'green',boxmean=True))

fig.add_trace(go.Box(y=data[data['race'] =='Asian'].age.values , name='Asian', marker_color = 'red',boxmean=True))

fig.add_trace(go.Box(y=data[data['race'] =='Native'].age.values , name='Native', marker_color = 'orange',boxmean=True))

fig.add_trace(go.Box(y=data[data['race'] =='Other'].age.values , name='Other', marker_color = 'violet',boxmean=True))

fig.update_layout(title_text='Age Distribution Race Wise', xaxis_title= "Race", yaxis_title="Age")

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.show()
fig = go.Figure()

fig.add_trace(go.Box(y=data[data['gender'] =='M'].age.values , name='Male', marker_color = 'blue',boxmean=True))

fig.add_trace(go.Box(y=data[data['gender'] =='F'].age.values , name='Female', marker_color = 'red',boxmean=True))

fig.update_layout(title_text='Age Distribution Gender wise',

                  xaxis_title= "Gender",

                  yaxis_title="Age")

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.show()
df=data['armed'].value_counts().reset_index().rename(columns={'index':'weapons used','armed':'count'})

fig = go.Figure([go.Pie(labels=df['weapons used'],values=df['count'], hole=0.5)])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial')

fig.update_layout(title="Different Weapons used in Shootings",title_x=0.5)

fig.show()



df=data['state'].value_counts().reset_index().rename(columns={'index':'state','state':'deaths'})



fig = go.Figure(go.Bar(x=df['state'], y=df['deaths'],

                       marker={'color': df['deaths'], 'colorscale': 'Viridis'},

                       text=df['deaths'],

                       textposition = "outside"))



fig.update_layout(title_text='Statewise Number of Deaths',xaxis_title="State",yaxis_title="Number of Shootings")

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.show()
df=data['signs_of_mental_illness'].value_counts().reset_index().rename(columns={'index':'signs_of_mental_illness','signs_of_mental_illness':'count'})

fig = go.Figure([go.Pie(labels=df['signs_of_mental_illness'],values=df['count'], hole = 0.5)])



fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial')



fig.update_layout(title="Signs_of_mental_illness",title_x=0.5)

fig.show()
#processing part

race_illness = data.groupby(by=["race", "signs_of_mental_illness"]).count()["id"].unstack()



# plotting part

no_illness = go.Bar(x=race_illness.index.values, y=race_illness.loc[:, 0].values ,marker=dict(color='green'),name="No Illness")

illness =go.Bar(x=race_illness.index.values, y=race_illness.loc[:, 1].values  ,marker=dict(color='orange'),name="Illness Present")

data_genderwise =[no_illness, illness]



fig = go.Figure(data_genderwise)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_layout(title="Racewise Mental Conditions",title_x=0.5,xaxis=dict(title="Race"),

                  yaxis=dict(title="Number of Shootings"), barmode="group")

fig.show()
df=data.groupby(['date','gender','race'])['manner_of_death'].count().reset_index()

df['date']=pd.to_datetime(df['date'])

df['year-month'] = df['date'].apply(lambda x: str(x.year))

df_ym=df.groupby(['year-month','gender','race'])[['manner_of_death']].sum().reset_index()

df_ym['year-month']=pd.to_datetime(df_ym['year-month'])

df_ym=df_ym.sort_values('year-month')

df_ym['year-month']=df_ym['year-month'].astype('str').apply(lambda x: x.split('-')[0])



fig = px.sunburst(df_ym, path=['year-month','gender','race'], values='manner_of_death')

fig.update_layout(title="Number of deaths  by Gender,year,race",title_x=0.5)

fig.show()
df=data['threat_level'].value_counts().reset_index().rename(columns={'index':'threat_level','threat_level':'count'})

fig = go.Figure([go.Pie(labels=df['threat_level'],values=df['count'], hole = 0.5)])



fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial')

fig.update_layout(title="threat_level",title_x=0.5)

fig.show()
data.groupby(by=["race", "threat_level"]).count()["id"].unstack().plot.bar(stacked=False, title = "Racewise Threat Level", xlabel="Race",  ylabel="Threat Counts")
df=data['state'].value_counts().reset_index().rename(columns={'index':'state','state':'deaths'}).head(10)

fig = go.Figure(go.Bar(x=df['state'], y=df['deaths'],

                       marker={'color': df['deaths'], 'colorscale': 'Viridis'},

                       text=df['deaths'],

                       textposition = "outside"))



fig.update_layout(title_text='Top 10 States involved  in Police Shooting in the US',xaxis_title="States",yaxis_title="Number of Shootings")

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.show()
df=data['state'].value_counts().reset_index().rename(columns={'index':'state','state':'deaths'}).iloc[::-1].tail(10)

fig = go.Figure(go.Bar(x=df['state'], y=df['deaths'],

                       marker={'color': df['deaths'], 'colorscale': 'Viridis'},

                       text=df['deaths'],

                       textposition = "outside"))



fig.update_layout(title_text='Last 10 States involved  in Police Shooting in the US',xaxis_title="States",yaxis_title="Number of Shootings")

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.show()
df=data['city'].value_counts().reset_index().rename(columns={'index':'city','city':'deaths'}).head(10)

fig = go.Figure(go.Bar(x=df['city'], y=df['deaths'],

                       marker={'color': df['deaths'], 'colorscale': 'Viridis'},

                       text=df['deaths'],textposition = "outside"))



fig.update_layout(title_text='Top 10 Cities involved  in Police Shooting in the US',xaxis_title="Cities",yaxis_title="Number of Shootings")

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

fig.show()
black_state=data[data['race']=='Black']['state'].value_counts().to_frame().reset_index().rename(columns={'index':'state','state':'count'})



fig = go.Figure(go.Choropleth(

    locations=black_state['state'],

    z=black_state['count'].astype(float),

    locationmode='USA-states',

    colorscale='Reds',

    autocolorscale=False,

    text=black_state['state'], # hover text

    marker_line_color='white', # line markers between states

    colorbar_title="Millions USD",showscale = False,

))

fig.update_layout(title_text='US Police shooting cases of black people',    title_x=0.5,

    geo = dict( scope='usa', projection=go.layout.geo.Projection(type = 'albers usa'), showlakes=True, 

               lakecolor='rgb(255, 255, 255)'))

fig.update_layout(template="simple_white")

fig.show()
white_state=data[data['race']=='White']['state'].value_counts().to_frame().reset_index().rename(columns={'index':'state','state':'count'})



fig = go.Figure(go.Choropleth(

    locations=white_state['state'],

    z=white_state['count'].astype(float),

    locationmode='USA-states',

    colorscale='Greens',

    autocolorscale=False,

    text=black_state['state'], # hover text

    marker_line_color='white', # line markers between states

    colorbar_title="Millions USD",showscale = False,

))

fig.update_layout(title_text='US Police shooting cases of White people',    title_x=0.5,

    geo = dict( scope='usa', projection=go.layout.geo.Projection(type = 'albers usa'), showlakes=True, 

               lakecolor='rgb(255, 255, 255)'))

fig.update_layout(template="simple_white")

fig.show()