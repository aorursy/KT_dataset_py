# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

import plotly.io as pio

import plotly.figure_factory as ff





from wordcloud import WordCloud

from seaborn import countplot



import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/us-police-shootings/shootings.csv',parse_dates=['date'])



df['year'] = pd.to_datetime(df['date']).dt.year

df['month'] = pd.to_datetime(df['date']).dt.month

print('Shape of the dataset',df.shape)
df.columns
df.info()
df.isnull().sum()
df.describe()
df.describe(include='O')
df.head()
df.tail()
df.nunique()
df.name.value_counts().sort_values(ascending=False).head(20)
df_tk = df[df['name']=='TK TK']

df_tk.head()
values = df.manner_of_death.value_counts().values

labels = ['Shot', 'Shot & Tasered']



fig = px.pie(df,labels=labels,values=values,title='Manner of Neutralizing',names= labels)

fig.update_traces(textfont_size=15,textposition='inside', textinfo='percent+label+value',pull=[0.2, 0]

                 ,hole=.4,insidetextorientation='radial')



#fig.update_layout(height=500,width=500)



fig.show()
data = [dict(

  x = df['date'],

  autobinx = False,

  autobiny = True,

  marker = dict(color = 'rgb(68, 68, 68)'),

  name = 'year',

  type = 'histogram',

  xbins = dict(

    end = '2020-12-31',

    size = 'M1',

    start = '2015-01-01'

  )

)]



layout = dict(

  #paper_bgcolor = 'rgb(240, 240, 240)',

  #plot_bgcolor = 'rgb(240, 240, 240)',

  title = '<b>Shooting Incidents 2015-2020</b>',

  xaxis = dict(

    title = '',

    type = 'date'

  ),

  yaxis = dict(

    title = 'Shootings Incidents 2015-2020',

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

            args = ['xbins.size', 'D1'],

            label = 'Day',

            method = 'restyle',

        ), dict(

            args = ['xbins.size', 'M1'],

            label = 'Month',

            method = 'restyle',

        ), dict(

            args = ['xbins.size', 'M3'],

            label = 'Quater',

            method = 'restyle',

        ), dict(

            args = ['xbins.size', 'M6'],

            label = 'Half Year',

            method = 'restyle',

        ), dict(

            args = ['xbins.size', 'M12'],

            label = 'Year',

            method = 'restyle',

        )]

  )]

)



fig_dict = dict(data=data, layout=layout)



pio.show(fig_dict, validate=False)
armed = df.groupby('armed')['armed'].count().sort_values(ascending=False).to_frame(name='count').reset_index()



fig = px.bar(armed, x='armed', y='count',title = 'Incidents with Armaments Type',color='armed')

fig.show()



armed.head(10).style.background_gradient(cmap='Reds')

plt.figure(figsize=(12,8))

wordcloud = WordCloud(collocations=False

                     ).generate_from_text('*'.join(df.armed))



plt.imshow(wordcloud,interpolation='bilinear')

plt.title('Most of the Suspects had')

plt.show()
# comparison of preferred foot over the male female ratio



plt.rcParams['figure.figsize'] = (10, 5)

countplot(df['gender'], palette = 'pink')

plt.title('Male Female distribution', fontsize = 20)

plt.show()
print('Maximum Age is ',df['age'].max())

print('Minimum Age is',df['age'].min())

print('Average age is ',round(df['age'].mean(),0))



fig = go.Figure(go.Box(y=df['age'],name='Age'))

fig.update_layout(title='Overall Distribution of Age',title_x=0.5)

fig.show()
df_male=df[df['gender']=='M']['age'].values

df_female=df[df['gender']=='F']['age'].values

labels=['Male','Female']



fig = ff.create_distplot([df_male, df_female], group_labels=labels, 

                         

                         )

fig.update_layout(title='Male-Female Age distribution',title_x=0.5)

fig.show()
fig = px.box(df, x="gender", y="age",points='all')

fig.update_layout(title=' Gender wise Age Distribution',title_x=0.5)



fig.show()
fig = px.box(df, x="race", y="age",color='race')

fig.update_layout(title=' Gender wise Age Distribution',title_x=0.5)

fig.show()
plt.figure(figsize=(12,8))

countplot(x='race',hue='gender',data=df)

plt.title('Race-Gender Wise Shooting')

plt.show()
df[df['age']<10]
df_elderly = df[df.age>80].sort_values(by='race',ascending=False)

df_elderly.style.background_gradient(cmap='winter_r')
df_race=df['race'].value_counts().reset_index().rename(columns={'index':'Race','race':'Counts'})



fig = px.bar(df_race, x=df_race['Race'], y=df_race['Counts'],title = 'Race with most number of Shooting by US Police',

             color=df_race['Counts'])

fig.update_layout(title_x=0.5)

fig.show()



df_race.head(10).style.background_gradient(cmap='prism')
#print('Unique State in the dataset are:\n',df['state'].unique())



df_state = df['state'].value_counts().reset_index().rename(columns={'index':'State','state':'Counts'})





fig= go.Figure(go.Bar(x=df_state['State'],

              y=df_state['Counts'],marker={'color':'darkturquoise'}

              ))

fig.update_layout(title='Number of Deaths State Wise',title_x=0.5)

fig.show()
df_city = df['city'].value_counts().reset_index().rename(columns={'index':'City','city':'Counts'}).head(10)





fig= go.Figure(go.Bar(x=df_city['City'],

              y=df_city['Counts'],marker={'color': 'darksalmon'}

              ))

fig.update_layout(title='Number of Deaths City Wise',title_x=0.5)

fig.show()
values = df.signs_of_mental_illness.value_counts().values

labels = ['Not Mentally Ill','Mentally Ill']



fig = px.pie(df,labels=labels,values=values,title='Mentally Stable',names= labels)

fig.update_traces(textfont_size=15,textposition='inside', textinfo='percent+label+value',pull=[0.2, 0]

                 ,hole=.4,insidetextorientation='radial')



#fig.update_layout(height=500,width=500)



fig.show()
values = df.threat_level.value_counts().values

labels = ['Attack', 'Other' ,'Undetermined']



fig = px.pie(df,labels=labels,values=values,title='Threat Category',names= labels)

fig.update_traces(textfont_size=15,textposition='inside', textinfo='percent+label+value'

                 ,hole=.4,insidetextorientation='radial')



#fig.update_layout(height=500,width=500)



fig.show()
values = df.flee.value_counts().values

labels = df.flee.value_counts().index



fig = px.pie(df,labels=labels,values=values,title='Fleeing or not?',names= labels)

fig.update_traces(textfont_size=15,textposition='inside', textinfo='percent+label+value'

                 ,hole=.4,insidetextorientation='radial')



#fig.update_layout(height=500,width=500)



fig.show()
fig = px.sunburst(df, path=['year', 'race', 'gender'], values='age')

fig.update_layout(title='Detailed chart US Shooting 2015-2020',title_x=0.5)

fig.show()
black_df = df[df['race']=='Black']['state'].value_counts().to_frame().reset_index().rename(columns={'index':'state','state':'count'})    

fig = go.Figure(go.Choropleth(    

    locations=black_df['state'],

    z=black_df['count'].astype(float),

    locationmode='USA-states',

    colorscale='Reds',

    autocolorscale=False,

    text=black_df['state'], 

    marker_line_color='white', 

   

))

fig.update_layout(

    title_text='US Police shooting of Black people',

    title_x=0.5,

    geo = dict(

        scope='usa',

        projection=go.layout.geo.Projection(type = 'albers usa'),

        showlakes=True, # lakes

        lakecolor='rgb(255, 255, 255)'))

fig.update_layout(

    template="plotly_dark")

fig.show()
white_df = df[df['race']=='White']['state'].value_counts().to_frame().reset_index().rename(columns={'index':'state','state':'count'})    

fig = go.Figure(go.Choropleth(    

    locations=white_df['state'],

    z=white_df['count'].astype(float),

    locationmode='USA-states',

    colorscale='Reds',

    autocolorscale=False,

    text=black_df['state'], 

    marker_line_color='white', 

   

))

fig.update_layout(

    title_text='US Police shooting cases of White People',

    title_x=0.5,

    geo = dict(

        scope='usa',

        projection=go.layout.geo.Projection(type = 'albers usa'),

        showlakes=True,

        lakecolor='rgb(255, 255, 255)'))

fig.update_layout(

    template="plotly_dark")

fig.show()