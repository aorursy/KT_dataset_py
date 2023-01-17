#importing libraries 

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import plot,iplot,init_notebook_mode

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

from plotly.subplots import make_subplots



%matplotlib inline
#reading the data

df = pd.read_csv('../input/videogamesales/vgsales.csv')

df.head()
df.info()
#dropping null values

df.dropna(how="any",inplace=True)

df.isnull().sum()
plt.figure(figsize=(12,8))

#creating an array having same shape as df.corr() 

mask = np.zeros_like(df.corr(),dtype= np.bool) 

#setting the upper triangle indices to True

mask[np.triu_indices_from(mask)] = True

#the mask function hides all the indices with value set to True from the heatmap

sns.heatmap(df.corr(),annot=True,linewidth=1,cmap="cool",mask=mask)

#this is a bit excessive and uneccessary but it looks neat this way.
#creating a dataframe to analyse the top 100 games

gamesales = df.iloc[:100,:]
px.scatter_3d(gamesales,x='Platform',y='Publisher',z='Rank',color='Genre',size ='Global_Sales',

              size_max=60,hover_data=['Name'],height=700)


sns.set(style= 'white',font_scale=1.2)

#creating a joint plot to take a closer look at sales over the years

sns.jointplot(kind = 'kde',y=gamesales['Global_Sales'],x=gamesales['Year'],height=8)
#creating a dataframe for year-wise sale analysis

sales =df.groupby('Year')[['Global_Sales','JP_Sales','EU_Sales','Other_Sales','NA_Sales']].sum()

sales.reset_index(inplace=True)



sales.head()
#https://plotly.com/python/creating-and-updating-figures/

#https://plotly.com/python/subplots/



#NA,EU,JP,Others Sale over the years



#initializing figure with subplots

fig = make_subplots(rows =1,cols =5,shared_yaxes=True,column_titles=["Global","NA","EU","JP","Other"])



#adding traces

fig.add_trace(

            go.Bar(

                name='Global',

                opacity=0.5,

                marker=dict(color = 'orange'),  

                orientation='h',

                x =sales['Global_Sales'],

               y =sales['Year']),

    row =1,col =1

)



fig.add_trace(

            go.Scatter(

                name='North America',

                marker=dict(color = 'rgb(25,56,34)'),

                mode ='lines+markers',

                x =sales['NA_Sales'],

                y =sales['Year']),

    row =1,col =2

)

fig.add_trace(

            go.Scatter(

                name='Europe',

                marker=dict(color = 'rgb(75,138,20)'),

                mode ='lines+markers',

                x =sales['EU_Sales'],

                y =sales['Year']),

    row =1,col =3

)

fig.add_trace(

            go.Scatter(

                name='Japan',

                marker=dict(color = 'rgb(157,167,21)'),

                mode ='lines+markers',

                x =sales['JP_Sales'],

                y =sales['Year']),

    row =1,col =4

)

fig.add_trace(

            go.Scatter(

                name='Others',

                marker=dict(color = 'rgb(194,182,59)'),

                mode ='lines+markers',

                x =sales['Other_Sales'],

                y =sales['Year']),

    row =1,col =5

)



#Updating layout

fig.update_layout(title=dict(text="Sales Over the Years",

                             x=0.45,

                             font =dict(family= "Franklin Gothic",size=25)), 

                             )



# Updating xaxis properties

fig.update_xaxes(tickangle=90, row=1, col=1)

fig.update_xaxes(tickangle=90, row=1, col=2)

fig.update_xaxes(tickangle=90, row=1, col=3)

fig.update_xaxes(tickangle=90, row=1, col=4)







#Creating a dataframe

genre = df['Genre'].value_counts().to_frame()

genre.reset_index(inplace = True)

genre.rename(columns={'index':"Genre",'Genre':'Count'},inplace = True)

genre.head()
figa =px.bar(data_frame= genre,y='Genre',x='Count',color = 'Count',color_continuous_scale='Plasma',orientation = 'h',text = 'Count')

#https://plotly.com/python/builtin-colorscales/

figa.update_layout(title = {'text': "Genre Count", 'x':0.5,'font':{'family': "Franklin Gothic",'size':25}})
figb =px.pie(data_frame=df,values = 'Global_Sales', names ='Genre',color_discrete_sequence=px.colors.qualitative.Set2)

#https://plotly.com/python/discrete-color/

figb.update_layout(title = {'text':"Distribution & Global Sales of Genre in the Database",'x':0.5,'font':dict(family="Franklin Gothic", size=25)})
#modifying the dataframe for plotting purpose

genre = df[['Global_Sales','NA_Sales','JP_Sales','EU_Sales','Other_Sales','Genre']].groupby('Genre').sum()

genre.reset_index(inplace = True)

genre.head(12)
trace1 = go.Bar(x = genre['Genre'],

                y = genre['NA_Sales'],

                name = "NORTH AMERICA",

                marker =dict(color ='rgb(0,60,48)'))

trace2 = go.Bar(x = genre['Genre'],

                y = genre['EU_Sales'],

                name = "EUROPE",

                marker =dict(color ='rgb(1,102,94)'))

trace3 = go.Bar(x = genre['Genre'],

                y = genre['JP_Sales'],

                name = "JAPAN",

                marker =dict(color ='rgb(53,151,143)'))

trace4 = go.Bar(x = genre['Genre'],

                y = genre['Other_Sales'],

                name = "OTHER",

                text=genre['Global_Sales'],

                marker =dict(color ='rgb(128,205,193)'))



data = [trace1,trace2,trace3,trace4]



layout = go.Layout(barmode = 'stack',

                   xaxis = dict(title="GENRE"),

                   yaxis = dict(title="GLOBAL SALES"),

                   title = {'text':"Global Sales of Different Genres & Regional Distribution",'x':0.5,'font':dict(family="Franklin Gothic", size=20)

})

fig = go.Figure(data = data, layout = layout)

iplot(fig)



#creating dataframe

platform =df[['Global_Sales','Platform','JP_Sales','NA_Sales','EU_Sales','Other_Sales']].groupby('Platform').sum()

platform['Count'] = df[['Global_Sales','Platform']].groupby('Platform').count()

platform.reset_index(inplace = True)

platform.head()

# a bubble-chart

figc=px.scatter(platform, x ='Global_Sales', y = 'Count',size='Global_Sales',color='Platform',

           size_max = 60)

figc.update_layout(title = {'text':"Global Sales of Platforms",'x':0.45,

                                  'font':dict(family="Franklin Gothic", size=25)})
#creating traces

trace1 = go.Bar(x = platform['Platform'],

                y = platform['NA_Sales'],

                name = "NA",

                marker =dict(color ='rgb(122,1,119)'))

trace2 = go.Bar(x = platform['Platform'],

                y = platform['EU_Sales'],

                name = "EU",

                marker =dict(color ='rgb(221,52,151)'))

trace3 = go.Bar(x = platform['Platform'],

                y = platform['JP_Sales'],

                name = "JP",

                marker =dict(color ='rgb(247,104,161)'))

trace4 = go.Bar(x = platform['Platform'],

                y = platform['Other_Sales'],

                name = "Other",

                marker =dict(color ='rgb(250,159,181)'))



#updating layout

layout = go.Layout(barmode = 'stack',

                   xaxis = dict(title="PLATFORM"),

                   yaxis = dict(title="GLOBAL SALES"),

                   title = {'text':"Platform Sales According to Regions",'x':0.5,'font':dict(family="Franklin Gothic", size=20)

})



#adding traces to the data

data =[trace1,trace2,trace3,trace4]



fig = go.Figure(layout= layout, data = data)



iplot(fig)



# creating dataframe for each genre 

action = df[df['Genre']=='Action']

sports = df[df['Genre']=='Sports']

misc = df[df['Genre']=='Misc']

roleplay = df[df['Genre']=='Role-Playing']

shooter = df[df['Genre']=='Shooter']

adventure = df[df['Genre']=='Adventure']

racing = df[df['Genre']=='Racing']

platform = df[df['Genre']=='Platform']

simulation = df[df['Genre']=='Simulation']

fighting = df[df['Genre']=='Fighting']

strategy = df[df['Genre']=='Strategy']

puzzle = df[df['Genre']=='Puzzle']

# this is to be able to plot bar-chart for each genre with different platforms

# and stack them all together
#creating traces

trace1 = go.Bar(x=action.groupby("Platform")['Global_Sales'].sum().index,

                y=action.groupby("Platform")['Global_Sales'].sum().values,

                name ="Action",

                marker = dict(color ='#ef55f1'))

trace2 = go.Bar(x=sports.groupby("Platform")['Global_Sales'].sum().index,

                y=sports.groupby("Platform")['Global_Sales'].sum().values,

                name ="Sports",

                marker = dict(color ='#fb84ce'))

trace3 = go.Bar(x=misc.groupby("Platform")['Global_Sales'].sum().index,

                y=misc.groupby("Platform")['Global_Sales'].sum().values,

                name ="Misc",

                marker = dict(color ='#fbafa1'))

trace4 = go.Bar(x=puzzle.groupby("Platform")['Global_Sales'].sum().index,

                y=puzzle.groupby("Platform")['Global_Sales'].sum().values,

                name ="Puzzle",marker = dict(color ='#f0ed35'))

trace5 = go.Bar(x=roleplay.groupby("Platform")['Global_Sales'].sum().index,

                y=roleplay.groupby("Platform")['Global_Sales'].sum().values,

                name ="Role-Playing",

                marker = dict(color = '#fcd471'))

trace6 = go.Bar(x=shooter.groupby("Platform")['Global_Sales'].sum().index,

                y=shooter.groupby("Platform")['Global_Sales'].sum().values,

                name ="Shooter",

                marker = dict(color ='#c6e516'))

trace7 = go.Bar(x=adventure.groupby("Platform")['Global_Sales'].sum().index,

                y=adventure.groupby("Platform")['Global_Sales'].sum().values,

                name ="Adventure",

                marker = dict(color = '#96d310'))

trace8 = go.Bar(x=racing.groupby("Platform")['Global_Sales'].sum().index,

                y=racing.groupby("Platform")['Global_Sales'].sum().values,

                name ="Racing",

                marker = dict(color ='#61c10b'))

trace9 = go.Bar(x=platform.groupby("Platform")['Global_Sales'].sum().index,

                y=action.groupby("Platform")['Global_Sales'].sum().values,

                name ="Platform",

                marker = dict(color ='#31ac28'))

trace10 = go.Bar(x=simulation.groupby("Platform")['Global_Sales'].sum().index,

                 y=simulation.groupby("Platform")['Global_Sales'].sum().values,

                 name ="Simulation",

                 marker = dict(color ='#439064'))

trace11 = go.Bar(x=fighting.groupby("Platform")['Global_Sales'].sum().index,

                 y=fighting.groupby("Platform")['Global_Sales'].sum().values,

                 name ="Fighting",

                 marker = dict(color = '#3d719a'))

trace12 = go.Bar(x=strategy.groupby("Platform")['Global_Sales'].sum().index,

                 y=strategy.groupby("Platform")['Global_Sales'].sum().values,

                 name ="Strategy",

                 marker = dict(color ='#284ec8'))



data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12]



#update layout

layout = go.Layout(barmode='stack',#to stack them together

                   title = {'text':"Contribution of Different Genres to Each Platform",'x':0.45,'font':dict(family="Franklin Gothic", size=20)},

                   xaxis=dict(title='PLATFORM'),

                   yaxis=dict( title='GLOBAL SALES'),

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)

publisher =df.groupby('Publisher')[['Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()

publisher.sort_values(by='Global_Sales',ascending = False, inplace = True)

publisher.reset_index(inplace=True)

publisher = publisher.iloc[:10,:]

figd =px.scatter(publisher,x ='Global_Sales',y='Publisher',size='Global_Sales',color = 'Publisher',size_max = 60)

figd.update_layout(title = {'text':"Global Sales of Top 10 Publishers",'x':0.45,

                                  'font':dict(family="Franklin Gothic", size=25)})
from plotly.subplots import make_subplots



# Initialize figure with subplots

fig = make_subplots(rows=2, cols=2,subplot_titles=("NA", "EU", "JP", "Other")) 



# Add traces

fig.add_trace(

    go.Scatter(x=publisher['Publisher'],

               y=publisher['NA_Sales'],

               name ='NorthAmerica'),

    row =1,col=1

    )

fig.add_trace(

    go.Scatter(x=publisher['Publisher'],

               y=publisher['EU_Sales'],

               name ='Europe'),

    row =1,col=2

    )

fig.add_trace(

    go.Scatter(x=publisher['Publisher'],

               y=publisher['JP_Sales'],

               name='Japan'),

    row =2,col=1

    )

fig.add_trace(

    go.Scatter(x=publisher['Publisher'],

               y=publisher['Other_Sales'],

               name='Other'),

    row =2,col=2

    )





# Update xaxis properties

fig.update_xaxes(showticklabels=False, row=1, col=1)

fig.update_xaxes(showticklabels=False, row=1, col=2)



# Update yaxis properties

fig.update_yaxes(title_text="Sales(in Mil.)", row=1, col=1)

fig.update_yaxes(title_text="Sales(in Mil.)", row=2, col=1)



# Update title and height

fig.update_layout(title=dict(text="Regional Sales Distribution of Publishers",

                             x=0.45,

                             font =dict(family= "Franklin Gothic",size=25)), 

                             height=700)

fig1=px.histogram(data_frame=df,x='Year')

fig1.update_layout(title=dict(text="Count of Games Released Each Year",

                             x=0.45,

                             font =dict(family= "Franklin Gothic",size=25)))
fig2 =px.histogram(data_frame=df,x='Year',color ='Genre',opacity=0.75)

fig2.update_layout(barmode='group', bargap=0.5,title=dict(text="Genre Count vs Year",

                             x=0.45,

                             font =dict(family= "Franklin Gothic",size=25)))
#creating a dataframe with only the top Publishers

pub_year = df.loc[df['Publisher'].isin(['Nintendo', 'Electronic Arts', 'Activision',

       'Sony Computer Entertainment', 'Ubisoft', 'Take-Two Interactive',

       'THQ', 'Konami Digital Entertainment', 'Sega',

       'Namco Bandai Games', 'Microsoft Game Studios', 'Capcom', 'Atari',

       'Square Enix', 'Warner Bros. Interactive Entertainment'])]

#creating a histogram

fig3 = px.histogram(pub_year,x='Year',color='Publisher')

fig3.update_layout(bargap =0.1,barmode='overlay',title=dict(text="Most Popular Publishers vs Years",

                                                x=0.45,

                                                font =dict(family= "Franklin Gothic",size=25)))

#to understand the selection/indexing proces better

#https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
fig4 = px.histogram(df,x='Year',color='Platform')

fig4.update_layout(bargap =0.1,barmode='overlay',title=dict(text="Platform vs Years",

                                                x=0.45,

                                                font =dict(family= "Franklin Gothic",size=25)))