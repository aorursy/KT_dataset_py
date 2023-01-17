#import basic libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#import the dataset

videoGames = pd.read_csv('../input/videogamesales/vgsales.csv')

videoGames.info()

videoGames.head(10)
videoGames.describe()
videoGames.shape
#import plotly features



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected = True)

import plotly.graph_objs as go

from plotly import tools

import plotly.figure_factory as ff

import plotly.graph_objects as go

from plotly.subplots import make_subplots
#scatterplot showing Other Sales and Global Sales vs Ranking for top 100 video games



df = videoGames.iloc[:100,:]



trace1 = go.Scatter(

    x = df.Rank,

    y = df.Other_Sales,

    mode = "lines",

    name = "Other Sales",

    marker = dict(color = 'rgb(255, 173, 222)'),

    text = df.Name

)

trace2 = go.Scatter(

    x = df.Rank,

    y = df.Global_Sales,

    mode = "lines",

    name = 'Global Sales',

    marker = dict(color = 'rgb(191, 141, 240)'),

    text = df.Name

)



data = [trace1, trace2]

layout = dict(title = "Other Sales, and Global Sales vs Ranking of top 100 video games",

             xaxis = dict(title='Ranking', ticklen = 25, zeroline = False)

             )

fig = dict(data=data, layout=layout)

iplot(fig)
#scatterplots with four subplots showing sales in North America, Europe, Japan, and Global Sales vs Ranking 

#for video games in 2014



df2014 = videoGames[videoGames.Year == 2014]



trace1 = go.Scatter(

    x = df2014.Rank,

    y = df2014.NA_Sales,

    name = 'Sales in North America',

    marker = dict(color = 'rgb(98, 179, 117)')

)



trace2 = go.Scatter(

    x = df2014.Rank,

    y = df2014.EU_Sales,

    xaxis = 'x2',

    yaxis = 'y2',

    name = 'Sales in Europe',

    marker = dict(color = 'rgb(164, 222, 245)')

)



trace3 = go.Scatter(

    x = df2014.Rank,

    y = df2014.JP_Sales,

    xaxis = 'x3',

    yaxis = 'y3',

    name = 'Sales in Japan',

    marker = dict(color = 'rgb(212, 172, 250)')

)



trace4 = go.Scatter(

    x = df2014.Rank,

    y = df2014.Global_Sales,

    xaxis = 'x4',

    yaxis = 'y4',

    name = 'Global Sales',

    marker = dict(color = 'rgb(255, 173, 222)')

)



data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.48]

    ),

    yaxis=dict(

        domain=[0, 0.48]

    ),

    xaxis2=dict(

        domain=[0.52, 1]

    ),

    xaxis3=dict(

        domain=[0, 0.48],

        anchor='y3'

    ),

    xaxis4=dict(

        domain=[0.52, 1],

        anchor='y4'

    ),

    yaxis2=dict(

        domain=[0, 0.48],

        anchor='x2'

    ),

    yaxis3=dict(

        domain=[0.52, 1]

    ),

    yaxis4=dict(

        domain=[0.52, 1],

        anchor='x4'

    ),

    title = 'Sales in North America, Europe, Japan,and Global Sales vs Rank of video games in 2014'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)

#scatterplots showing the name of video games vs their rank in 2013, 2014,and 2015



df2013 = videoGames[videoGames.Year == 2013]

df2014 = videoGames[videoGames.Year == 2014]

df2015 = videoGames[videoGames.Year == 2015]



trace1 = go.Scatter(

    x = df2013.Rank,

    y = df2013.Name,

    mode = "markers",

    name = "2013",

    marker = dict(color = 'rgb(255, 224, 150)'),

    text = df2013.Name

)

trace2 = go.Scatter(

    x = df2014.Rank,

    y = df2014.Name,

    mode = "markers",

    name = "2014",

    marker = dict(color = 'rgb(161, 255, 206)'),

    text = df2014.Name

)

trace3 = go.Scatter(

    x = df2015.Rank,

    y = df2015.Name,

    mode = "markers",

    name = "2015",

    marker = dict(color = 'rgb(247, 182, 250)'),

    text = df2014.Name

)

data = [trace1, trace2, trace3]

layout = dict(title = "Name vs Ranking of top 100 video games in 2013, 2014, and 2015",

             xaxis = dict(title = 'Ranking', ticklen = 10, zeroline = False),

             yaxis = dict(title = 'Name', ticklen = 10, zeroline = False)

             )

fig = dict(data=data, layout=layout)

iplot(fig)
#scatterplots showing the global sales of top 100 video games vs their rank in 2013, 2014, 2015



df2013 = videoGames[videoGames.Year == 2013]

df2014 = videoGames[videoGames.Year == 2014]

df2015 = videoGames[videoGames.Year == 2015]



trace1 = go.Scatter(

    x = df2013.Rank,

    y = df2013.Publisher,

    mode = "markers",

    name = "2013",

    marker = dict(color = 'rgb(255, 224, 150)'),

    text = df2013.Name

)

trace2 = go.Scatter(

    x = df2014.Rank,

    y = df2014.Publisher,

    mode = "markers",

    name = "2014",

    marker = dict(color = 'rgb(113, 201, 134)'),

    text = df2014.Name

)

trace3 = go.Scatter(

    x = df2015.Rank,

    y = df2015.Publisher,

    mode = "markers",

    name = "2015",

    marker = dict(color = 'rgb(247, 182, 250)'),

    text = df2014.Name

)

data = [trace1, trace2, trace3]

layout = dict(title = "Global Sales vs Rank of top 100 video games in 2013, 2014, and 2015",

             xaxis = dict(title = 'Ranking', ticklen = 10, zeroline = False),

             yaxis = dict(title = 'Global Sales', ticklen = 10, zeroline = False)

             )

fig = dict(data=data, layout=layout)

iplot(fig)
#Bar plots showing the sales in North America, Europe, Japan, Other Sales, and Global Salesn of

#top 3 video games



df2014 = videoGames[videoGames.Year == 2014].iloc[:3,:]



trace1 = go.Bar(

    x = df2014.Name,

    y = df2014.NA_Sales,

    name = 'Sales in North America',

    marker = dict(color = 'rgb(252, 151, 226)'),

    text = df2014.Genre

)

trace2 = go.Bar(

    x = df2014.Name,

    y = df2014.EU_Sales,

    name = 'Sales in Europe',

    marker = dict(color = 'rgb(126, 205, 217)'),

    text = df2014.Genre

)

trace3 = go.Bar(

    x = df2014.Name,

    y = df2014.Global_Sales,

    name = 'Global Sales',

    marker = dict(color = 'rgb(188, 217, 126)'),                

    text = df2014.Genre

)

trace4 = go.Bar(

    x = df2014.Name,

    y = df2014.Other_Sales,

    name = 'Other Sales',

    marker = dict(color = 'rgb(151, 169, 240)'),                

    text = df2014.Genre

)

trace5 = go.Bar(

    x = df2014.Name,

    y = df2014.JP_Sales,

    name = 'Sales in Japan',

    marker = dict(color = 'rgb(227, 164, 245)'),                

    text = df2014.Genre

)

data = [trace1, trace2, trace3, trace4, trace5]

layout = go.Layout(barmode = "group")

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#unique bar plots showing sales in North America, Europe, Japan, Other Sales, and Global Sales 

#for top 3 video games



df2014 = videoGames[videoGames.Year == 2014].iloc[:5,:]

x = df2014.Name



trace1 = {

  'x': x,

  'y': df2014.NA_Sales,

  'name': 'Sales in North America',

  'marker' : dict(color = 'rgb(252, 151, 226)'),

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': df2014.EU_Sales,

  'name': 'Sales in Europe',

  'marker' : dict(color = 'rgb(126, 205, 217)'),

  'type': 'bar'

};

trace3 = {

  'x': x,

  'y': df2014.JP_Sales,

  'name': 'Sales in Japan',

  'marker' : dict(color = 'rgb(188, 217, 126)'),

  'type': 'bar'

};

trace5 = {

  'x': x,

  'y': df2014.Global_Sales,

  'name': 'Global Sales',

  'marker' : dict(color = 'rgb(227, 164, 245)'),

  'type': 'bar'

};

trace4 = {

  'x': x,

  'y': df2014.Other_Sales,

  'name': 'Other Sales',

  'marker' : dict(color = 'rgb(151, 169, 240)'),

  'type': 'bar'

};



data = [trace1, trace2, trace3, trace5, trace4];

layout = {

  'xaxis': {'title': 'Top 3 video games'},

  'barmode': 'relative',

  'title': 'Sales in North America, Europe, Japan, Other Sales, Global Sales for first  video games in 2014'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
#bar plots showing Other Sales, and scatterplots showing Global Sales for top 7 video games



df2014 = videoGames[videoGames.Year == 2014].iloc[:7,:]



y_otherSales = [each for each in df2014.Other_Sales]

y_globalSales = [float(each) for each in df2014.Global_Sales]

x_otherSales = [each for each in df2014.Name]

x_globalSales = [each for each in df2014.Name]

trace0 = go.Bar(

                x=y_otherSales,

                y=x_otherSales,

                marker=dict(color='rgb(247, 182, 250)',line=dict(color='rgb(255, 255, 255)',width=1.2)),

                name='Other Sales',

                orientation='h',   

)

trace1 = go.Scatter(

                x=y_globalSales,

                y=x_globalSales,

                mode='lines+markers',

                line=dict(color='rgb(138, 103, 245)'),

                name='Global Sales',

)

layout = dict(

                title='Other Sales and Global Sales of top 7 video games',

                yaxis=dict(showticklabels=True,domain=[0, 0.85]),

                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),

                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),

                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),

                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),

                margin=dict(l=200, r=20,t=70,b=70),

                paper_bgcolor='rgb(255, 255, 255)',

                plot_bgcolor='rgb(255, 255, 255)',

)

annotations = []

y_s = np.round(y_otherSales, decimals=2)

y_nw = np.rint(y_globalSales)

# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, x_otherSales):

    # labeling the scatter savings

    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))

    # labeling the bar net worth

    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))



layout['annotations'] = annotations



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                          shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



fig['layout'].update(layout)

iplot(fig)
#pieplots showing sales in North America, Europe, Japan, and Global Sales

#for top 8 video games in 2014



df2014 = videoGames[videoGames.Year == 2014].iloc[:8,:]



myColor = ['rgb(235, 140, 129)','rgb(235, 214, 138)','rgb(193, 207, 128)','rgb(153, 247, 161)','rgb(138, 235, 237)','rgb(140, 172, 237)','rgb(220, 184, 252)','rgb(148, 189, 134)']

specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs, subplot_titles = ['Sales in NA','Sales in Europe','Sales in Japan','Global Sales'])



# Define pie charts

fig.add_trace(go.Pie(labels=df2014.Name, values=df2014.NA_Sales,pull=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],

                     marker_colors=myColor), 1, 1)

fig.add_trace(go.Pie(labels=df2014.Name, values=df2014.EU_Sales,pull=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],

                     marker_colors=myColor), 1, 2)

fig.add_trace(go.Pie(labels=df2014.Name, values=df2014.JP_Sales,pull=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],

                     marker_colors=myColor), 2, 1)

fig.add_trace(go.Pie(labels=df2014.Name, values=df2014.Global_Sales,pull=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],

                     marker_colors=myColor), 2, 2)



# Tune layout and hover info

fig.update_traces(hoverinfo='label+percent+name', textinfo='none')

fig.update(layout_title_text='Game Sales')

fig.update_layout()



fig = go.Figure(fig)

fig.show()
#pieplots showing Other Sales and Global Sales for top 8 video games 

#in 2014



labels = df2014.Name



fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],

                    subplot_titles=['Other Sales', 'Global Sales'])

fig.add_trace(go.Pie(labels=labels, values=df2014.Other_Sales, scalegroup='one',

                     name="Other Sales",pull=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], marker_colors = ['rgb(235, 140, 129)','rgb(235, 214, 138)','rgb(193, 207, 128)','rgb(153, 247, 161)','rgb(138, 235, 237)','rgb(140, 172, 237)','rgb(220, 184, 252)','rgb(148, 189, 134)']), 1, 1)

fig.add_trace(go.Pie(labels=labels, values=df2014.Global_Sales, scalegroup='one',

                     name="Global Sales",pull=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]), 1, 2)



fig.update_layout(title_text='Game Sales')

fig.show()
#barplots showing count of genres in 2014, 2015



df2015 = videoGames.Genre[videoGames.Year == 2015]

df2014 = videoGames.Genre[videoGames.Year == 2014]



trace1 = go.Histogram(

    x=df2014,

    opacity=0.50,

    name = "2014",

    marker=dict(color='rgb(115, 195, 222)'))

trace2 = go.Histogram(

    x=df2015,

    xaxis = 'x2',

    yaxis = 'y2',

    opacity=0.50,

    name = "2015",

    marker=dict(color='rgb(217, 153, 242)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' Count of genres in 2014, and 2015',

                   xaxis2=dict(title='Genre',domain=[0.6, 0.95],anchor='y2'),

                   yaxis2=dict( title='Count',domain=[0.6, 0.95],anchor='y2'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#bar plots showing count of publishers in 2014, 2015



df2015 = videoGames.Publisher[videoGames.Year == 2015]

df2014 = videoGames.Publisher[videoGames.Year == 2014]



trace1 = go.Histogram(

    x=df2014,

    opacity=0.50,

    name = "2014",

    marker=dict(color='rgb(115, 195, 222)'))

trace2 = go.Histogram(

    x=df2015,

    xaxis = 'x2',

    yaxis = 'y2',

    opacity=0.50,

    name = "2015",

    marker=dict(color='rgb(217, 153, 242)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' Count of publishers in 2014, and 2015',

                   xaxis2=dict(title='Publisher',domain=[0.6, 0.95],anchor='y2'),

                   yaxis2=dict( title='Count',domain=[0.6, 0.95],anchor='y2'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#bar plots shwoing count of platforms in 2014, 2015



df2015 = videoGames.Platform[videoGames.Year == 2015]

df2014 = videoGames.Platform[videoGames.Year == 2014]



trace1 = go.Histogram(

    x=df2014,

    opacity=0.50,

    name = "2014",

    marker=dict(color='rgb(115, 195, 222)'))

trace2 = go.Histogram(

    x=df2015,

    xaxis = 'x2',

    yaxis = 'y2',

    opacity=0.50,

    name = "2015",

    marker=dict(color='rgb(217, 153, 242)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' Count of publishers in 2014, and 2015',

                   xaxis2=dict(title='Platform',domain=[0.6, 0.95],anchor='y2'),

                   yaxis2=dict( title='Count',domain=[0.6, 0.95],anchor='y2'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#wordcloud showing the names of video games



from wordcloud import WordCloud



df2014 = videoGames.Name[videoGames.Year == 2014]

plt.subplots(figsize = (8,8))

wordcloud = WordCloud(

    background_color = 'white',

    width = 1000,

    height = 1000

).generate(" ".join(df2014))

plt.imshow(wordcloud)

plt.axis('Off')

plt.show()
#boxplots showing sales in North America, Europe, Japan, Other Sales and Global Sales in 2014



df2014 = videoGames[videoGames.Year == 2014]

fig = make_subplots(

    rows=3, cols=2,

    specs=[[{}, {}],[{}, {}],

           [{"colspan": 2}, None]],

    subplot_titles=("Sales in NA in 2014","Sales in Europe in 2014", "Sales in Japan in 2014","Other Sales","Global Sales"))



fig.add_trace(go.Box(

    x = df2014.NA_Sales,

    name = 'Sales in NA in 2014',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=1, col=1)

fig.add_trace(go.Box(

    x = df2014.EU_Sales,

    name = 'Sales in Europe in 2014',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=1, col=2)

fig.add_trace(go.Box(

    x = df2014.JP_Sales,

    name = 'Sales in Japan in 2014',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=2, col=1)

fig.add_trace(go.Box(

    x = df2014.Other_Sales,

    name = 'Other Sales',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=2, col=2)

fig.add_trace(go.Box(

    x = df2014.Global_Sales,

    name = 'Global Sales',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=3, col=1)

fig.update_layout(height=1000, width=2000,showlegend=False, title_text="Game Sales")

fig.show()
#scatterplot matrix to express relation between sales in North America, Europe, Japan, Other Sales,and Global Sales

#using scatterplots and boxplots



df2014 = videoGames[videoGames.Year == 2014].loc[:,["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]]

df2014["index"] = np.arange(1, len(df2014) + 1)



fig = ff.create_scatterplotmatrix(df2014, diag='box', index='index',

                                 colormap='Portland', colormap_type='cat',

                                 height=1500, width=1500)

iplot(fig)
#3-d plot with Rank, Publisher, and Global Sales representing the 3 axes



df2014 = videoGames[videoGames.Year == 2014]



trace1 = go.Scatter3d(

    x = df2014.Rank,

    y = df2014.Publisher,

    z = df2014.Global_Sales,

    mode = 'markers',

    marker = dict(

        size = 10,

        color = 'rgb(141, 175, 227)',

    )

)

data = [trace1]

layout = go.Layout(

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)