#import basic libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#import the datasets

d2015 = pd.read_csv('../input/world-happiness/2015.csv')

d2016 = pd.read_csv('../input/world-happiness/2016.csv')

d2017 = pd.read_csv('../input/world-happiness/2017.csv')

d2018 = pd.read_csv('../input/world-happiness/2018.csv')

d2019 = pd.read_csv('../input/world-happiness/2019.csv')
d2015.info()
d2015.head(10)
d2015.describe()
d2015.shape
#import plotly features



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected = True)

import plotly.graph_objs as go

from plotly import tools

import plotly.figure_factory as ff

import plotly.graph_objects as go

from plotly.subplots import make_subplots
#wordcloud showing the names of countries for 2015



from wordcloud import WordCloud



plt.subplots(figsize = (8, 8))

wordcloud = WordCloud(

    background_color = 'white',

    width = 1000,

    height = 1000

).generate(" ".join(d2015.Country))

plt.imshow(wordcloud)

plt.axis('Off')

plt.show()
# Contribution of Economy, and Generosity towards global happiness VS Ranking



df2015 = d2015.iloc[:100,:]



trace1 = go.Scatter(

    x = df2015['Happiness Rank'],

    y = df2015['Economy (GDP per Capita)'],

    mode = "lines+markers",

    name = "Economy",

    marker = dict(color = 'rgb(255, 173, 222)'),

    text = df2015.Country

)

trace2 = go.Scatter(

    x = df2015['Happiness Rank'],

    y = df2015['Generosity'],

    mode = "lines+markers",

    name = 'Generosity',

    marker = dict(color = 'rgb(191, 141, 240)'),

    text = df2015.Country

)



data = [trace1, trace2]

layout = dict(title = "Contribution of Economy, and Generosity towards happiness VS Ranking",

             xaxis = dict(title='Ranking', ticklen = 25, zeroline = False)

             )

fig = dict(data=data, layout=layout)

iplot(fig)
#Contribution of Family, Health, Freedom, and Trust towards global happiness VS ranking



df2015 = d2015.iloc[:50,:]



trace1 = go.Scatter(

    x = df2015['Happiness Rank'],

    y = df2015.Family,

    mode = 'lines+markers',

    name = 'Family',

    marker = dict(color = 'rgb(255, 173, 222)'),

)



trace2 = go.Scatter(

    x = df2015['Happiness Rank'],

    y = df2015['Health (Life Expectancy)'],

    xaxis = 'x2',

    yaxis = 'y2',

    mode = 'lines+markers',

    name = 'Health',

    marker = dict(color = 'rgb(191, 141, 240)'),

)



trace3 = go.Scatter(

    x = df2015['Happiness Rank'],

    y = df2015['Freedom'],

    xaxis = 'x3',

    yaxis = 'y3',

    mode = 'lines+markers',

    name = 'Freedom',

    marker = dict(color = 'rgb(98, 179, 117)'),

)



trace4 = go.Scatter(

    x = df2015['Happiness Rank'],

    y = df2015['Trust (Government Corruption)'],

    xaxis = 'x4',

    yaxis = 'y4',

    mode = 'lines+markers',

    name = 'Trust',

    marker = dict(color = 'rgb(164, 222, 245)'),

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

    title = 'Contribution of Family, Health, Freedom, Trust towards happiness VS Ranking'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
#Countries VS their global happiness ranking for top 100 observations in 2015, 2016, 2017, 2018, and 2019



df2015 = d2015.iloc[:50,:]

df2016 = d2016.iloc[:50,:]

df2017 = d2017.iloc[:50,:]

df2018 = d2018.iloc[:50,:]

df2019 = d2019.iloc[:50,:]



trace1 = go.Scatter(

    x = df2015['Happiness Rank'],

    y = df2015.Country,

    mode = "lines",

    name = "2015",

    marker = dict(color = 'rgb(190, 207, 66)'),

    text = df2015.Country

)

trace2 = go.Scatter(

    x = df2016['Happiness Rank'],

    y = df2016.Country,

    mode = "lines",

    name = "2016",

    marker = dict(color = 'rgb(88, 237, 138)'),

    text = df2016.Country

)

trace3 = go.Scatter(

    x = df2017['Happiness.Rank'],

    y = df2017.Country,

    mode = "lines",

    name = "2017",

    marker = dict(color = 'rgb(247, 182, 250)'),

    text = df2017.Country

)

trace4 = go.Scatter(

    x = df2018['Overall rank'],

    y = df2018['Country or region'],

    mode = "lines",

    name = "2018",

    marker = dict(color = 'rgb(128, 176, 255)'),

    text = df2018['Country or region']

)

trace5 = go.Scatter(

    x = df2019['Overall rank'],

    y = df2019['Country or region'],

    mode = "lines",

    name = "2019",

    marker = dict(color = 'rgb(252, 144, 153)'),

    text = df2019['Country or region']

)

data = [trace1, trace2, trace3, trace4, trace5]

layout = dict(title = "Countries vs their happiness ranking for top 100 observations in 2015, 2016, 2017, 2018 and 2019",

             xaxis = dict(title = 'Ranking', ticklen = 10, zeroline = False),

             yaxis = dict(title = 'Name', ticklen = 10, zeroline = False)

             )

fig = dict(data=data, layout=layout)

iplot(fig)
#Contribution of Family, Health, Freedom, Trust and Generosity towards global happiness using bar plots



df2015 = d2015.iloc[:3,:]



trace1 = go.Bar(

    x = df2015.Country,

    y = df2015['Family'],

    name = 'Family',

    marker = dict(color = 'rgb(252, 151, 226)'),

    text = df2015.Country

)

trace2 = go.Bar(

    x = df2015.Country,

    y = df2015['Health (Life Expectancy)'],

    name = 'Health',

    marker = dict(color = 'rgb(126, 205, 217)'),

    text = df2015.Country

)

trace3 = go.Bar(

    x = df2015.Country,

    y = df2015['Freedom'],

    name = 'Freedom',

    marker = dict(color = 'rgb(188, 217, 126)'),                

    text = df2015.Country

)

trace4 = go.Bar(

    x = df2015.Country,

    y = df2015['Trust (Government Corruption)'],

    name = 'Trust',

    marker = dict(color = 'rgb(151, 169, 240)'),                

    text = df2015.Country

)

trace5 = go.Bar(

    x = df2015.Country,

    y = df2015['Generosity'],

    name = 'Generosity',

    marker = dict(color = 'rgb(227, 164, 245)'),                

    text = df2015.Country

)

data = [trace1, trace2, trace3, trace4, trace5]

layout = go.Layout(barmode = "group", title='Contribution of various factors towards global happiness')

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#using unique barplots to reflect contribution of Family, Health, Freedom, Trust, and Generosity towards global happiness



df2015 = d2015.iloc[:5, :]

x = df2015.Country



trace1 = {

    'x':x,

    'y':df2015['Family'],

    'name':'Family',

    'marker':dict(color = 'rgb(252, 151, 226)'),

    'type':'bar'

}



trace2 = {

    'x':x,

    'y':df2015['Health (Life Expectancy)'],

    'name':'Health',

    'marker':dict(color = 'rgb(126, 205, 217)'),

    'type':'bar'

}



trace3 = {

    'x':x,

    'y':df2015['Freedom'],

    'name':'Freedom',

    'marker':dict(color = 'rgb(188, 217, 126)'),

    'type':'bar'

}



trace4 = {

    'x':x,

    'y':df2015['Trust (Government Corruption)'],

    'name':'Trust',

    'marker':dict(color = 'rgb(227, 164, 245)'),

    'type':'bar'

}



trace5 = {

    'x':x,

    'y':df2015['Generosity'],

    'name':'Generosity',

    'marker':dict(color = 'rgb(151, 169, 240)'),

    'type':'bar'

}



data = [trace1, trace2, trace3, trace5, trace4];

layout = {

    'xaxis':{'title':'Top 5 countries for world happiness ranking'},

    'barmode':'relative',

    'title':'Contribution of Family, Health, Freedom, Trust, and Generosity towards global happiness'

};

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#using pieplots to reflect contribution of Health, Generosity, Freedom, and Trust towards global happiness



df2015 = d2015.iloc[:8,:]



myColor = ['rgb(235, 140, 129)','rgb(235, 214, 138)','rgb(193, 207, 128)','rgb(153, 247, 161)','rgb(138, 235, 237)','rgb(140, 172, 237)','rgb(220, 184, 252)','rgb(148, 189, 134)']

specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs, subplot_titles = ['Health','Generosity','Freedom','Trust'])



# Define pie charts

fig.add_trace(go.Pie(labels=df2015.Country, values=df2015['Health (Life Expectancy)'],pull=[0.2,0,0,0,0,0,0,0],

                     marker_colors=myColor), 1, 1)

fig.add_trace(go.Pie(labels=df2015.Country, values=df2015['Generosity'],pull=[0.2,0,0,0,0,0,0,0],

                     marker_colors=myColor), 1, 2)

fig.add_trace(go.Pie(labels=df2015.Country, values=df2015['Freedom'],pull=[0.2,0,0,0,0,0,0,0],

                     marker_colors=myColor), 2, 1)

fig.add_trace(go.Pie(labels=df2015.Country, values=df2015['Trust (Government Corruption)'],pull=[0.2,0,0,0,0,0,0,0],

                     marker_colors=myColor), 2, 2)



# Tune layout and hover info

fig.update_traces(hoverinfo='label+percent+name', textinfo='none')

fig.update(layout_title_text='Health, Generosity,Freedom,Trust contribution towards global happiness')

fig.update_layout()



fig = go.Figure(fig)

fig.show()
#barplots reflecting regions in 2015, and 2016



df2015 = d2015.Region

df2016 = d2016.Region



trace1 = go.Histogram(

    x=df2015,

    opacity=0.50,

    name = "2015",

    marker=dict(color='rgb(115, 195, 222)'))

trace2 = go.Histogram(

    x=df2016,

    xaxis = 'x2',

    yaxis = 'y2',

    opacity=0.50,

    name = "2016",

    marker=dict(color='rgb(217, 153, 242)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' Count vs Regions in 2015, and 2016',

                   xaxis2=dict(title='Region',domain=[0.6, 0.95],anchor='y2'),

                   yaxis2=dict( title='Count',domain=[0.6, 0.95],anchor='y2'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
#boxplots representation of Family, Health, Freedom, Trust, and Generosity contribution towards global happiness



fig = make_subplots(

    rows=3, cols=2,

    specs=[[{}, {}],[{}, {}],

           [{"colspan": 2}, None]],

    subplot_titles=("Family","Health", "Freedom","Trust","Generosity"))



fig.add_trace(go.Box(

    x = d2015.Family,

    name = 'Family',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=1, col=1)

fig.add_trace(go.Box(

    x = d2015['Health (Life Expectancy)'],

    name = 'Health (Life Expectancy)',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=1, col=2)

fig.add_trace(go.Box(

    x = d2015.Freedom,

    name = 'Freedom',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=2, col=1)

fig.add_trace(go.Box(

    x = d2015['Trust (Government Corruption)'],

    name = 'Trust (Government Corruption)',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=2, col=2)

fig.add_trace(go.Box(

    x = d2015['Generosity'],

    name = 'Generosity',

    marker = dict(

        color = 'rgb(145, 209, 111)',

    )

),row=3, col=1)

fig.update_layout(height=1000, width=2000,showlegend=False, title_text="Boxplots")

fig.show()
#scatterplot matrix showing correlation between Family, Health, Freedom, Trust and Generosity contribution towards global happiness



df2015 = d2015.loc[:,["Family","Health (Life Expectancy)","Freedom","Trust (Government Corruption)","Generosity"]]

df2015["index"] = np.arange(1, len(df2015) + 1)



fig = ff.create_scatterplotmatrix(df2015, diag='box', index='index',

                                 colormap='Portland', colormap_type='cat',

                                 height=1500, width=1500)

iplot(fig)
#3-d plot with global happiness ranking, Region and Family along the three axes



trace1 = go.Scatter3d(

    x = d2015['Happiness Rank'],

    y = d2015.Region,

    z = d2015.Family,

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