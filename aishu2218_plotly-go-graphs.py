import pandas as pd

import numpy as np

%matplotlib inline
import plotly.offline as pyo

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.figure_factory as ff
df = pd.read_csv("../input/schoolearnings/schoolearnings.csv")



table = ff.create_table(df)

pyo.iplot(table)
num = 100

x = np.random.randn(num)

y = np.random.randn(num)



# Create a trace



trace = go.Scatter(

    x = x,

    y = y,

    mode = 'markers'

)



data = [trace]

 

fig = go.Figure(data=data)

fig.show()
sales = pd.read_excel('../input/superstore-sales/Sample - Superstore Sales.xls',0)

sales.head()
trace0= go.Scatter(

        x = sales['Sales'],

        y = sales['Profit'],

        mode ='markers',

        marker= dict(

                size=15,

                color= 'rgb(150,120,180)',

                symbol= 'square'

        ))



data=[trace0]

lay = go.Layout(title='Sales vs Profit')



fig= go.Figure(data=data,layout= lay)

fig.show()
fifa = pd.read_csv('../input/fifa-world-cup/WorldCupMatches.csv')
fifa.head()
fifa.columns
fifa_avg = pd.pivot_table(fifa,values='Home Team Goals',index='Year')

fifa_avg.head()
trace0= go.Scatter(

        x = fifa_avg.index,

        y = fifa_avg['Home Team Goals'],

        mode ='lines',

        name = 'Home Team Goals'

)



data=[trace0]

layout = go.Layout(title='Home Team Goals')



fig = go.Figure(data=data,layout=layout)

fig.show()
fifa_avg = pd.pivot_table(fifa,values=['Home Team Goals','Away Team Goals'],index='Year')

fifa_avg.head()
trace0= go.Scatter(

        x = fifa_avg.index,

        y = fifa_avg['Away Team Goals'],

        mode ='lines+markers',

        name = 'Away Team Goals'

)



trace1 = go.Scatter(

        x = fifa_avg.index,

        y = fifa_avg['Home Team Goals'],

        mode ='lines+markers',

        name = 'Home Team Goals'

)



data=[trace0,trace1]

layout = go.Layout(title='Average Away and Home Team Goals Per Year')



fig = go.Figure(data=data,layout=layout)

fig.show()
order_priority = pd.pivot_table(sales,index= 'Order Priority',values='Sales')

order_priority.head()
trace0 = go.Bar(

         x = order_priority.index,

         y = order_priority['Sales'],

         marker =dict(color='rgb(130,220,300)'),

         name = "Order Priority"

)



data=[trace0]

layout= go.Layout(title='Order Priority')



fig = go.Figure(data=data,layout=layout)

fig.show()
trace0= go.Bar(

        x = fifa_avg.index,

        y = fifa_avg['Away Team Goals'],

        marker =dict(color='rgb(10,20,30)')

)



trace1 = go.Bar(

        x = fifa_avg.index,

        y = fifa_avg['Home Team Goals']

)



data=[trace0,trace1]

layout = go.Layout(title='Average Away and Home Team Goals Per Year')



fig = go.Figure(data=data,layout=layout)

fig.show()
trace0= go.Bar(

        x = fifa_avg.index,

        y = fifa_avg['Away Team Goals'],

)



trace1 = go.Bar(

        x = fifa_avg.index,

        y = fifa_avg['Home Team Goals'],

)



data=[trace0,trace1]

layout = go.Layout(title='Average Away and Home Team Goals Per Year',barmode='stack')



fig = go.Figure(data=data,layout=layout)

fig.show()
trace0= go.Scatter(

        x = fifa_avg.index,

        y = fifa_avg['Away Team Goals'],

        mode = 'lines',

        marker = dict(color= 'rgb(20,30,40)'),

        name='Away Team Goals'

)



trace1 = go.Bar(

        x = fifa_avg.index,

        y = fifa_avg['Home Team Goals'],

        marker = dict(color= 'rgb(10,150,120)'),

        name= 'Home Team Goals'

        

)



data=[trace0,trace1]

layout = go.Layout(title='Average Away and Home Team Goals Per Year')



fig = go.Figure(data=data,layout=layout)

fig.show()
trace0= go.Scatter(

        x = sales['Sales'],

        y = sales['Profit'],

        text = sales['Customer Name'],

        mode = 'markers',

        marker = dict(size=100 * sales['Discount'],color='rgb(220,110,120)')

)



data= [trace0]



layout = go.Layout(

        title ='Sales vs Profit',

        xaxis = dict(title='Sales'),

        yaxis = dict(title='Profit'),

        hovermode = 'closest'

        )



fig = go.Figure(data=data,layout=layout)

fig.show()
sales['Ship Mode'].value_counts()
trace = go.Pie(labels = ['Regular Air','Delivery Truck','Express Air'], values = sales['Ship Mode'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['pink', 'purple'], 

               line=dict(color='#000000', width=1.5)))

           



layout= go.Layout(

        title={

        'text': "Ship Modes",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig = go.Figure(data = [trace], layout=layout)

fig.show()
trace0 = go.Histogram(

         x = sales['Sales'],

         name= 'Sales',

         opacity =0.5

)



trace1 = go.Histogram(

         x = sales['Profit'],

         name= 'Profit',

         opacity =0.5

)



data= [trace0,trace1]

layout = go.Layout(title='Sales vs Profit Distribution')



fig = go.Figure(data=data,layout=layout)

fig.show()
sales['Customer Segment'].value_counts()
corp = sales['Sales'][sales['Customer Segment']=='Corporate']

home_office = sales['Sales'][sales['Customer Segment']=='Home Office']

cons = sales['Sales'][sales['Customer Segment']=='Consumer']

small_business = sales['Sales'][sales['Customer Segment']=='Small Business']
trace0 = go.Box(

         y = corp,

         name = 'Corporate'

)



trace1 = go.Box(

         y = home_office,

         name = 'Home Office'

)



trace2 = go.Box(

         y = cons,

         name = 'Consumer'

)



trace3 = go.Box(

         y = small_business,

         name = 'Small Business'

)
data =[trace0,trace1,trace2,trace3]

layout = go.Layout(title='Distribution of Customer Segment')



fig = go.Figure(data=data,layout=layout)

fig.show()
x = np.random.randn(500)

data = [x]

label = ['DISTPLOT'] 

 

fig = ff.create_distplot(data, label)

fig.show()