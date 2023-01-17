# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



#plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import HTML, Image

from plotly import tools

import folium 

from folium import plugins 

import squarify



from geopy.geocoders import Nominatim

from geopy.extra.rate_limiter import RateLimiter



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
rail_data = pd.read_csv('../input/renfe.csv')
rail_data.head()
rail_data.tail()
rail_data.shape
rail_data.info()
for i in ['insert_date','start_date','end_date']:

    rail_data[i] = pd.to_datetime(rail_data[i])

rail_data.info()
rail_data.isnull().mean()*100
cols = ['train_class','fare']

for c in cols:

    rail_data[c].fillna(rail_data[c].mode()[0], inplace=True)
rail_data.loc[rail_data.price.isnull(), 'price'] = rail_data.groupby('fare').price.transform('mean')
#check for is nan values correctly imputed

rail_data.isnull().any()
print(f" started date minimum value {rail_data.start_date.min()}")

print(f" started date maximum value {rail_data.start_date.max()}")
print(f" end date minimum value {rail_data.end_date.min()}")

print(f" end date maximum value {rail_data.end_date.max()}")
print(f" Inserted date minimum value {rail_data.insert_date.min()}")

print(f" Inserted date maximum value {rail_data.insert_date.max()}")
# lets create some important features using date columns

rail_data['start_hour'] = rail_data['start_date'].dt.hour

rail_data['end_hour'] = rail_data['end_date'].dt.hour

rail_data['is_journey_end_on_sameday'] = np.where(rail_data['start_date'].dt.date==rail_data['end_date'].dt.date, 

                                           'yes', 'no')

rail_data['travel_time_in_mins'] = rail_data['end_date'] - rail_data['start_date']

rail_data['travel_time_in_mins']=rail_data['travel_time_in_mins']/np.timedelta64(1,'m')

rail_data['journey_day_of_week'] = rail_data['insert_date'].dt.weekday_name

rail_data['journey_month'] = rail_data['insert_date'].dt.month
geolocator = Nominatim(user_agent="specify_your_app_name_here")

geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

dictt_latitude = {}

dictt_longitude = {}

for i in rail_data['origin'].unique():

    location = geocode(i)

    print(location.address)

    print(location.latitude, location.longitude)

    dictt_latitude[i] = location.latitude

    dictt_longitude[i] = location.longitude
rail_data['start_latitude']= rail_data['origin'].map(dictt_latitude)

rail_data['start_longitude'] = rail_data['origin'].map(dictt_longitude)

rail_data['end_latitude'] = rail_data['destination'].map(dictt_latitude)

rail_data['end_longitude'] = rail_data['destination'].map(dictt_longitude)
#having a glimpse at data

rail_data.head()
count_  = rail_data['start_date'].dt.date.value_counts()

count_ = count_[:50,]

plt.figure(figsize=(20,10))

sns.barplot(count_.index, count_.values, alpha=0.8,palette = "GnBu_d")

plt.title('Plot of most journeys started dates')

plt.xticks(rotation='vertical')

plt.ylabel('Number of journeys', fontsize=12)

plt.xlabel('Date', fontsize=12)

plt.show()
count_  = rail_data['end_date'].dt.date.value_counts()

count_ = count_[:50,]

plt.figure(figsize=(20,10))

sns.barplot(count_.index, count_.values, alpha=0.8,palette = "cubehelix")

plt.title('Plot of most journeys ended dates')

plt.xticks(rotation='vertical')

plt.ylabel('Number of journeys', fontsize=12)

plt.xlabel('Date', fontsize=12)

plt.show()
cnt_ = rail_data['is_journey_end_on_sameday'].value_counts()

cnt_ = cnt_.sort_index() 

fig = {

  "data": [

    {

      "values": cnt_.values,

      "labels": cnt_.index,

      "domain": {"x": [0, .5]},

      "name": "Percentage of journeys started and ended on same date",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Percentage of journeys started and ended on same date",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

             "text": "Pie Chart",

                "x": 0.50,

                "y": 1

            },

        ]

    }

}

iplot(fig)

cnt_
import plotly.graph_objs as go

cnt_srs = rail_data['start_hour'].value_counts()

trace1 = go.Bar(

                x = cnt_srs.index,

                y = cnt_srs.values,

                marker = dict(color = 'rgba(0, 255, 200, 0.8)',

                             line=dict(color='rgb(0,0,0)',width=0.2)),

                text = cnt_srs.index)



data = [trace1]

layout = go.Layout(title = 'Plot of most journeys started according to hour')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
cnt_srs = rail_data['start_hour'].value_counts()

trace1 = go.Scatter(

                    x = cnt_srs.index,

                    y = cnt_srs.values,

                    mode = "markers",

                    marker = dict(color = 'rgba(100, 35, 55, 0.8)')

                    )



data = [trace1]

layout = dict(title = 'Journeys started according to hour',

              xaxis= dict(title= 'Journeys per hour',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
cnt_srs = rail_data['end_hour'].value_counts()

trace1 = go.Bar(

                x = cnt_srs.index,

                y = cnt_srs.values,

                marker = dict(color = 'rgba(0, 155, 100, 0.8)',

                             line=dict(color='rgb(0,0,0)',width=0.2)),

                text = cnt_srs.index)



data = [trace1]

layout = go.Layout(title = 'Plot of most journeys ended according to hour')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
cnt_srs = rail_data['end_hour'].value_counts()

trace1 = go.Scatter(

                    x = cnt_srs.index,

                    y = cnt_srs.values,

                    mode = "markers",

                    marker = dict(color = 'rgba(155, 28, 155, 0.8)')

                    )



data = [trace1]

layout = dict(title = 'Journeys ended according to hour',

              xaxis= dict(title= 'Journeys per hour',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
cnt_srs = rail_data['journey_month'].value_counts()

trace1 = go.Bar(

                x = cnt_srs.values,

                y = cnt_srs.index,orientation = 'h',

                marker = dict(color = 'rgba(155, 0, 100, 0.8)',

                             line=dict(color='rgb(0,0,0)',width=0.2)),

                text = cnt_srs.index)



data = [trace1]

layout = go.Layout(title = 'Plot of count of journeys by month')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
print('The average travelling time was {} mins \nThe maximum travelling time was {} mins \nThe minimum travelling time was {} mins'.format(rail_data.travel_time_in_mins.mean(),rail_data.travel_time_in_mins.max()

                                                                                                                                               ,rail_data.travel_time_in_mins.min()))
fig = ff.create_distplot([rail_data.travel_time_in_mins[:50000,]],['travel_time_in_mins'],bin_size=5)

iplot(fig, filename='Basic Distplot')
trace1 = go.Box(

    y=rail_data.travel_time_in_mins[:50000,],

    name = 'Box plot of average travelling time in minutes only 50k observations',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

data = [trace1]

iplot(data)
cnt_srs = rail_data['journey_day_of_week'].value_counts()

trace1 = go.Bar(

                x = cnt_srs.index,

                y = cnt_srs.values,

                marker = dict(color = 'rgba(55, 25, 55, 0.3)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = cnt_srs.index)



data = [trace1]

layout = go.Layout()

fig = go.Figure(data = data, layout = layout)

iplot(fig)
df = rail_data['origin'].value_counts()

df = pd.DataFrame(df)

df = df.reset_index()

df.columns = ['origin', 'counts'] 

df['start_latitude']= df['origin'].map(dictt_latitude)

df['start_longitude'] = df['origin'].map(dictt_longitude)

map1 = folium.Map(location=[40.4637, 3.7492], tiles='CartoDB dark_matter', zoom_start=5)

markers = []

for i, row in df.iterrows():

    loss = row['counts']

    if row['counts'] > 0:

        count = row['counts']*0.00003    

    folium.CircleMarker([float(row['start_latitude']), float(row['start_longitude'])], radius=float(count), color='#ef4f61', fill=True).add_to(map1)

map1
df = rail_data['destination'].value_counts()

df = pd.DataFrame(df)

df = df.reset_index()

df.columns = ['destination', 'counts'] 

df['start_latitude']= df['destination'].map(dictt_latitude)

df['start_longitude'] = df['destination'].map(dictt_longitude)

map1 = folium.Map(location=[40.4637, 3.7492], tiles='CartoDB dark_matter', zoom_start=5)

markers = []

for i, row in df.iterrows():

    loss = row['counts']

    if row['counts'] > 0:

        count = row['counts']*0.00003   

    folium.CircleMarker([float(row['start_latitude']), float(row['start_longitude'])], radius=float(count), color='#ef4f61', fill=True).add_to(map1)

map1
rail_data['route'] = rail_data['origin']+' to '+rail_data['destination']

print('There are {} number of routes in dataframe'.format(rail_data['route'].nunique()))
cnt_ = rail_data['route'].value_counts()



fig = {

  "data": [

    {

      "values": cnt_.values,

      "labels": cnt_.index,

      "domain": {"x": [0, .5]},

      "name": "Routes",

      "hoverinfo":"label+percent+name",

      "hole": .5,

      "type": "pie"

    },],

  "layout": {

        "title":"Pie chart of routes",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

             "text": "Pie Chart",

                "x": 0.50,

                "y": 1

            },

        ]

    }

}

iplot(fig)

cnt_
x = 0.

y = 0.

width = 50.

height = 50.

type_list = list(rail_data['route'].unique())

values = [len(rail_data[rail_data['route'] == i]) for i in type_list]



normed = squarify.normalize_sizes(values, width, height)

rects = squarify.squarify(normed, x, y, width, height)



color_brewer = ['#2D3142','#4F5D75','#BFC0C0','#F2D7EE','#EF8354','#839788','#EEE0CB','#494949']

shapes = []

annotations = []

counter = 0



for r in rects:

    shapes.append( 

        dict(

            type = 'rect', 

            x0 = r['x'], 

            y0 = r['y'], 

            x1 = r['x']+r['dx'], 

            y1 = r['y']+r['dy'],

            line = dict( width = 2 ),

            fillcolor = color_brewer[counter]

        ) 

    )

    annotations.append(

        dict(

            x = r['x']+(r['dx']/2),

            y = r['y']+(r['dy']/2),

            text = "{}-{}".format(type_list[counter], values[counter]),

            showarrow = False

        )

    )

    counter = counter + 1

    if counter >= len(color_brewer):

        counter = 0



# For hover text

trace0 = go.Scatter(

    x = [ r['x']+(r['dx']/2) for r in rects ], 

    y = [ r['y']+(r['dy']/2) for r in rects ],

    text = [ str(v) for v in values ], 

    mode = 'text',

)

        

layout = dict(

    height=1000,

    width=1250,

    xaxis=dict(showgrid=False,zeroline=False),

    yaxis=dict(showgrid=False,zeroline=False),

    shapes=shapes,

    annotations=annotations,

    hovermode='closest',

    font=dict(color="#FFFFFF")

)



# With hovertext

figure = dict(data=[trace0], layout=layout)

iplot(figure, filename='treemap')
rail_data['train_type'].value_counts()
cnt_ = rail_data['train_type'].value_counts()



fig = {

  "data": [

    {

      "values": cnt_.values,

      "labels": cnt_.index,

      "domain": {"x": [0, .5]},

      "name": "Train types",

      "hoverinfo":"label+percent+name",

      "hole": .7,

      "type": "pie"

    },],

  "layout": {

        "title":"Pie chart Train types",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

             "text": "Pie Chart",

                "x": 0.50,

                "y": 1

            },

        ]

    }

}

iplot(fig)

cnt_
cnt_ = rail_data['train_class'].value_counts()



fig = {

  "data": [

    {

      "values": cnt_.values,

      "labels": cnt_.index,

      "domain": {"x": [0, .5]},

      "name": "Train Class",

      "hoverinfo":"label+percent+name",

      "hole": .8,

      "type": "pie"

    },],

  "layout": {

        "title":"Pie chart Train Class",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

             "text": "Pie Chart",

                "x": 0.50,

                "y": 1

            },

        ]

    }

}

iplot(fig)

cnt_
cnt_ = rail_data['fare'].value_counts()

fig = {

  "data": [

    {

      "values": cnt_.values,

      "labels": cnt_.index,

      "domain": {"x": [0, .5]},

      "name": "Train Class",

      "hoverinfo":"label+percent+name",

      "hole": .9,

      "type": "pie"

    },],

  "layout": {

        "title":"Pie chart Fare",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

             "text": "Pie Chart",

                "x": 0.50,

                "y": 1

            },

        ]

    }

}

iplot(fig)

cnt_
rail_data.groupby(['route','train_type'])['train_type'].count()
plt.figure(figsize=(12,10))

sns.countplot(x= 'route', hue = 'train_type', data = rail_data,alpha=1.0,linewidth=5)

plt.title('Count plot most used train type by passengers according to routes')

plt.xticks(rotation='vertical')

plt.ylabel('Number of journeys', fontsize=12)

plt.xlabel('Route', fontsize=12)

plt.show()
rail_data.groupby(['route','train_class'])['train_class'].count()
plt.figure(figsize=(12,10))

sns.countplot(x= 'route', hue = 'train_class', data = rail_data,alpha=1.0,linewidth=5)

plt.title('Count plot most used train class by passengers according to routes')

plt.xticks(rotation='vertical')

plt.ylabel('Number of journeys', fontsize=12)

plt.xlabel('Route', fontsize=12)

plt.show()
rail_data.groupby(['route','fare'])['fare'].count()
plt.figure(figsize=(12,10))

sns.countplot(x= 'route', hue = 'fare', data = rail_data,alpha=1.0,linewidth=5)

plt.title('Count plot most used fare type by passengers according to routes')

plt.xticks(rotation='vertical')

plt.ylabel('Number of journeys', fontsize=12)

plt.xlabel('Route', fontsize=12)

plt.show()
tools.set_credentials_file(username='Ratan2513', api_key='T94PMqZ1KYsD6E8JPw1g')

def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace

#train_type

cnt_srs = rail_data.groupby('train_type')['price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace0 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(50, 71, 96, 0.6)')



#train_class

cnt_srs = rail_data.groupby('train_class')['price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(71, 58, 131, 0.8)')



#route

cnt_srs = rail_data.groupby('route')['price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace2 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(246, 78, 139, 0.6)')



#fare

cnt_srs = rail_data.groupby('fare')['price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace3 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(200, 108, 39, 0.6)')



# Creating two subplots

fig = tools.make_subplots(rows=4, cols=1, vertical_spacing=0.04, 

                          subplot_titles=['Average prices by Train Type','Average prices by Train Class','Average prices by Route','Average prices by Fare'])



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)

fig.append_trace(trace3, 4, 1)





fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Price(Euros) Plots")

py.iplot(fig, filename='Price(Euros) plots')
cnt_srs = rail_data.groupby('train_type')['price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs['train_type'] = cnt_srs.index



data = [

    {

        'x': cnt_srs['train_type'],

        'y': cnt_srs['mean'],

        'mode': 'markers+text',

        'text' : cnt_srs['train_type'],

        'textposition' : 'bottom center',

        'marker': {

            'color': "#f27da6",

            'size': 15,

            'opacity': 0.9

        }

    }

]



layout = go.Layout(title="Average fare prices according to Train type", 

                   xaxis=dict(title='Train type'),

                   yaxis=dict(title='Average price(Euros)')

                  )

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename='scatter0')
cnt_srs = rail_data.groupby('train_class')['price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs['train_class'] = cnt_srs.index



data = [

    {

        'x': cnt_srs['train_class'],

        'y': cnt_srs['mean'],

        'mode': 'markers+text',

        'text' : cnt_srs['train_class'],

        'textposition' : 'bottom center',

        'marker': {

            'color': "#d889f9",

            'size': 15,

            'opacity': 0.9

        }

    }

]



layout = go.Layout(title="Average fare prices according to Train class", 

                   xaxis=dict(title='Train class'),

                   yaxis=dict(title='Average price(Euros)')

                  )

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename='scatter1')
cnt_srs = rail_data.groupby('route')['price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs['route'] = cnt_srs.index



data = [

    {

        'x': cnt_srs['route'],

        'y': cnt_srs['mean'],

        'mode': 'markers+text',

        'text' : cnt_srs['route'],

        'textposition' : 'bottom center',

        'marker': {

            'color': "#7ae6ff",

            'size': 15,

            'opacity': 0.9

        }

    }

]



layout = go.Layout(title="Average fare prices according to routes", 

                   xaxis=dict(title='Routes'),

                   yaxis=dict(title='Average price(Euros)')

                  )

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename='scatter2')
cnt_srs = rail_data.groupby('fare')['price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs['fare'] = cnt_srs.index



data = [

    {

        'x': cnt_srs['fare'],

        'y': cnt_srs['mean'],

        'mode': 'markers+text',

        'text' : cnt_srs['fare'],

        'textposition' : 'bottom center',

        'marker': {

            'color': "#42f4bc",

            'size': 15,

            'opacity': 0.9

        }

    }

]



layout = go.Layout(title="Average fare prices according to fare type", 

                   xaxis=dict(title='Fare'),

                   yaxis=dict(title='Average price(Euros)')

                  )

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename='scatter')
cnt_srs = rail_data.groupby('route')['travel_time_in_mins'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs['route'] = cnt_srs.index



data = [

    {

        'x': cnt_srs['route'],

        'y': cnt_srs['mean'],

        'mode': 'markers+text',

        'text' : cnt_srs['route'],

        'textposition' : 'bottom center',

        'marker': {

            'color': "#d069f7",

            'size': 15,

            'opacity': 0.9

        }

    }

]



layout = go.Layout(title="Average Time taken for journeys by routes (in mins)", 

                   xaxis=dict(title='Routes'),

                   yaxis=dict(title='Average time in minutes')

                  )

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename='scatter3')
cnt_srs = rail_data.groupby('train_type')['travel_time_in_mins'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs['train_type'] = cnt_srs.index



data = [

    {

        'x': cnt_srs['train_type'],

        'y': cnt_srs['mean'],

        'mode': 'markers+text',

        'text' : cnt_srs['train_type'],

        'textposition' : 'bottom center',

        'marker': {

            'color': "#d62728",

            'size': 15,

            'opacity': 0.9

        }

    }

]



layout = go.Layout(title="Average Time taken for journeys by train type (in mins)", 

                   xaxis=dict(title='train_type'),

                   yaxis=dict(title='Average time in minutes')

                  )

fig = go.Figure(data = data, layout = layout)

iplot(fig, filename='scatter4')