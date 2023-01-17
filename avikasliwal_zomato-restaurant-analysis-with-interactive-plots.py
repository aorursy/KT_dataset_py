import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from PIL import Image

from wordcloud import WordCloud, STOPWORDS



import plotly as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.offline as pyo

pyo.init_notebook_mode()

from plotly import tools



py.offline.init_notebook_mode(connected=True)



print('Done!')
# Importing the dataset

zomato = pd.read_csv('../input/zomato.csv', encoding = "ISO-8859-1")



# Viewing 3 random sample from the dataframe

zomato.sample(3)
# Size of the data

print('Shape of the df : ', zomato.shape)
# dropping duplicates

zomato.drop_duplicates(subset = 'Restaurant ID', keep = 'first', inplace = True)



# Size of the data

zomato.shape
zomato.replace({

    'Country Code' : {

        1 : 'India',

        14 : 'Australia',

        30 : 'Brazil',

        37 : 'Canada',

        94 : 'Indonesia',

        148 : 'New Zealand',

        162 : 'Phillipines',

        166 : 'Qatar',

        184 : 'Singapore',

        189 : 'South Africa',

        191 : 'Sri Lanka',

        208 : 'Turkey',

        214 : 'UAE',

        215 : 'United Kingdom',

        216 : 'United States',

    }

}, inplace = True)



# renaming col country code to country 

zomato.rename(columns = {'Country Code' : 'Country'}, inplace = True)
# creating a df 'za' to save country and no of restaurants in that country



# country name

x = zomato.Country.value_counts().index.values



# no. of restaurants in a particular country

y = zomato.Country.value_counts().values



# creating a df

za = pd.DataFrame({

    'Country': x,

    '# of Restaurants': y,

})



za.head(3)
choropleth_map = dict (

    type = 'choropleth',

    locations = za['Country'],

    locationmode='country names',

    colorscale = 'Rainbow',

    z = za['# of Restaurants'],

    showscale = False,)



layout = go.Layout (

    title = go.layout.Title(

        text = "Zomato's presence on the planet"

    ),

)



fig = go.Figure(data = [choropleth_map], layout = layout)

py.offline.iplot(fig)
# Of restaurants in India

nor_India = za[za['Country'] == 'India']['# of Restaurants'][0]



# of Indian restaurants

percent = nor_India / (za['# of Restaurants'].sum())



print('Percentage of indian restaurants is: ', percent * 100)
# Restaurants from India only

zomato_India = zomato[zomato.Country == 'India']



# Viewing 3 random samples

zomato_India.sample(3)
# Lets remove some noise from zomato_India dataset

zomato_India.drop(columns = ['Restaurant ID', 'Country', 'Currency', 'Rating color'], inplace = True)



# shape of dataset

print('Shape :', zomato_India.shape)



# 3 random samples

zomato_India.sample(3)
mapbox_access_token = 'pk.eyJ1IjoiYXZpa2FzbGl3YWwiLCJhIjoiY2p4MDhzYjI0MTg1bjQwcG05cjZqNjRtaiJ9.9MKul4M02Wp2TV3Fx-TwoQ'
data = [

    go.Scattermapbox(

        lat = zomato_India['Latitude'],

        lon = zomato_India['Longitude'],

        mode = 'markers',

        marker = go.scattermapbox.Marker(

            size = 5,

            color = '#cb202d'

        ),

        text = zomato_India['Restaurant Name'],

    )

]



layout = go.Layout(

    autosize = True,

    hovermode = 'closest',

    mapbox = go.layout.Mapbox(

        accesstoken = mapbox_access_token,

        bearing = 0,

        center = go.layout.mapbox.Center(

            lat = 26.52,

            lon = 78.37

        ),

        pitch = 1,

        zoom = 4,

        style = 'light'

    ),

    title = go.layout.Title (

        text = 'Zomato restaurants in India'

    ),

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig, filename='Multiple Mapbox')
cities10_name = zomato_India.City.value_counts()[:9].index

cities10_value_log = np.log(zomato_India.City.value_counts()[:9])



data = [go.Bar(

            x = cities10_name,

            y = cities10_value_log,

            text = cities10_name,

            marker=dict(

                color=['red', 'orange',

               'green', 'blue','violet',

               'rgba(204,204,204,1)','rgba(204,204,204,1)',

               'rgba(204,204,204,1)','rgba(204,204,204,1)',

               'rgba(204,204,204,1)']),

        )]



layout = go.Layout(

    title = 'Top 10 citites for zomato',

    xaxis = dict(

        tickfont=dict(

            size=14,

            color='rgb(107, 107, 107)'

        )

    ),

    yaxis = dict(

        title='Log10 of Number of Restaurant',

        tickfont=dict(

            size=14,

            color='rgb(107, 107, 107)'

        ),

        titlefont=dict(

            size=16,

            color='rgb(107, 107, 107)'

        ),

    )

)



fig = go.Figure(data = data, layout = layout)



pyo.iplot(fig, filename='color-bar')
# ncr

zomato_ncr = zomato_India.loc[(zomato_India.City).isin(['New Delhi','Gurgaon','Noida','Faridabad'])]



# rest of india

zomato_india = zomato_India.loc[~(zomato_India.City).isin(['New Delhi','Gurgaon','Noida','Faridabad'])]
# Non NCR

zomato_india['Average Cost for two'].describe()
zero = zomato_india[zomato_india['Average Cost for two'] == 0]

len(zero)
# replacing 0 by mean price

zomato_india['Average Cost for two'].replace({

    0: zomato_India['Average Cost for two'].mean()

}, inplace = True)



zomato_india['Average Cost for two'].describe()
# NCR

zomato_ncr['Average Cost for two'].describe()
trace0 = go.Bar(

    x = zomato_india['Average Cost for two'].value_counts().index,

    y = np.log(zomato_india['Average Cost for two'].value_counts().values),

    text = (zomato_india['Average Cost for two'].value_counts().values),

)



layout = go.Layout(

    barmode ='group',

    shapes = [

        # Line reference to the axes

        {

            'type': 'line',

            'xref': 'x',

            'yref': 'y',

            'x0': zomato_india['Average Cost for two'].mean(),

            'y0': 0,

            'x1': zomato_india['Average Cost for two'].mean(),

            'y1': 8,

            'line': {

                'color': 'red',

                'width': 3,

                'dash': 'dashdot',

            },

        },

    ],

    annotations = [

        dict(

            x = zomato_india['Average Cost for two'].mean() + 5,

            y = 7,

            xref = 'x',

            yref = 'y',

            text = 'Average Price = 881.0 Rs',

            showarrow = True,

            arrowhead = 3,

            ax = 100,

            ay = 0,

        )

    ],

    title = go.layout.Title(

        text='India (Not including NCR)'  + '<br />' + 'Hover on bars for No. of Restaurants',

    ),

    xaxis = go.layout.XAxis(

        title = go.layout.xaxis.Title(

            text = 'Average Price in Rs. for Two People',

            font = dict(

                family = 'Courier New, monospace',

                size = 18,

                color = '#7f7f7f'

            )

        )

    ),

    yaxis = go.layout.YAxis(

        title = go.layout.yaxis.Title(

            text = 'Log10 of Number of restaurants',

            font = dict(

                family = 'Courier New, monospace',

                size = 18,

                color = '#7f7f7f'

            )

        )

    )

)



data = [trace0]



fig = go.Figure(data = data, layout = layout)

pyo.iplot(fig, filename='grouped-bar')
trace0 = go.Bar(

    x = zomato_ncr['Average Cost for two'].value_counts().index,

    y = np.log(zomato_ncr['Average Cost for two'].value_counts().values),

    text = (zomato_ncr['Average Cost for two'].value_counts().values),

)



layout = go.Layout(

    barmode ='group',

    shapes = [

        # Line reference to the axes

        {

            'type': 'line',

            'xref': 'x',

            'yref': 'y',

            'x0': zomato_ncr['Average Cost for two'].mean(),

            'y0': 0,

            'x1': zomato_ncr['Average Cost for two'].mean(),

            'y1': 8,

            'line': {

                'color': 'red',

                'width': 3,

                'dash': 'dashdot',

            },

        },

    ],

    annotations = [

        dict(

            x = zomato_ncr['Average Cost for two'].mean() + 5,

            y = 7,

            xref = 'x',

            yref = 'y',

            text = 'Average Price = 600.0 Rs',

            showarrow = True,

            arrowhead = 3,

            ax = 100,

            ay = 0,

        )

    ],

    title = go.layout.Title(

        text='India (NCR Only)' + '<br />' + 'Hover on bars for No. of Restaurants',

    ),

    xaxis = go.layout.XAxis(

        title = go.layout.xaxis.Title(

            text = 'Average Price in Rs. for Two People',

            font = dict(

                family = 'Courier New, monospace',

                size = 18,

                color = '#7f7f7f'

            )

        )

    ),

    yaxis = go.layout.YAxis(

        title = go.layout.yaxis.Title(

            text = 'Log10 of Number of restaurants',

            font = dict(

                family = 'Courier New, monospace',

                size = 18,

                color = '#7f7f7f'

            )

        )

    )

)



data = [trace0]



fig = go.Figure(data = data, layout = layout)

pyo.iplot(fig, filename='grouped-bar')
# Rated India (Not including NCR)

zomato_india_rated = zomato_india[zomato_india.Votes > 49]



# Rated India (NCR only)

zomato_ncr_rated = zomato_ncr[zomato_ncr.Votes > 49]
pvr_india = go.Scatter(

    y = zomato_india_rated['Aggregate rating'],

    x = zomato_india_rated['Average Cost for two'],

    mode = 'markers',

    marker = dict(

        size = 6,

        color = zomato_india_rated['Aggregate rating'], #set color equal to a variable

        colorscale = 'Viridis',

        showscale = True

    ),

)



pvr_ncr = go.Scatter(

    y = zomato_ncr_rated['Aggregate rating'],

    x = zomato_ncr_rated['Average Cost for two'],

    mode = 'markers',

    marker = dict(

        size = 6,

        color = zomato_ncr_rated['Aggregate rating'], #set color equal to a variable

        colorscale = 'Viridis',

        showscale = False

    ),

)



fig = tools.make_subplots(rows = 1, cols = 2,subplot_titles=('Non-NCR', 'NCR'))



fig['layout']['yaxis1'].update(title = 'Rating (Outside NCR) on scale of 5')

fig['layout']['yaxis2'].update(title = 'Rating (NCR) on scale of 5')



fig['layout']['xaxis1'].update(title = 'Avg Price for 2')

fig['layout']['xaxis2'].update(title = 'Avg Price for 2')



fig.append_trace(pvr_india, 1, 1)

fig.append_trace(pvr_ncr, 1, 2)



fig['layout'].update(height = 600, width = 800, title = 'Price Vs Rating')

pyo.iplot(fig, filename = 'simple-subplot-with-annotations')
zomato_ncr_votes = zomato_ncr_rated[zomato_ncr_rated.Votes >= 500]

zomato_ncr_rating = zomato_ncr_votes[zomato_ncr_votes['Aggregate rating'] > 3.9]

zomato_ncr_eco = zomato_ncr_rating[zomato_ncr_rating['Average Cost for two'] < 700]
data = [

    go.Scattermapbox(

        lat = zomato_ncr_eco['Latitude'],

        lon = zomato_ncr_eco['Longitude'],

        mode = 'markers',

        marker = go.scattermapbox.Marker(

            size = 5,

            color = '#cb202d'

        ),

        text = zomato_ncr_eco['Restaurant Name'] + '<br />' + zomato_ncr_eco['Locality Verbose'],

    )

]



layout = go.Layout(

    autosize = True,

    hovermode = 'closest',

    mapbox = go.layout.Mapbox(

        accesstoken = mapbox_access_token,

        bearing = 0,

        center = go.layout.mapbox.Center(

            lat = 28.59,

            lon = 77.22,

        ),

        pitch = 1,

        zoom = 9,

        style = 'light'

    ),

    title = 'Best Value for money Restaurants in NCR'

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig, filename='Multiple Mapbox')
zomato_ncr_bakery = zomato_ncr_eco.sort_values(by = 'Cuisines')[:5]
data = [

    go.Scattermapbox(

        lat = zomato_ncr_bakery['Latitude'],

        lon = zomato_ncr_bakery['Longitude'],

        mode = 'markers',

        marker = go.scattermapbox.Marker(

            size = 5,

            color = '#cb202d'

        ),

        text = zomato_ncr_bakery['Restaurant Name'] + '<br />' + zomato_ncr_bakery['Locality Verbose'],

    )

]



layout = go.Layout(

    autosize = True,

    hovermode = 'closest',

    mapbox = go.layout.Mapbox(

        accesstoken = mapbox_access_token,

        bearing = 0,

        center = go.layout.mapbox.Center(

            lat = 28.59,

            lon = 77.22,

        ),

        pitch = 1,

        zoom = 9,

        style = 'light'

    ),

    title = 'Best Value for money bakeries in NCR'

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig, filename='Multiple Mapbox')
fast_food_chains = ["Domino's Pizza", "Momo's King", 'Wow! India', "Dunkin' Donuts", 'Subway', "McDonald's",

                   "Pizza Hut", "Pizza Hut Delivery", "KFC", "Burger King", 'Chicago Pizza', 'Burger Point',

                   'Gopala', 'Yo! China', 'Ovenstory Pizza', 'RollsKing', 'Mad Over Donuts']



zomato_ncr_fast = zomato_ncr.loc[zomato_ncr['Restaurant Name'].isin(fast_food_chains)]



data = [

    go.Scattermapbox(

        lat = zomato_ncr_fast['Latitude'],

        lon = zomato_ncr_fast['Longitude'],

        mode = 'markers',

        marker = go.scattermapbox.Marker(

            size = 5,

            color = '#cb202d'

        ),

        text = zomato_ncr_fast['Restaurant Name'] + '<br />' + zomato_ncr_fast['Locality Verbose'],

    )

]



layout = go.Layout(

    autosize = True,

    hovermode = 'closest',

    mapbox = go.layout.Mapbox(

        accesstoken = mapbox_access_token,

        bearing = 0,

        center = go.layout.mapbox.Center(

            lat = 28.59,

            lon = 77.22,

        ),

        pitch = 1,

        zoom = 9,

    ),

    title = 'Fast Food Centers in NCR'

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig, filename='Multiple Mapbox')
# Download the image

# !wget --quiet https://www.redbytes.in/wp-content/uploads/2018/09/zomato-logo-AD6823E433-seeklogo.com_.png

    

# save mask to alice_mask

zomato_mask = np.array(Image.open('../input/zomato-logo-AD6823E433-seeklogo.com_.png'))



zomato_cuisines = []

for c in zomato_India['Cuisines']:

    listc = c.split(',')

    zomato_cuisines += listc

    

zomato_cuisines_text = ''

for text in zomato_cuisines:

    zomato_cuisines_text += (text + ' ')

    

stopwords = set(STOPWORDS)



# instantiate a word cloud object

zomato_wc = WordCloud(background_color='white', max_words=2000, mask=zomato_mask, stopwords=stopwords)



# generate the word cloud

zomato_wc.generate(zomato_cuisines_text)



# display the word cloud

fig = plt.figure()

fig.set_figwidth(8) # set width

fig.set_figheight(8) # set height



plt.imshow(zomato_wc, interpolation='bilinear')

plt.axis('off')

plt.show()
zomato_india['Has Table booking'].describe()
zomato_ncr['Has Table booking'].describe()
x0 = zomato_ncr_rated['Has Table booking']

y0 = zomato_ncr_rated['Aggregate rating']



trace0 = go.Box(

    y = y0,

    x = x0,

    marker = dict(

        color = '#FF851B',

    ),

    boxmean=True,

    boxpoints = 'outliers',

)



layout = go.Layout(

    title = "Table Booking Vs Rating"

)



data = [trace0]

fig = go.Figure(data = data, layout = layout)

pyo.iplot(fig)
top_fine_dine = zomato_ncr.loc[(zomato_ncr['Has Table booking'] == "Yes") & (zomato_ncr['Aggregate rating'] > 4) & (zomato_ncr['Votes'] > 200)].sort_values('Aggregate rating', ascending = False)



data = [

    go.Scattermapbox(

        lat = top_fine_dine['Latitude'],

        lon = top_fine_dine['Longitude'],

        mode = 'markers',

        marker = go.scattermapbox.Marker(

            size = 5,

            color = '#cb202d'

        ),

        text = top_fine_dine['Restaurant Name'] + '<br />' + top_fine_dine['Locality Verbose'],

    )

]



layout = go.Layout(

    autosize = True,

    hovermode = 'closest',

    mapbox = go.layout.Mapbox(

        accesstoken = mapbox_access_token,

        bearing = 0,

        center = go.layout.mapbox.Center(

            lat = 28.59,

            lon = 77.22,

        ),

        pitch = 1,

        zoom = 9,

    ),

    title = 'Top Fine Dines in NCR'

)



fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig, filename='Multiple Mapbox')
zomato_india['Has Online delivery'].value_counts()
zomato_ncr['Has Online delivery'].value_counts()
x0 = zomato_india_rated['Has Online delivery']

y0 = zomato_india_rated['Aggregate rating']



trace0 = go.Box(

    y = y0,

    x = x0,

    marker = dict(

        color = '#3D9970',

    ),

    boxmean=True,

    boxpoints = 'outliers',

)



layout = go.Layout(

    title = "Online Delivery Vs Rating (India not including NCR)"

)



data = [trace0]

fig = go.Figure(data = data, layout = layout)

pyo.iplot(fig)
x0 = zomato_ncr_rated['Has Online delivery']

y0 = zomato_ncr_rated['Aggregate rating']



trace0 = go.Box(

    y = y0,

    x = x0,

    marker = dict(

        color = '#FF4136',

    ),

    boxmean=True,

    boxpoints = 'outliers',

)

layout = go.Layout(

    title = "Online Delivery Vs Rating (India only NCR)"

)



data = [trace0]

fig = go.Figure(data = data, layout = layout)

pyo.iplot(fig)
pvr_india = go.Scatter(

    y = zomato_india_rated['Aggregate rating'],

    x = zomato_india_rated['Votes'],

    mode = 'markers',

    marker = dict(

        size = 4,

        color = zomato_india_rated['Aggregate rating'], #set color equal to a variable

        colorscale = 'Rainbow',

        showscale = True

    ),

)



pvr_ncr = go.Scatter(

    y = zomato_ncr_rated['Aggregate rating'],

    x = zomato_ncr_rated['Votes'],

    mode = 'markers',

    marker = dict(

        size = 4,

        color = zomato_ncr_rated['Aggregate rating'], #set color equal to a variable

        colorscale = 'Rainbow',

        showscale = False

    ),

)



fig = tools.make_subplots(rows = 1, cols = 2,subplot_titles=('Non-NCR', 'NCR'))



fig['layout']['yaxis1'].update(title = 'Rating (Outside NCR) on scale of 5')

fig['layout']['yaxis2'].update(title = 'Rating (NCR) on scale of 5')



fig['layout']['xaxis1'].update(title = 'Number of Votes')

fig['layout']['xaxis2'].update(title = 'Number of Votes')



fig.append_trace(pvr_india, 1, 1)

fig.append_trace(pvr_ncr, 1, 2)



fig['layout'].update(height = 600, width = 800, title = 'Votes Vs Rating')

pyo.iplot(fig, filename = 'simple-subplot-with-annotations')