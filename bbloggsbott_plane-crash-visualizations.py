import numpy as np 
import pandas as pd

import os
from os import path
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
print(os.listdir("../input"))
import urllib
# Loading the data
data_d = pd.read_csv('../input/plane-crash/planecrashinfo_20181121001952.csv')
data_d.head()
# Loading geocoded data
data_geocoded = pd.read_csv('../input/air-crash-geocoded-20/geocoded_locations_new.csv')
data_geocoded.head()
py.init_notebook_mode(connected=True)

py.init_notebook_mode(connected=True)


data = [ dict(
        type = 'scattergeo',
        lon = data_geocoded['Longitude'],
        lat = data_geocoded['Latitude'],
        text = data_geocoded['Locations'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.6,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            color = 'red',
        ))]

layout = dict(
        title = 'Plane crashes<br>(Hover for crash locations)',
        geo = dict(
            showland = True,
            landcolor = "rgb(255, 255, 255)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='historic-crashes' )
fig = ff.create_distplot([data_geocoded['year']], ['Crashes'])
fig['layout'].update(title='Distplot of Crashes through years', xaxis=dict(title='Year'))
py.iplot(fig, filename='Basic Distplot')
data = [ dict(
        type = 'scattergeo',
        lon = data_geocoded['Longitude'][data_geocoded['year']<1940],
        lat = data_geocoded['Latitude'][data_geocoded['year']<1940],
        text = data_geocoded['Locations'][data_geocoded['year']<1940],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.6,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            color = 'red',
        ))]

layout = dict(
        title = 'Plane crashes Before 1940<br>(Hover for crash locations)',
        geo = dict(
            showland = True,
            landcolor = "rgb(255, 255, 255)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='historic-crashes' )
data = [ dict(
        type = 'scattergeo',
        lon = data_geocoded['Longitude'][(data_geocoded['year']>=1940).values & (data_geocoded['year']<1960).values],
        lat = data_geocoded['Latitude'][(data_geocoded['year']>=1940).values & (data_geocoded['year']<1960).values],
        text = data_geocoded['Locations'][(data_geocoded['year']>=1940).values & (data_geocoded['year']<1960).values],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.6,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            color = 'red',
        ))]

layout = dict(
        title = 'Plane crashes 1940-1960<br>(Hover for crash locations)',
        geo = dict(
            showland = True,
            landcolor = "rgb(255, 255, 255)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='historic-crashes' )
data = [ dict(
        type = 'scattergeo',
        lon = data_geocoded['Longitude'][(data_geocoded['year']>=1960).values & (data_geocoded['year']<1980).values],
        lat = data_geocoded['Latitude'][(data_geocoded['year']>=1960).values & (data_geocoded['year']<1980).values],
        text = data_geocoded['Locations'][(data_geocoded['year']>=1960).values & (data_geocoded['year']<1980).values],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.6,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            color = 'red',
        ))]

layout = dict(
        title = 'Plane crashes 1960-1980<br>(Hover for crash locations)',
        geo = dict(
            showland = True,
            landcolor = "rgb(255, 255, 255)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='historic-crashes' )
data = [ dict(
        type = 'scattergeo',
        lon = data_geocoded['Longitude'][(data_geocoded['year']>=1980).values & (data_geocoded['year']<2000).values],
        lat = data_geocoded['Latitude'][(data_geocoded['year']>=1980).values & (data_geocoded['year']<2000).values],
        text = data_geocoded['Locations'][(data_geocoded['year']>=1980).values & (data_geocoded['year']<2000).values],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.6,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            color = 'red',
        ))]

layout = dict(
        title = 'Plane crashes 1980-2000<br>(Hover for crash locations)',
        geo = dict(
            showland = True,
            landcolor = "rgb(255, 255, 255)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='historic-crashes' )
data = [ dict(
        type = 'scattergeo',
        lon = data_geocoded['Longitude'][(data_geocoded['year']>=2000).values & (data_geocoded['year']<2020).values],
        lat = data_geocoded['Latitude'][(data_geocoded['year']>=2000).values & (data_geocoded['year']<2020).values],
        text = data_geocoded['Locations'][(data_geocoded['year']>=2000).values & (data_geocoded['year']<2020).values],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.6,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            color = 'red',
        ))]

layout = dict(
        title = 'Plane crashes 2000-2020<br>(Hover for crash locations)',
        geo = dict(
            showland = True,
            landcolor = "rgb(255, 255, 255)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='historic-crashes' )
text = ' '.join(data_d['route'][data_d['route']!='?'].values)
import urllib
file = urllib.request.urlopen('https://i.pinimg.com/564x/f8/98/f2/f898f2b1d68f0218f7dbc2a459a60bb0.jpg')
img = Image.open(file)
alice_mask = np.array(img)
stopwords = set(STOPWORDS)
wc = WordCloud(background_color="white", max_words=200, mask=alice_mask,
               stopwords=stopwords, contour_width=0)

# generate word cloud
wc.generate(text.lower())
plt.figure(figsize = (10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
text = ' '.join(data_d['ac_type'][data_d['ac_type']!='?'].values)
file = urllib.request.urlopen('https://i.pinimg.com/564x/f8/98/f2/f898f2b1d68f0218f7dbc2a459a60bb0.jpg')
img = Image.open(file)
alice_mask = np.array(img)
stopwords = set(STOPWORDS)
wc = WordCloud(background_color="white", max_words=200, mask=alice_mask,
               stopwords=stopwords, contour_width=0)

# generate word cloud
wc.generate(text.lower())
plt.figure(figsize = (10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
text = ''
for i in data_d['summary']:
    if i != '?':
        text = text + i

file = urllib.request.urlopen('https://i.pinimg.com/564x/f8/98/f2/f898f2b1d68f0218f7dbc2a459a60bb0.jpg')
img = Image.open(file)
alice_mask = np.array(img)
stopwords = set(STOPWORDS)
stopwords.add("aircraft")
stopwords.add("crashed")
stopwords.add("airplane")
stopwords.add("plane")
stopwords.add("helicopter")
wc = WordCloud(background_color="white", max_words=200, mask=alice_mask,
               stopwords=stopwords, contour_width=0)

# generate word cloud
wc.generate(text.lower())
plt.figure(figsize = (10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
fatalities = []
for i in data_geocoded['fatalities']:
    n = i.split()[0]
    if n != '?':
        if int(n) <=100:
            fatalities.extend([int(n)])

fig = ff.create_distplot([fatalities], ['Fatalities'])
layout = go.Layout(
    title='Distplot of Fatalities',
    xaxis=dict(
        title='No. of Fatalities',
        titlefont=dict(
            color='#000000'
        )
    )
)
fig['layout'].update(title='Distplot of Fatalities', xaxis=dict(
        title='No. of Fatalities',
        titlefont=dict(
            color='#7f7f7f'
        )
    ))
py.iplot(fig, filename='Basic Distplot')