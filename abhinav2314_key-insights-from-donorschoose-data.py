import pandas as pd
import numpy as np

import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

folder = "../input/"
schools = pd.read_csv(folder + "Schools.csv", error_bad_lines=False)
school_county = schools['School State'].value_counts().reset_index()
school_county.columns = ['state', 'schools']

for col in school_county.columns:
    school_county[col] = school_county[col].astype(str)

state_codes = {'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

school_county['code'] = school_county['state'].map(state_codes)
word_string = schools['School Name'].str.cat(sep=' ')

wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='white',
    width=3000,
    height=1000).generate(word_string)

plt.figure(figsize=(20,40))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
scl = [[0, 'rgb(242,240,247)'],[.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],
            [0.6, 'rgb(158,154,200)'],[.8, 'rgb(117,107,177)'],[1, 'rgb(84,39,143)']]

# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = school_county['code'], # The variable identifying state
        z = school_county['schools'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = school_county['state'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "# of Schools")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = '# of Schools by State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)