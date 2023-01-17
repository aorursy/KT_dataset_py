!pip install pytrends

# Setup for the search terms and colour scheme
search_terms = ['North','East','South','West']
color_schemes = [['#C3BFD4','#695D96','#27195C'],
    ['#FFFAE2','#D9C87C','#867217'],
    ['#FFE4E2','#D9817C','#861C17'],
     ['#C7E0CA','#61A869','#12681C']
    
]
        
    
#     # Setup for the search terms and colour scheme
# search_terms = ['North']
# color_schemes = [['#2c82c9']
    
# ]
# #https://paletton.com/#uid=74c1u0kdFrg3CJ-8gwyjHmdq-gN


import pandas as pd 
import numpy as np
from pytrends.request import TrendReq
pytrend = TrendReq()

dma_mapping = pd.read_csv('../input/google-trends-countydma-mapping/GoogleTrends_CountyDMA_Mapping.csv');
data_pack = dma_mapping
factor = 100/len(search_terms)
scales = []


pytrend.build_payload(kw_list=search_terms,geo='US')
df = pytrend.interest_by_region(resolution='DMA')
data_pack = pd.merge(df,data_pack, left_on='geoName', right_on='GOOGLE_DMA')
    
for search_term in search_terms:
    scales.append(factor/max(data_pack[search_term]-min(data_pack[search_term])))

breakout = []
for index, row in data_pack.iterrows():
    max_idx = max(range(len(row[search_terms])), key=row[search_terms].__getitem__)
    anchor_value = factor*max_idx
    v = (row[search_terms[max_idx]]-min(data_pack[search_terms[max_idx]]))*scales[max_idx]
    breakout.append(anchor_value+v)

# Use the split column for this comparison, otherwise just use the search term!
data_pack['breakout'] = breakout

label = []
major = []
for index, row in data_pack.iterrows():
    max_idx = max(range(len(row[search_terms])), key=row[search_terms].__getitem__)
    major.append(search_terms[max_idx])
    label_str=''
    for search_term in search_terms: 
        label_str = label_str +  str(row[search_term]) + '/'
    label_str = label_str + str(row['breakout'])
    label.append(label_str)
data_pack['label'] = label
data_pack['major'] = major



# Generate the FIPS codes from the mapping
fips = []
for index, row in data_pack.iterrows():
    state_fips = str(row['STATEFP']).zfill(2)
    cnty_fips = str(row['CNTYFP']).zfill(3)
    fips.append(state_fips+cnty_fips)

# Load the county geojson boundaries
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# Make a function that sets up a colour scale based on input category sets.
def generateDiscreteColourScale(colour_set):
    #colour set is a list of lists
    colour_output = []
    num_colours = len(colour_set)
    divisions = 1./num_colours
    c_index = 0.
    for cset in colour_set:
        num_subs = len(cset)
        sub_divisions = divisions/num_subs
        for subcset in cset:
            colour_output.append((c_index,subcset))
            colour_output.append((c_index + sub_divisions-0.001,subcset))
            c_index = c_index + sub_divisions
    colour_output[-1]=(1,colour_output[-1][1])
    return colour_output
colorscale = generateDiscreteColourScale(color_schemes)
print(colorscale)
import plotly.graph_objects as go

fig = go.Figure()


# Add a Choropleth map and use z to determine what you want to look at
fig.add_trace(go.Choropleth(
    geojson=counties,
    locations=fips,
    z=data_pack['breakout'].astype(float),
    colorscale=colorscale,
    marker_line_width=0.0, # line markers between states
    marker_line_color='rgba(0,0,0,0.1)',
    text=data_pack[search_terms[0]], # hover text
))

# If you want boundaries
fig.update_geos(
    visible=False, resolution=110, scope="usa",
    showcountries=False, countrycolor="Black",
    showsubunits=True, subunitcolor="Black",subunitwidth=0.1
)

# Sizing and titles
fig.update_layout(
    autosize=False,
    height=1000,
    width=1000,
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
)
# fig.update_traces(showscale=False) # Do I want the colour scale bar to show?
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()


# This section is to show just state boundaries!
import plotly.graph_objects as go

fig = go.Figure(go.Scattergeo())
fig.update_geos(
    visible=False, scope="usa",
    showcountries=False, countrycolor="Black",
    showsubunits=True, subunitcolor="Black",subunitwidth=1
)
fig.update_layout(height=2000,width=2000, margin={"r":0,"t":0,"l":0,"b":0}, 
          geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)')
                 )
fig.show()