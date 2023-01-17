import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import gc
import warnings
warnings.filterwarnings("ignore")
# Load and merge datasets. It may take a little while as the datasets are large.
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv')
df = donations.merge(donors, on="Donor ID", how="left")
# Delete and collect garbage
del donations, donors
gc.collect()
# A quick look at the variable names and data types
df.dtypes
# Get aggregated data at the state level
state = df.groupby('Donor State', as_index=False).agg({'Donor ID': 'nunique',
                                                       'Donation ID': 'count',
                                                       'Donation Amount':'sum'})    
# rename the columns
state.columns = ["State", "Donor_num", "Donation_num", "Donation_sum"]
# Get average donation amount
state["Donation_ave"]=state["Donation_sum"]/state["Donation_num"]
# Clean garbage
del df
gc.collect()
# A quick look at the dataframe we got
state.head()
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()
# Convert numerical variables into strings first
for col in state.columns:
    state[col] = state[col].astype(str)

state['text'] = state['State'] + '<br>' +\
    'Number of donors: $' + state['Donor_num']+ '<br>' +\
    'Number of donations: $'+ state['Donation_num']+ '<br>'+\
    'Average amount per donation: $' + state['Donation_ave']+ '<br>' +\
    'Total donation amount:  $' + state['Donation_sum']
state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

state['code'] = state['State'].apply(lambda x : state_codes[x])
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = state['code'], # The variable identifying state
        z = state['Donation_sum'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = state['text'], # Text to show when mouse hovers on each state
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(  
            title = "USD")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Donation by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )

fig = dict(data=data, layout=layout)
iplot(fig)