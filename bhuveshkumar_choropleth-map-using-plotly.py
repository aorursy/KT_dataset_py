# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import basic modules which help in data processing 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Importing plotly to help visualise the Choropleth Maps
import plotly.graph_objects as plot

# Importing below given modules to help in Data Scrapping
import requests
import lxml.html as lh

# importing datetime module
from datetime import date

# getting today's date
today = date.today()
today = today.strftime("%B %d, %Y")


# function to scrap data from https://www.mygov.in/covid-19/

def get_data():
    url='https://www.mygov.in/covid-19/'
    
    #Create a handle, page, to handle the contents of the website
    page = requests.get(url)
    
    #Store the contents of the website under doc
    doc = lh.fromstring(page.content)
    #Parse data that are stored between <tr>..</tr> of HTML

    table = []

    table_row_element = doc.xpath('//tr')
    
    # for loop from 1 to 38 as I know the data for states is till 37
    for i in range(1,38):
        # initialising some variables to help in
        # save data in a strcutured form
        j = 0
        table_cols = []

        for elem in table_row_element[i]:
            # getting first content of table (state name)
            j+=1
            elem_content = elem.text_content()
        
            if j == 1:
                table_cols.append(elem_content)
         
            if j == 2:
                # getting second content of table (active cases count)
                # the count can be in string form so using
                # .replace to remove any ',' commas and converting string into integer
                table_cols.append(int(elem_content.replace(',', '')))
                table.append(table_cols)
                
                # other elements are not required so
                # breaking the loop
                break

    # Create the pandas DataFrame 
    df = pd.DataFrame(table, columns = ['state', 'active cases']) 
    return df


# before moving forward, I want to import the geojson file
# which is requried by plotly to draw the map of india (state-wise)
# I search for geojson data and fount it on github
# I want to give credits to the author of the geojson data
# link: https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw
# /e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson

# More details:
# I downloaded the grojson file and edited it a bit to fit my requiment.

dataset = get_data()
#dropping lakshydeep
dataset = dataset.drop(19)

# dropping daman and diu as it is not in geojson file
dataset = dataset.drop(8)


#reset the index
dataset.reset_index(inplace=True)

# adding new column of state names 
# which is as per the geojson key (id)
list_of_states = ['Andaman & Nicobar', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
 'Dadra and Nagar Haveli and Daman and Diu', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir',
 'Jharkhand', 'Karnataka', 'Kerala', 'Ladakh', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttarakhand',
 'Uttar Pradesh', 'West Bengal']

# adding the new column with states names as per geojson
dataset['new state'] = pd.Series(list_of_states).values

# fitting the data into plotly's choropleth function
india_map = plot.Figure(data=plot.Choropleth(
    geojson='https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson',
    featureidkey='properties.ST_NM',
    locationmode='geojson-id',
    locations=dataset['new state'],
    z=dataset['active cases'],
    autocolorscale=False,
    colorscale='blues',
    marker_line_color='black',

    colorbar=dict(title={'text': "Active Cases"},
                          thickness=10,
                          len=0.5,
                          bgcolor='rgba(255,255,255,0.2)',
                          tick0=0,
                          dtick=30000,)))

# can find details here https://plotly.com/python/map-configuration/
india_map.update_geos(visible=False,
                      projection=dict(
                      type='conic conformal',
                      parallels=[12.472944444, 35.172805555556],
                      rotation={'lat': 24, 'lon': 80}),
    lonaxis={'range': [68, 98]},
    lataxis={'range': [6, 38]})

# add a tittle
india_map.update_layout(
    title=dict(
        text="Active COVID-19 Cases in India by State as of " + str(today),
        xanchor='center',
        x=0.5,
        yref='paper',
        yanchor='bottom',
        y=1,
        pad={'b': 5}),
    
    margin={'r': 0, 't': 50, 'l': 0, 'b': 0},
    height=600,
    width=800)

# display map
india_map.show()
