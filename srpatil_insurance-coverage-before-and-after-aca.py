# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/health-insurance")) # states.csv

print(os.listdir("../input/usstates")) # us-states.json

# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

from sklearn.preprocessing import Imputer



import warnings

warnings.filterwarnings('ignore')
# Plotly Packages and folium

from plotly import tools

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import folium
# Matplotlib and Seaborn and Folium Map Embed Function

import matplotlib.pyplot as plt

import seaborn as sns

from string import ascii_letters
# Path for installing new packages (<the output> -m pip install pandas)

# Install packages to the python 3 kernel in jupyter notebook

# //anaconda/envs/ipykernel_py3/bin/python -m pip install <package-name>



from sys import executable

print(executable)



# Plotly offline is surprisingly inconsistent regarding when iplot works/ does not work, increasing data rate limit 

# hels. If not a permanent and solution would be using the online api with user credentials and links to plots



# jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
ACA= pd.read_csv("../input/health-insurance/states.csv")

ACA.head()
# Data Cleaning



df = ACA.copy()



# State

df['State']=ACA['State'].str.strip()



# Uninsured Rate (2010)

df['Uninsured Rate (2010)']=ACA['Uninsured Rate (2010)'].str.strip("%").astype(float)



# Uninsured Rate (2015)

df['Uninsured Rate (2015)']=ACA['Uninsured Rate (2015)'].str.strip("%").astype(float)



# Uninsured Rate Change (2010-2015) 

df['Uninsured Rate Change (2010-2015)']=df['Uninsured Rate Change (2010-2015)'].str.strip("% ") # whitespace after %

df['Uninsured Rate Change (2010-2015)']=df['Uninsured Rate Change (2010-2015)'].str.replace('−', '-')

df['Uninsured Rate Change (2010-2015)']=df['Uninsured Rate Change (2010-2015)'].astype(float)



# Average Monthly Tax Credit (2016)

df['Average Monthly Tax Credit (2016)']=ACA['Average Monthly Tax Credit (2016)'].str.strip("$").astype(float)



# State Medicaid Expansion (2016)

df['State Medicaid Expansion (2016)']=ACA['State Medicaid Expansion (2016)']



# Medicaid Enrollment (2013) & Medicaid Enrollment (2016)

df['Medicaid Enrollment (2013)']=ACA['Medicaid Enrollment (2013)'].astype(float)

df['Medicaid Enrollment (2016)']=ACA['Medicaid Enrollment (2016)'].astype(float)



df.head()
ACA['Uninsured Rate Change (2010-2015)'].str.strip("% ").head()
# Load the shape of the zone (US states)

# Find the original file here: https://github.com/python-visualization/folium/tree/master/examples/data

# You have to download this file and set the directory where you saved it

# Folium package may give a value error in kaggle notebook, but works fine in desktop version



print(os.listdir("../input/health-insurance")) # states.csv

print(os.listdir("../input/usstates")) # us-states.json



state_geo = os.open ('../input/usstates/us-states.json',os.O_RDONLY )

 

# Load the unemployment value of each state

# Find the original file here: https://github.com/python-visualization/folium/tree/master/examples/data

Coverage = os.open('../input/health-insurance/states.csv',os.O_RDONLY )

#state_data = pd.read_csv(Coverage)



state_data= df

 

# Initialize the map:

m = folium.Map(location=[37, -102], zoom_start=5)

 

# Add the color for the chloropleth:

m.choropleth(

 geo_data=state_geo,

 name='choropleth',

 data=state_data,

 columns=['State', 'Uninsured Rate (2010)'],

 key_on='feature.properties.name',

 fill_color='YlGn',

 fill_opacity=0.5,

 line_opacity=0.5,

 legend_name='Uninsured Rate (2010) %',

 highlight=True

)

folium.LayerControl().add_to(m)

 

# Save to html

m.save('Uninsured_2010.html')
 # Initialize the map:

m = folium.Map(location=[37, -102], zoom_start=5)

 

# Add the color for the chloropleth:

m.choropleth(

 geo_data=state_geo,

 name='choropleth',

 data=state_data,

 columns=['State', 'Uninsured Rate (2015)'],

 key_on='feature.properties.name',

 fill_color='YlGn',

 fill_opacity=0.5,

 line_opacity=0.5,

 legend_name='Uninsured Rate (2015) %',

 highlight=True

)

folium.LayerControl().add_to(m)

 

# Save to html

m.save('Uninsured_2015.html')
# Initialize the map:

m = folium.Map(location=[37, -102], zoom_start=5)

 

# Add the color for the chloropleth:

m.choropleth(

 geo_data=state_geo,

 name='choropleth',

 data=state_data,

 columns=['State', 'Uninsured Rate Change (2010-2015)'],

 key_on='feature.properties.name',

 fill_color='YlGn',

 fill_opacity=0.5,

 line_opacity=0.5,

 legend_name='Uninsured Rate Change (2010-2015)',

 highlight=True

)

folium.LayerControl().add_to(m)

 

# Save to html

m.save('Uninsured_Rate_Change.html')
# Initialize the map:

m = folium.Map(location=[37, -102], zoom_start=5)

 

    

#  ColorBrewer has a minimum of 3 data classes, for boolean we need two   

# Add the color for the chloropleth:

m.choropleth(

 geo_data=state_geo,

 name='choropleth',

 data=state_data,

 columns=['State', 'State Medicaid Expansion (2016)'],

 key_on='feature.properties.name',

 fill_color='YlGn',      

 fill_opacity=0.5,

 line_opacity=0.5,

 legend_name='State Medicaid Expansion (2016):- 0=False and 1=True',

 highlight=True

)

folium.LayerControl().add_to(m)

 

# Save to html

m.save('State_Medicaid_Expansion.html')
# Make the dataset:

height = df['Average Monthly Tax Credit (2016)'][::9]

bars = df['State'][::9]

y_pos = np.arange(len(bars))



# Create bars

plt.bar(y_pos, height)

 

# Create names on the x-axis

plt.xticks(y_pos, bars)



plt.title("Average Monthly Tax Credit (2016)")



# Show graphic

plt.show()
# Make the dataset:

height = df['Marketplace Tax Credits (2016)'][::9]

bars = df['State'][::9]

y_pos = np.arange(len(bars))



# Create bars

plt.bar(y_pos, height)

 

# Create names on the x-axis

plt.xticks(y_pos, bars)



plt.title("Marketplace Tax Credits (2016)")



# Show graphic

plt.show()
import numpy as np

import matplotlib.pyplot as plt

 

# data to plot

n_groups = 6

medicaid_2013 = df['Medicaid Enrollment (2013)'][::9]

medicaid_2016 = df['Medicaid Enrollment (2016)'][::9]

 

# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.35

opacity = 0.8

 

rects1 = plt.bar(index, medicaid_2013, bar_width,

alpha=opacity,

color='b',

label='Medicaid Enrollment (2013)')

 

rects2 = plt.bar(index + bar_width, medicaid_2016, bar_width,

alpha=opacity,

color='g',

label='Medicaid Enrollment (2016)')

 

plt.xlabel('State')

plt.ylabel('Enrollment')

plt.title('Enrollment by State')

plt.xticks(index + bar_width, df['State'][::9])

plt.legend()

 

plt.tight_layout()

plt.show()  