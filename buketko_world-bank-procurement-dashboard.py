# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
proc_data = pd.read_csv('../input/procurement-notices/procurement-notices.csv')
proc_data.head(10)
proc_data.info()
# looks like the data types are all objects, we need to assign the correct data type 
# in order to use correct operations on columns
# we need to convert the Publication date and Deadline date into datetime data type
proc_data['Deadline Date'] = pd.to_datetime(proc_data['Deadline Date'])
proc_data['Publication Date'] = pd.to_datetime(proc_data['Publication Date'])
# We also want to tidy up the column names for easier scripting
proc_data.columns = proc_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
# now let's check the data types and new colun names
proc_data.info()
# we need to find the number of id's where the deadline is after today
import datetime
today = pd.Timestamp.today()
open_calls = proc_data[(proc_data.deadline_date > today) |(proc_data.deadline_date.isna())] 
#  and group the data by counrty name
calls_by_country  = open_calls.groupby(by ='country_name').size()
calls_by_country.head(5)
# now we need to convert this series into dataframe in order to add country codes 
# in the plot.ly example I will use country codes as the location codes for 'choropleth' type of map 
cc = calls_by_country.to_frame(name = 'size')
cc.sort_values('size', ascending=False, inplace=True)
cc.reset_index(inplace=True)
# I will change the column names slightly because the country codes dataset will be merged
# for a standard inner emrge, I will make column names identical between two data sets
cc.columns = ['COUNTRY', 'call_size']
cc.head()
# When I will pull the country codes from this data set the Kernel times out so I will upload the data 
df = pd.read_csv('../input/world-country-codes/country_codes.csv')
df.head(3)
# we don't need the GDP data so we will drop that column while merging the two data sets
map_data = pd.merge(cc, df).drop(columns=['GDP (BILLIONS)','Unnamed: 0'])
map_data.head(3)
# Now we can import the plotly libraries and plot the map data
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

iplot([go.Choropleth(
        locations = map_data['CODE'],
        z = map_data['call_size'],
        text = map_data['COUNTRY'],
    colorbar = dict(
            tickwidth = 0,
            title = 'number of calls')
    
    )])
# Time to create deadline distribution. First let's get t he dataframe
dd_dist = (open_calls.groupby('deadline_date').size()).to_frame(name = 'size')
dd_dist.head(3)
# Now we can plot it
data = go.Scatter(
    x = dd_dist.index,
    y= dd_dist['size']

)
iplot([data])