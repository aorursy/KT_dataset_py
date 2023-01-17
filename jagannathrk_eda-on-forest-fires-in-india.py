import pandas as pd ## for working with data.

import numpy as np ## for Linear Algebra.

import plotly.express as px ## Visualization

import plotly.graph_objects as go ## Again, Visualization

import matplotlib.pyplot as plt ## Again, Visualization

pd.set_option('display.max_rows',200)

import os ## data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore') ## I hate warnings.
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = os.path.join(dirname, filename)
df = pd.read_csv(path) 
## Let's see if the data has any null values or not.

df.isnull().sum() 
## Let's take a look at the data.

df.head() 
df['Percent_first'] = (df['2009-10']-df['2008-09'])/df['2008-09']

df['Percent_second'] = (df['2010-2011']-df['2009-10'])/df['2009-10']

df.fillna(0, inplace=True)
first = df[['2008-09', 'States/UTs', 'Percent_first', 'Percent_second']]

first.loc[:,'Year'] = '2008-09'

first.columns = ['Fires', 'States/UTs', 'Percent_first', 'Percent_second', 'Year']



second = df[['2009-10', 'States/UTs', 'Percent_first', 'Percent_second']]

second.loc[:,'Year'] = '2009-10'

second.columns = ['Fires', 'States/UTs', 'Percent_first', 'Percent_second', 'Year']



third = df[['2010-2011', 'States/UTs', 'Percent_first', 'Percent_second']]

third.loc[:,'Year'] = '2010-11'

third.columns = ['Fires', 'States/UTs', 'Percent_first', 'Percent_second', 'Year']





df1 = pd.concat([first,second,third])

del first,second,third
df1.head()  ## df1 will be our data that we'll use for analysis.
px.bar(df1, 'States/UTs', 'Fires', color='Year', title='Total Forest Fires by State')
px.line(df1, 'States/UTs', 'Fires', color='Year', title='Fires in States throughout Years')
temp = df1.groupby(by='States/UTs')['Fires'].sum().sort_values().reset_index()

px.bar(temp.tail(), 'States/UTs', 'Fires', color='Fires', title = 'states with most recorded forest fires')
temp.head() ## All these states have zero forest fires reported so we can't plot them.
temp = df1[df1['Year']=='2008-09'].sort_values(by='Fires')

px.bar(temp.tail(), 'States/UTs', 'Fires', title = 'Year : 2008-09')
temp = df1[df1['Year']=='2009-10'].sort_values(by='Fires')

px.bar(temp.tail(), 'States/UTs', 'Fires', title = 'Year : 2009-10')
temp = df1[df1['Year']=='2010-11'].sort_values(by='Fires')

px.bar(temp.tail(), 'States/UTs', 'Fires', title = 'Year : 2010-11')
temp = df1.groupby(by='Year')['Fires'].sum()

fig = go.Figure(data=[go.Pie(labels=temp.index, values=temp.values)])

fig.update_traces(marker=dict(line=dict(color='#000000', width=4)))

fig.show()
temp=df1.sort_values(by='Percent_first')

px.bar(temp, 'States/UTs', 'Percent_first', color='Percent_first', title = 'First Year')
temp=df1.sort_values(by='Percent_second')

px.bar(temp, 'States/UTs', 'Percent_second', color='Percent_second', title = 'Second Year')