# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import plotly.plotly as py
import datetime
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib
init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/procurement-notices.csv', delimiter=',')
df1.head(15)
#delete the T00:00:00 part from the deadline date column
df1 ['Deadline Date'] = pd.to_datetime(df1['Deadline Date'])
df1.head(15)
#how many null values are there in each column?
print("dataframe shape:",df1.shape)
print("nulls on each column:\n",df1.iloc[:,:].isnull().sum()) 
#deadlines after today, the ones with NaT do not have a deadline
#so they are included
df1[(df1['Deadline Date'] > pd.Timestamp.today()) | 
    (df1['Deadline Date'].isnull())].count().ID
# distribution by country
current_calls = df1[(df1['Deadline Date'] > pd.Timestamp.today()) | 
    (df1['Deadline Date'].isnull())]
calls_by_country = current_calls.groupby('Country Name').size()
print("calls_by_country:\n",calls_by_country)
iplot([go.Choropleth(
    locationmode='country names',
    locations=calls_by_country.index.values,
    text=calls_by_country.index,
    z=calls_by_country.values
)])
# distribution of due dates
ax = current_calls.groupby('Deadline Date').size().plot.line(figsize = (12,6))
ax.set_title('Number of Deadlines per Date')
#some tests to print differently did not work
# distribution of due dates
#newDates=pd.DataFrame(matplotlib.dates.date2num(current_calls['Deadline Date']))
#print(df1['Formated_Dates'])
#current_calls.loc['Formated_Dates']=newDates.loc['0']
#newDates.index.name='index','formDates'
#datesForGraph=newDates.loc['formDates']
#print(newDates.loc['0,:0,'])


