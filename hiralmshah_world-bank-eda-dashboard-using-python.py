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

proc_data = pd.read_csv('../input/procurement-notices.csv')

proc_data.head(5)



proc_data.columns

proc_data['Deadline Date'] = pd.to_datetime(proc_data['Deadline Date'])

proc_data.dtypes



#number of calls currently out

#cells with NA deadline are currently out]



proc_data[(proc_data['Deadline Date'] > pd.Timestamp.today()) | (proc_data['Deadline Date'].isnull())].count().ID
#distribution by country

import plotly

import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected = True)



current_calls = proc_data[proc_data['Deadline Date'] > pd.Timestamp.today()]

calls_by_country = current_calls.groupby('Country Name').size()



plotly.offline.iplot({

    "data": [go.Choropleth(

        locations = calls_by_country.index,

        z = calls_by_country.values,

        locationmode='country names',

    )],

    "layout": go.Layout(title="number_of_bids")

})
due_dates = current_calls.groupby('Deadline Date').size()



plotly.offline.iplot({

    "data": [go.Scatter(

        x = due_dates.index,

        y = due_dates.values,

    )],

    "layout": go.Layout(title="deadline_date")

})
