# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/daysholidayxl2/daysHoliday.xlsx')



#name the columns



title = ['country','holiday','public','total']

df.columns = title

location =df['country'].tolist()

len(location)



values = df['total'].tolist()

publicHolidays = df['public'].tolist()



holidays = df['holiday'].tolist()

import plotly as pl

import plotly.graph_objs as gobj

import pandas as pd

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot









#initializing the data variable

data = dict(type = 'choropleth',

            

            locations = location,

            locationmode = 'country names',

            colorscale= 'Portland',

            

            text= location,

            z=values,

            colorbar = {'title':'Country Colours', 'len':200,'lenmode':'pixels' })



layout = dict(geo = {'scope':'world'}, title_text ='Holidays per year')



col_map = gobj.Figure(data = [data],layout = layout)



iplot(col_map)


#initializing the data variable

data = dict(type = 'choropleth',

            locations = location,

            locationmode = 'country names',

            colorscale= 'Electric',

            text= location,

            z=publicHolidays,

            colorbar = {'title':'Country Colours', 'len':200,'lenmode':'pixels' })



layout = dict(geo = {'scope':'world'}, title_text ='Public holidays per year',)



col_map = gobj.Figure(data = [data],layout = layout)



iplot(col_map)
#initializing the data variable

data = dict(type = 'choropleth',

            locations = location,

            locationmode = 'country names',

            colorscale= 'Picnic',

            text= location,

            z=holidays,

            colorbar = {'title':'Country Colours', 'len':200,'lenmode':'pixels' })



layout = dict(geo = {'scope':'world'}, title_text ='Paid Holidays per year')



col_map = gobj.Figure(data = [data],layout = layout)



iplot(col_map)