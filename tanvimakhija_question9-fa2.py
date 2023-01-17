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
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import matplotlib.cm as cm

import plotly.offline as po

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

po.init_notebook_mode(connected=True)
import os

print(os.listdir("../input"))
population_df=pd.read_excel("../input/population.xlsx")
population_df.head()
population_df.insert(5,"Boolean",(population_df['2015_female']>population_df['2015_male'])*1,True)
population_df.head()
data = dict(type='choropleth',

locations = population_df['name'],

locationmode = 'country names', z = population_df['Boolean'],

text = population_df['name'], colorbar = dict(title='Comparison'),

colorscale=[[0,"rgb(255, 0, 0)"],[1,"rgb(0,64,255)"]],    

        autocolorscale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ))



layout = dict(title='Population Comparision',              

geo = dict(showframe = False, projection={'type':'mercator'},showlakes = False,

        showcoastlines = True,showland = True,

             ))



choromap = go.Figure(data = [data], layout = layout)

iplot(choromap, validate=False)
from wordcloud import WordCloud, STOPWORDS
population_df.insert(6,"Percentage",(population_df["2015_total"]/population_df["2015_total"].sum())*100,True)
population_df.drop(['Percentage'], axis=1,inplace=True)
population_df.head()
population_df.insert(6,"Percentage",(population_df["2015_total"]/population_df["2015_total"].sum())*100,True)
population_df.head()
country = dict(zip(population_df.name, population_df.Percentage))

plt.figure(figsize = (15,10))

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black')

wordcloud.fit_words(country)

plt.imshow(wordcloud,interpolation = 'bilinear')