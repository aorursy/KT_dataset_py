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

import pandas as pd

from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.cm as cm

import plotly.offline as po

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

po.init_notebook_mode(connected=True)



population=pd.read_excel("../input/population.xlsx")

population.head(4)

population.insert(5,"Percentage",(population["2015_total"]/population["2015_total"].

                                  sum())*100,True)

population.insert(6,"Boolean",(population['2015_female']>population['2015_male'])*1,True)

population.head()

data = dict(type='choropleth',

locations = population['name'],

locationmode = 'country names', z = population["Boolean"],

text = population['name'], colorbar = dict(title='more'),

colorscale=[[0,"rgb(176,58,46)"],[1,"rgb(100,100,255)"]],    

        autocolorscale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ))

layout = dict(title='Population distribution',              

geo = dict(showframe = False, projection={'type':'mercator'},showlakes = False,

        showcoastlines = True,showland = False

             ))

choromap = go.Figure(data = [data], layout = layout)

iplot(choromap, validate=False)



z=population.name

y=population.Percentage

frames = [z,y]

result = pd.concat(frames,axis=1)

d = dict(zip(population.name, population.Percentage))

plt.figure(figsize = (15,10))

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black')

wordcloud.fit_words(d)

plt.imshow(wordcloud,interpolation = 'bilinear')




