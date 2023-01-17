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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

population=pd.read_excel(r'../input/population.xlsx')



population.head()

population.describe()
population.insert(5,"Boolean",(population['2015_female']>population['2015_male'])*1,True)

population.insert(6,"Percentage",(population["2015_total"]/population["2015_total"].

                                  sum())*100,True)

population.head()
from wordcloud import WordCloud, STOPWORDS

d = {}

for a, x in population[["name","Percentage"]].values:

    d[a] = x

wordcloud = WordCloud(background_color="white",max_font_size=50)

wordcloud.generate_from_frequencies(frequencies=d)

fig=plt.figure()

fig.set_figwidth(12)

fig.set_figheight(7)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
import matplotlib.cm as cm

import plotly.offline as po

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

po.init_notebook_mode(connected=True)
data = dict(type='choropleth',

locations = population['name'],

locationmode = 'country names', z = population["Boolean"],

text = population['name'], colorbar = dict(title='More'),

colorscale=[[0,"rgb(255, 0, 0)"],[1,"rgb(0,0,255)"]],    

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