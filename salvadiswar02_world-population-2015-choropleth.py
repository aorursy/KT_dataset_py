# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#import plotly.plotly as py
#from plotly.graph_objs import *
#py.sign_in('salva02', 'dq4m5rc0Hwbg1i39tVWG')

import io
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#  Question 9 a
#  Choropleth map indicating more female population than male population

population = pd.read_csv("../input/population-csv/population.csv")

df = pd.read_csv('../input/2014-gdp/2014_world_gdp_with_codes.csv')

population = pd.merge(population,df,how='left',left_on='name',right_on='COUNTRY')

population['More_Female'] = np.where(population['2015_female'] > population['2015_male'],1,0)

population.head()
scl = [[0, 'rgb(30,144,255)'],[1, 'rgb(255,48,48)']]

data = [ dict(
        type = 'choropleth',
        locations = population['CODE'],
        z = population['More_Female'],
        text = population['name'],
        showlegend = True,
        colorscale = scl,
        autocolorscale = False,
        reversescale = True,
        showscale = True,
      ) ]

layout = dict(
    title = 'Male vs Female Population 2015',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        showlegend = True,
    )
  
)


fig = dict( data=data, layout=layout ,showlegend = True )

iplot( fig,validate=False, filename='d3-world-map' )

# b) Plot a word cloud for country name where size of the word represents the percentage 
#    of the total world population it has.


import numpy as np # linear algebra
import pandas as pd 


data = pd.read_csv("../input/population-csv/population.csv")

data['Percent_Population'] = (data['2015_total']/data['2015_total'].sum())*100

#print(data['Percent_Population'])

data_extract = data[['name','Percent_Population']]

d = {}
for name,x  in data_extract.values:
    d[name] = x
   
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width=2000, height=1000)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()