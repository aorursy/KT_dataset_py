# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
population_df= pd.read_excel('../input/population.xlsx', sheet_name='Sheet1')
population_df['More_Female']=np.where(population_df['2015_female']>population_df['2015_male'],1,0)
country_code = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
population_df=population_df.merge(country_code, left_on='name', right_on='COUNTRY', how='left')
plotly.tools.set_credentials_file(username='rucha.dighe', api_key='R153G8dYXV9rC6srtFD1')
data = [ dict(
        type = 'choropleth',
        locations = population_df['CODE'],
        z = population_df['More_Female'],
        text = population_df['name'],
        colorscale = [[0,"rgb(5, 10, 172)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'World Population'),
      ) ]

layout = dict(
    title = 'World Population Map',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )


cloud_df=population_df[['name']].copy()
cloud_df['population percent']= population_df['2015_total']/population_df['2015_total'].sum()
d = {}
for a, x in cloud_df.values:
    d[a] = x
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
tuples = [tuple(x) for x in cloud_df.values]
wordcloud = WordCloud().generate_from_frequencies(tuples)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
