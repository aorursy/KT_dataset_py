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
#import pandas as pd

#import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

import pycountry

import plotly.express as px
colors = ["#3366cc","#dc3912","#ff9900","#109618","#990099","#0099c6","#dd4477","#66aa00","#b82e2e","#316395","#3366cc","#994499","#22aa99","#aaaa11","#6633cc","#e67300",\

          "#8b0707","#651067","#329262","#5574a6","#3b3eac","#b77322","#16d620","#b91383","#f4359e","#9c5935","#a9c413","#2a778d","#668d1c","#bea413","#0c5922","#743411"]
# importing the libraries

from bs4 import BeautifulSoup

import requests

import lxml





url = "https://www.quebec.ca/en/health/health-issues/a-z/2019-coronavirus/situation-coronavirus-in-quebec/"



# Make a GET request to fetch the raw HTML content

html_content = requests.get(url).text



# Parse the html content

soup = BeautifulSoup(html_content, "lxml")

# print(soup.prettify()) # print the parsed data of html
gdp_table = soup.find("table", attrs={"class": "contenttable"})
from IPython.display import display_html

display_html(str(gdp_table), raw=True)
dfs = pd.read_html(str(gdp_table))

dfx = dfs[0]
for col in dfx.columns:

    dfx.rename(columns={col:col.replace('\xa0',' ')}, inplace=True)

    print(col)
dfx.columns

label = dfx.columns[1]

label
dfx.rename(columns={ dfx.columns[1]: "cnt" }, inplace = True)
dfx = dfx.drop(18)

dfx = dfx.drop(19)
dfx['cnt'] = dfx['cnt'].str.replace(u'\xa0','')
dfx.isna().sum()
dfx.info()
fig = px.treemap(dfx.sort_values(by='cnt', ascending=False).reset_index(drop=True), 

                 path=["Regions"], values="cnt", height=700, #width=1200,

                 title='Quebec - ' + label) #px.colors.qualitative.Prism color_discrete_sequence = colors

fig.data[0].textinfo = 'label+text+value'

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=dfx['Regions'],

                y=dfx['cnt'],

                name='Count',

                marker_color=colors[1],

                text=dfx['cnt']

                ))



fig.update_layout(

    autosize=False,

    #width=1400,

    height=600,            

    title='Quebec - ' + label,

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number of Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    plot_bgcolor='#DDD',

    paper_bgcolor='rgb(255,255,255)', # set the background colour

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.2, # gap between bars of adjacent location coordinates.

    bargroupgap=0.01 # gap between bars of the same location coordinate.

)

#https://plotly.com/python/setting-graph-size/

fig.update_yaxes(automargin=True)

fig.update_traces(textposition='outside')

#fig.update_layout(uniformtext_minsize=9, uniformtext_mode='show')            

fig.show()