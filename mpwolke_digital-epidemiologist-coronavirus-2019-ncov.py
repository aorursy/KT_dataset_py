#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlG_n5AofvAGt6A_IwuJkc1b6et0rMVCVizzRacpMh19RKP0NN9A&s',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
corona = pd.read_csv('/kaggle/input/corona-virus-report/novel corona virus situation report who.csv')
corona.head()
corona.dtypes
corona["China"].plot.hist()

plt.show()
corona["Nepal"].plot.hist()

plt.show()
corona["India"].plot.hist()

plt.show()
corona["Germany"].plot.hist()

plt.show()
corona["France"].plot.hist()

plt.show()
corona["USA"].plot.hist()

plt.show()
corona["China"].plot.box()

plt.show()
corona["USA"].plot.box()

plt.show()
coronacorr=corona.corr()

coronacorr
sns.heatmap(coronacorr,annot=True,cmap='Greens')

plt.show()
#Necessary Functions: 

def pie_plot(labels, values, colors, title):

    fig = {

      "data": [

        {

          "values": values,

          "labels": labels,

          "domain": {"x": [0, .48]},

          "name": "Job Type",

          "sort": False,

          "marker": {'colors': colors},

          "textinfo":"percent+label+value",

          "textfont": {'color': '#FFFFFF', 'size': 10},

          "hole": .6,

          "type": "pie"

        } ],

        "layout": {

            "title":title,

            "annotations": [

                {

                    "font": {

                        "size": 25,



                    },

                    "showarrow": False,

                    "text": ""



                }

            ]

        }

    }

    return fig
import plotly.offline as py

value_counts = corona['USA'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts,['#1B9E77', '#7570B3'], "USA"))
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrZ-IJMwdsg6SwLSzpG4BSc8dD96Y4OTNi101qOdBUTE_xfmI&s',width=400,height=400)
import plotly.offline as py

value_counts = corona['China'].value_counts()

labels = value_counts.index.tolist()

py.iplot(pie_plot(labels, value_counts,['#1B9E77', '#7570B3'], "China"))
#codes by Andre Sionek

import plotly.express as px



# Grouping it by country

plot_data = corona.groupby(['China', 'Hong Kong', 'Thailand', 'Macau', 'Taipei', 'Japan', 'South Korea', 'Viet Nam' ], as_index=False).USA.sum()



fig = px.bar(plot_data, x='China', y='USA', color='Hong Kong')

fig.update_layout(

    title_text='Novel Coronavirus 2019-nCoV',

    height=500, width=1000)

fig.show()
import plotly.express as px



# Grouping it by Country 

plot_data = corona.groupby(['China', 'Hong Kong', 'Thailand', 'Macau', 'Taipei', 'Japan', 'South Korea', 'Viet Nam'], as_index=False).USA.sum()



fig = px.line_polar(plot_data, theta='China', r='USA', color='Hong Kong')

fig.update_layout(

    title_text='Novel Coronavirus 2019-nCoV ',

    height=500, width=1000)

fig.show()
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(corona.China, corona.Japan, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="China", y="Japan", data=corona, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$China$", "$Japan$");
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQiD8OMZ0LGFxKlE75sj0tFR7yDdAKpSRTAkve9AiID_IZhGRHEcA&s',width=400,height=400)