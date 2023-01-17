# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

from wordcloud import WordCloud
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
playData = pd.read_csv("../input/googleplaystore.csv")
playData.head()
artDesign = playData[playData.Category == 'ART_AND_DESIGN']

trace0 = go.Box(
    y=artDesign.Size,
    name = 'Art and Design kategorisindeki uygulamaların boyutları',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=artDesign.Installs,
    name = 'Art and Design kategorisindeki uygulamaların indirilme sayıları',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
trace1 = go.Scatter3d(
    x=playData.Rating,
    y=playData.Reviews,
    z=playData.Installs,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
x1milyon = playData.Category[playData.Installs =='1,000,000+']
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x1milyon))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
df2016 = playData[playData.Installs =='100,000+'].iloc[:100,:]
pie1 = df2016.Reviews
#pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4
labels = df2016.Genres
# figure
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"İndirilme Sayıları 100.000'den fazla olan uygulamaların kategorileri",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "İndirilme Sayıları",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)