# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected= True)
import plotly.graph_objs as go

#word cloud lib
from wordcloud import WordCloud


#matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
timesData = pd.read_csv("../input/timesData.csv")
timesData.info()
timesData.head(10)
df = timesData.iloc[:100,:]

trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = "rgba(50,122,0,0.8)"),
                    text= df.university_name)

trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teacing",
                    marker = dict(color = "rgba(200,10,30,0.8)"),
                    text= df.university_name)

data = [trace1,trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data , layout = layout)
iplot(fig)
df2016 = timesData[timesData.year == 2016].iloc[:7,:]
df2016.head(7)
df2016.info()
# can't convert to float if it's ','
df2016.num_students.astype("float")
df2016.info()
df2016 = timesData[timesData.year == 2016].iloc[:7,:]
pie1 = df2016.num_students
pie1_list = [float(each.replace(',','.')) for each in df2016.num_students]

fig = {
    "data" : [
        {
            "values" : pie1_list,
            "labels" : df2016.university_name,
            "domain" : {"x": [0, .5]},
            "name": "Number Of Students Rates",
            "hoverinfo":"label+percent+name",
            "hole": .3,
            "type": "pie"
    },], 
    "layout": {
        "title":"Universities Number of Students rates",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Students",
                "x": 0.20,
                "y": 1
        },
    ]
}
}
iplot(fig)
