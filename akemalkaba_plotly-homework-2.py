# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import plotly.graph_objs as go
from plotly import tools
import matplotlib.pyplot as plt


from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
timesData =pd.read_csv('../input/timesData.csv')
x2011 = timesData.student_staff_ratio[timesData.year == 2011]
x2012 = timesData.student_staff_ratio[timesData.year == 2012]

trace2 = go.Histogram(
    x=x2012,
    opacity=0.75,
    name ="2012",
    marker = dict(color='rgba(10,50,200,0.6)'))
trace1 = go.Histogram(
    x=x2011,
    opacity=0.75,
    name ="2011",
    marker = dict(color='rgba(100,50,200,0.6)'))
data = [trace1,trace2]

layout = go.Layout(barmode='overlay',
                  title='students-staff ratio',
                  xaxis=dict(title='students-staff ratio'),
                  yaxis=dict(title='Count')
                  )
figure = go.Figure(data=data, layout=layout)
iplot(figure)

x2011 = timesData.country[timesData.year ==2011]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=500,
                          height=400,).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()
x2015 = timesData[timesData.year == 2015]

trace1 = go.Box(
    y = x2015.total_score,
    name = 'total score of unis in 2015',
    marker = dict(color = 'rgb(10,10,140)',)
)

trace2 = go.Box(
    y = x2015.research,
    name = 'research of unis in 2015',
    marker = dict(color = 'rgb(100,10,10)',)
)

data = [trace1,trace2]
iplot(data)
dataFrame
import plotly.figure_factory as ff

dataFrame = timesData[timesData.year ==2015]
data2015 = dataFrame.loc[:,["research","international","total_score"]]
data2015["index"] = np.arange(1,len(data2015)+1)

fig = ff.create_scatterplotmatrix(data2015,diag ='box',index = 'index',colormap='Portland',colormap_type = 'cat',height=700,width =700)
iplot(fig)



