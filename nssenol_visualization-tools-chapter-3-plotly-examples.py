# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)

from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
timesData = pd.read_csv('../input/timesData.csv')
timesData.info()
timesData.head(10)
df = timesData.iloc[:100,:]
import  plotly.graph_objs as go

trance1 = go.Scatter(
                     x = df.world_rank,
                     y = df.citations,
                     mode = "lines",
                     name = "citation",
                     marker = dict(color = 'rgba(134,45,78,0.8)'),
                     text = df.university_name)
trance2 = go.Scatter(
                     x = df.world_rank,
                     y = df.teaching,
                     mode = "lines+markers",
                     name = "teaching",
                     marker = dict(color = 'rgba(32,128,21,0.8)'),
                     text = df.university_name)
data = [trance1, trance2]
layout = dict(title = 'citation and Teaching vs World Rank of Top 100 Universities',
             xaxis = dict(title = 'World Rank', ticklen = 5, zeroline = False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:100,:]
df2015 = timesData[timesData.year == 2015].iloc[:100,:]
df2016 = timesData[timesData.year == 2015].iloc[:100,:]
import plotly.graph_objs as go
trace1 = go.Scatter(
                     x = df2014.world_rank,
                     y = df2014.citations,
                     mode = "markers",
                     name = "2014",
                     marker = dict(color = 'rgba(134,45,78,0.8)'),
                     text = df2014.university_name)
trace2 = go.Scatter(
                     x = df2015.world_rank,
                     y = df2015.citations,
                     mode = "markers",
                     name = "2015",
                     marker = dict(color = 'rgba(2,123,42,0.8)'),
                     text = df2015.university_name)
trace3 = go.Scatter(
                     x = df2016.world_rank,
                     y = df2016.citations,
                     mode = "markers",
                     name = "2015",
                     marker = dict(color = 'rgba(99,55,178,0.8)'),
                     text = df2015.university_name)
data = [trace1, trace2, trace3]
layout = dict(title = 'Citation Vs World Rank of top 100 Universities with 2014, 2015 and 2016 years',
    xaxis = dict(title = 'World Rank', ticklen = 9, zeroline = True),
    yaxis = dict(title = 'Citation', ticklen = 9, zeroline = True)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

                    



df2014 = timesData[timesData.year == 2014].iloc[:3,:]
df2014
df2014 = timesData[timesData.year == 2014].iloc[:3,:]
import plotly.graph_objs as go
trace1 = go.Bar(
    x = df2014.university_name,
    y = df2014.citations,
    name = 'citation',
    marker = dict(color = 'rgba(255,213,13,0.8)',
        line = dict(color = 'rgba(12,12,14)', width = 1.5)),
    text = df2014.country)
trace2 = go.Bar(
    x = df2014.university_name,
    y = df2014.citations,
    name = 'teaching',
    marker = dict(color = 'rgba(123,98,76,0.8)',
        line = dict(color = 'rgba(12,12,14,0.8)', width = 1.5)),
text = df2014.country)


data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)







df2014
df2014 = timesData[timesData.year == 2014].iloc[:3,:]
import plotly.graph_objs as go
x = df2014.university_name

trace1 ={
    'x': x,
    'y': df2014.citations,
    'name': 'citation',
    'type':'bar'
};
trace2 ={
    'x': x,
    'y': df2014.teaching,
    'name': 'teaching',
    'type':'bar'
        };
data = [trace1, trace2];
layout = {
        'xaxis':{'title': 'Top 3 universities'},
        'barmode':'relative',
        'title':'citations and teaching of top 3 universities in 2014'
        
        
    };

fig = go.Figure(data = data, layout = layout)
iplot(fig)




df2016 = timesData[timesData.year == 2016].iloc[:7,:]
df2016

df2016 = timesData[timesData.year == 2016].iloc[:7,:]
pie1 = df2016.num_students
pie1_list = [float(each.replace(',','.')) for each in df2016.num_students]
labels = df2016.university_name
fig = {
    "data": [
        {
        "values": pie1_list,
        "labels": labels,
        "domain": {"x":[0,0.5]},
        "name":"Number of students Rates",
        "hoverinfo": "label+percent+name",
        "hole":.3,
        "type": "pie"
        },
    ],
    "layout":{
        "title": "Universities Number of Students Rates",
        "annotations":[
            { "font":{"size":20},
             "showarrow":False,
             "text": "Number of Students",
                "x":0.20,
             "y":1
            },
        ]
    }
    
    
    
}
iplot(fig)
df2016
#Bubble Chart

df2016 = timesData[timesData.year == 2016].iloc[:20,:]
num_students_size = [float(each.replace(',','.')) for each in df2016.num_students] 
international_color = [float(each) for each in df2016.international]
data = [
    {
    'y': df2016.teaching,
    'x': df2016.world_rank,
    'mode': 'markers',
    'marker':{
        'color': international_color,
        'size': num_students_size,
        'showscale': True
            },
    "text" : df2016.university_name 
        }
      
    ]
iplot(data)
#Histogram 



timesData.head()
import  plotly.graph_objs as go
x2011 = timesData.student_staff_ratio[timesData.year == 2011]
x2012 = timesData.student_staff_ratio[timesData.year == 2012]
trace1 = go.Histogram(
    x = x2011,
    opacity = 0.74,
    name = "2011",
    marker = dict(color = 'rgba(123,213,1,0.6)'))
trace2 = go.Histogram(
    x = x2012,
    opacity = 0.74,
    name = "2012",
    marker = dict(color = 'rgba(223,120,113,0.6)'))

data = [trace1 , trace2]
layout = go.Layout(barmode = 'overlay',
                  title = 'Students-Staff Ratio in 2011 2012',
                  xaxis = dict(title = 'Students-staff ratio'),
                  yaxis = dict(title = 'Count'),
                  )
fig = go.Figure(data = data, layout = layout)
iplot(fig)
#World Cloud
#Do not forget to import word cloud
x2011 = timesData.country[timesData.year == 2011]
plt.subplots(figsize = (8,8))
wordcloud = WordCloud(
    background_color = 'white',
    width = 512,
    height = 384,
    ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('graph.png')
plt.show()





#Box Plot
x2015 = timesData[timesData.year == 2015]

trace0 = go.Box(
    y=x2015.total_score,
    name = 'total score of universities in 2015',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)

trace1 = go.Box(
    y = x2015.research,
    name = 'Total Score of Universities in 2015',
    marker = dict(
        color = 'rgb(12, 128, 128)', 
       )
)

data = [trace0, trace1]
iplot(data)



#Scatter Plot Matrix

import plotly.figure_factory as ff
dataFrame = timesData[timesData.year == 2015]
data2015 = dataFrame.loc[:,["research","international","total_score"]]
data2015["index"] = np.arange(1,len(data2015)+1)
fig = ff.create_scatterplotmatrix(data2015, diag = 'box', index = 'index', colormap = 'Portland',colormap_type = 'cat',height = 700, width = 700)
iplot(fig)


#Inset Plot


trace1 = go.Scatter(
    x = dataFrame.world_rank,
    y = dataFrame.teaching,
    name = 'Teaching',
    marker = dict(color = 'rgb(12, 128, 128)',)
)

trace2 = go.Scatter(
    x = dataFrame.world_rank,
    y = dataFrame.income,
    xaxis = 'x2',
    yaxis = 'y2',
    name = 'Income',
    marker = dict(color = 'rgb(112, 28, 158)',)
)

data = [trace1, trace2]
layout = go.Layout(
    xaxis2 = dict(domain=[0.6,0.95], anchor = 'y2'),
    yaxis2 = dict(domain=[0.6,0.95], anchor = 'x2'),
    title = 'Income and Teaching vs World Rank of Universities'
)

fig = go.Figure(data = data, layout = layout)
iplot(fig)



#3D Scatter Plot with Colorscaling


trace1 = go.Scatter3d(
    x = dataFrame.world_rank,
    y = dataFrame.teaching,
    z = dataFrame.citations,
    mode = 'markers',
    marker = dict(size = 10,  color='rgb(255,0,0)')
    )

data = [trace1]
layout = go.Layout(margin = dict(
    l = 0,
    r = 0,
    b = 0,
    t = 0
    )
)
fig = go.Figure(data = data, layout = layout)
iplot(fig)





#Multiple Subplots

trace1 = go.Scatter(
    x = dataFrame.world_rank,
    y = dataFrame.research,
    name = 'Research',
)
trace2 = go.Scatter(
    x = dataFrame.world_rank,
    y = dataFrame.citations,
    xaxis = 'x2',
    yaxis = 'y2',
    name = 'Citations',
)
trace3 = go.Scatter(
    x = dataFrame.world_rank,
    y = dataFrame.income,
    xaxis = 'x3',
    yaxis = 'y3',
    name = 'Income',
)
trace4 = go.Scatter(
    x = dataFrame.world_rank,
    y = dataFrame.total_score,
    xaxis = 'x4',
    yaxis = 'y4',
    name = 'Total_Score',
)

data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis = dict(domain=[0,0.45]),
    yaxis = dict(domain=[0,0.45]),
    xaxis2 = dict(domain=[0.55,1]),
    xaxis3 = dict(domain=[0,0.45], anchor = 'y3'),
    xaxis4 = dict(domain =[0.55, 1], anchor = 'y4'),
    yaxis2 = dict(domain =[0, 0.45]),
    yaxis3 = dict(domain =[0.55, 1]),
    yaxis4 = dict(domain =[0.55, 1], anchor = 'x4'),
    title = 'Research citation, income and total score Vs World Rank of Universities'
)
fig = go.Figure(data = data, layout = layout)
iplot(fig)




La