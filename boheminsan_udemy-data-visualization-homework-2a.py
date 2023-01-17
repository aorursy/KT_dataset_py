# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df=pd.read_csv('../input/athlete_events.csv')#dataset will be edited below
dfy=pd.read_csv('../input/athlete_events.csv')#raw data

# Any results you write to the current directory are saved as output.
df.head()
df=df.dropna() # i don't wanna even one NaN values. 
#And over 180K rows are deleted.
df.Age=df.Age.astype('str')
# Creating trace1
df1=df[(df.Height>=200)].head(500)
df1.Height=df.Height.astype('str')
df1.Weight=df.Weight.astype('str')
trace1 = go.Scatter(
                    x = df1.head(150).Name,
                    y = df1.head(150).Height,
                    mode = "lines",
                    name = "Height",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= "Height: " + df1.Height + ", " + df1.Name
)
# Creating trace2
trace2 = go.Scatter(
                    x = df1.head(150).Name,
                    y = df1.head(150).Weight ,
                    mode = "lines+markers",
                    name = "Weight",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= "Weight: "+ df1.Weight + ", " + df1.Name
)
data2 = [trace1, trace2]
layout = go.Layout(
    autosize=False,
    width=1000,
    height=500,
              xaxis= dict(
                  title= 'Name',
                  ticklen= 20,
                  dtick=5 
              )
        )
  
fig = dict(data = data2, layout = layout)
iplot(fig)
df2=df[(df.Year>2008) & (df.Season=='Summer')].head(1000)
df2.describe()
trace1 = go.Scatter(
    x = df2.Name,
    y = df2.Age,
    mode = "markers",
    name = "Age",
    marker = dict(color = 'rgba(255, 112, 2, 0.8)'),
    text= df2.Name)
trace2 = go.Scatter(
    x = df2.Name,
    y = df2.Weight,
    mode = "markers",
    name = "Weight",
    marker = dict(color = 'rgba(44, 44, 255, 0.7)'),
    text= df2.Name)
trace3 = go.Scatter(
    x = df2.Name,
    y = df2.Height,
    mode = "markers",
    name = "Height",
    marker = dict(color = 'rgba(16, 255, 2, 0.8)'),
    text= df2.Name)
data2 = [trace1, trace2,trace3]
layout = go.Layout(
    autosize=False,
    width=1100,
    height=500,
    xaxis= dict(
        ticklen= 10,
        dtick=10))
fig = dict(data = data2, layout = layout)
iplot(fig)
df00=df[(df.Year==2000) & (df.Season=='Summer')]
df04=df[(df.Year==2004) & (df.Season=='Summer')]
df08=df[(df.Year==2008) & (df.Season=='Summer')]
df12=df[(df.Year==2012) & (df.Season=='Summer')]
df2016=df[(df.Year==2016) & (df.Season=='Summer')]

a=df00.Sex.value_counts()
df00=pd.Series.to_frame(a).T
b=df04.Sex.value_counts()
df04=pd.Series.to_frame(b).T
c=df08.Sex.value_counts()
df08=pd.Series.to_frame(c).T
d=df12.Sex.value_counts()
df12=pd.Series.to_frame(d).T
e=df2016.Sex.value_counts()
df16=pd.Series.to_frame(e).T

df3=pd.concat([df00,df04,df08,df12,df16],ignore_index=True,axis=0)
df3=pd.concat([df00,df04,df08,df12,df16],keys=['2000','2004','2008','2012','2016'],axis=0)
df3=df3.reset_index()
df3['level_0']=['2000','2004','2008','2012','2016']
df3=df3.rename(columns={'level_0': 'Year'})
df3=df3.drop(columns='level_1')
df3
trace1 = go.Bar(
                    x = df3.Year,
                    y = df3.M,
                    name = "Male",
                    marker = dict(color = 'rgba(25, 42, 86,1.0)',
                    line=dict(color='rgb(113, 128, 147)',width=1.5)),
                    text= df3.Year)
# Creating trace2
trace2 = go.Bar(
                    x = df3.Year,
                    y = df3.F,
                    name = "Female",
                    marker = dict(color = 'rgba(0, 168, 255,1.0)',
                    line=dict(color='rgb(113, 128, 147)',width=1.5)),
                    text= df3.Year)

data2 = [trace1, trace2]
layout = go.Layout(
    barmode = "group",
    autosize=False,
    width=1000,
    height=600,
              xaxis= dict(
                  ticklen= 2,
                  dtick=4,
                  title="Genders at Olympic Games"
              )
        )
  
fig = dict(data = data2, layout = layout)
iplot(fig)
# swarm plot
dfnew=df2016.head(2000)
fig, ax = plt.subplots(figsize=(15,8))
sns.swarmplot(x="Medal", y="Age",hue="Sex", size=5, data=dfnew, ax=ax)
plt.show()
pie1 = df2016.Medal
labels = df2016.Medal.unique()
pie1_list=list(df2016.Medal.value_counts())
# figure
fig = {
  "data": [
    {
      "values": pie1_list,
      "labels": labels,
      "domain": {"x": [0, .75]},
      "name": "Ratio Of Medals",
      "hoverinfo":"label+value+name",
      "hole": .5,
      "type": "pie"
    }],
  "layout": {
        "title":"Medals of 2016 Olympic Games",
        
    }
}
iplot(fig)
dfh1=df.groupby('Team')['Medal'].value_counts()
dfh1=pd.Series.to_frame(dfh1)
dfh1=dfh1.rename(columns={'Medal': 'MedalCount'})
dfh1=dfh1.T
countries=list(df.Team.unique())
countries.sort()
medals=[]
dft=pd.DataFrame()
for j in countries:
    toplam=0
    for k in dfh1[j]:
        toplam+=int(dfh1[j][k])
    #print(j,toplam)
    data = [toplam]
    abc=pd.DataFrame(data,columns=[j]).T
    dft=dft.append(abc,sort=False)
dft = dft.rename(columns={0: 'TotalMedal'})
dft=dft.sort_values(by=['TotalMedal'],kind='quicksort',ascending=False)
dft.head(15)
# data preparation
dfh11 = dft.TotalMedal.head(20)
dfh12=dft.head(20).index
num_students_size  = [float(each/20) for each in dft.TotalMedal]
international_color = [float(each) for each in dft.TotalMedal]
data = [
    {
        'y': dfh11,
        'x': dft.index,
        'mode': 'markers',
        'marker': {
            'color': international_color,
            'size': num_students_size,
            'showscale': True,
            'colorbar':dict(title='Colorbar'),
            'colorscale':'Viridis'
        },
        "text" :  dfh12 
    }
]
iplot(data)
trace1 = go.Histogram(
    x=df.Sport,
    opacity=0.8,
    name = "Top Events",
    nbinsx=10,
    autobinx = False,
    marker=dict(color='rgb(10, 220, 20)'))

data = [trace1]
layout = go.Layout(barmode='overlay',
                   title='Top Events',
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# data prepararion
dfwc=dfy.Team
plt.subplots(figsize=(11,6))
wordcloud = WordCloud(
                          background_color='black',
                          width=550,
                          height=300
                         ).generate(" ".join(dfwc))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()
trace0 = go.Box(
    y=df3.M,
    name = 'M',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=df3.F,
    name = 'F',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
df2016=df[(df.Year==2016) & (df.Season=='Summer')]
data2016 = df2016.loc[:,["Age", "Height", "Weight"]]
data2016["index"] = np.arange(1,len(data2016)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data2016.head(200), diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
dfev1=pd.Series.to_frame(df.Sport.value_counts())
dfev1=dfev1[(dfev1.Sport>=600)]
dfev2=pd.Series.to_frame(df.Event.value_counts())
dfev2=dfev2[(dfev2.Event>=200)]
# first line plot
trace1 = go.Scatter(
    x=dfev2.index,
    y=dfev2.Event,
    name = "Event",
    marker = dict(color = 'rgba(31, 58, 147, 1)'),
)
# second line plot
trace2 = go.Scatter(
    x=dfev1.index,
    y=dfev1.Sport,
    xaxis='x2',
    yaxis='y2',
    name = "Sport",
    marker = dict(color = 'rgba(22, 160, 133, 1)'),
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Top Events'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=df.head(100).Name,
    y=df.head(100).Height,
    z=df.head(100).Weight,
    mode='markers',
    marker=dict(
        size=8,
        color='rgba(84, 160, 255,1.0)',                # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=10,
        r=10,
        b=10,
        t=10  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
dfh11 = dft.TotalMedal.head(20)
dfh12=dft.head(20).index
trace1 = go.Scatter(
    x=dfh12,
    y=dfh11,
    name = "Total Medals",
    marker = dict(color = 'rgba(247, 202, 24, 1)')
)
trace2 = go.Scatter(
    x=df.head(100).Name,
    y=df.head(100).Weight,
    xaxis='x2',
    yaxis='y2',
    name = "Weight",
    marker = dict(color = 'rgba(30, 130, 76, 1)')
)
trace3 = go.Scatter(
    x=['2000','2004','2008','2012','2016'],
    y=df3.F,
    xaxis='x3',
    yaxis='y3',
    name = "Female",
    marker = dict(color = 'rgba(140, 20, 252, 1)')
)
trace4 = go.Scatter(
    x=['2000','2004','2008','2012','2016'],
    y=df3.M,
    xaxis='x4',
    yaxis='y4',
    name = "Male",
    marker = dict(color = 'rgba(31, 58, 147, 1)')
)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    autosize=False,
    width=1000,
    height=500,
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        tickmode='linear',
        ticks='outside',
        dtick=4,
        ticklen=8,
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        tickmode='linear',
        ticks='outside',
        tick0=0,
        dtick=4,
        ticklen=8,
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        tickmode='linear',
        ticks='outside',
        tick0=2000,
        dtick=4,
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = 'Some Stats About Olympic Games'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
