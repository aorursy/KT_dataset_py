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
df=pd.read_csv('../input/2015_16_Statewise_Elementary.csv')
df.head()

df.isnull().any()
import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import numpy as np



x0 = df['FEMALE_LIT'].values

x1 = df['MALE_LIT'].values



trace1 = go.Histogram(

    x=x0,

    name='Female',

    opacity=0.75

)

trace2 = go.Histogram(

    x=x1,

    name='Male',

    opacity=0.75

)



data = [trace1, trace2]

layout = go.Layout(title='literacy rate difference between females and males students',

    xaxis=dict(

        title='count'

    ),

    bargap=0.2,

    bargroupgap=0.1,

    barmode='stack')

fig = go.Figure(data=data, layout=layout)



iplot(fig)



x = df['GROWTHRATE'].values

data = [go.Histogram(x=x,

                     cumulative=dict(enabled=True))]

layout = go.Layout(title='GROWTHRATE OF EDUCATION IN INDIA ',

    xaxis=dict(

        title='GROWTHRATE'

    ),

    yaxis=dict(

        title='STATECOUNT'

    )

                   )

fig = go.Figure(data=data, layout=layout)



iplot(fig)
labels = df['STATNAME'].values

values = df['TOTPOPULAT'].values



trace = go.Pie(labels=labels, values=values)

layout = go.Layout(title='TOTALPOPULATION ')

fig = go.Figure(data=[trace], layout=layout)



iplot(fig)
l= []

y= []



N= 35

c= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]



for i in range(int(N)):

    y.append((1000+i))

    trace0= go.Scatter(

        x= df['STATCD'],

        y= df['P_URB_POP']+(i*3),

        mode= 'markers',

        marker= dict(size= 14,

                    line= dict(width=1),

                    color= c[i],

                    opacity= 0.3

                   ),name= y[i],

        text= df['STATNAME']) # The hover text goes here... 

    l.append(trace0);



layout= go.Layout(

    title= 'URBAN POPULATION',

    hovermode= 'closest',

    xaxis= dict(

        title= 'STATCD',

        

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'POPULATION',

        

        gridwidth= 2,

    ),

    showlegend= False

)

fig= go.Figure(data=l, layout=layout)

iplot(fig)