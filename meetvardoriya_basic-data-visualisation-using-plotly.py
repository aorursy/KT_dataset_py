# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sn

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import iplot

import matplotlib.pyplot as plt

%matplotlib inline
pd.set_option('display.max_columns',None)

df1 = pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/file_02.csv')

df2 = pd.read_csv('/kaggle/input/daily-power-generation-in-india-20172020/State_Region_corrected.csv')

df2.head()
df = pd.concat([df2,df1],axis=1)

df.head()
df.shape
df.isnull().sum()
df = df.dropna()
df.head()
df.shape
df.columns = ['State/Ut','Area (km2)','index','Region','National Share','Date','Region1','ThermGenActual','ThermGenEst','NucGenAct','NucGenest','hydroact','hydroEst']
df = df.drop(['Region1','index'],axis = 1)

df.head()
df['Date'].unique()
df = df.drop(['Date'],axis = 1)

df.head()
def subplots(df):

    trace1 = go.Bar(x = df['State/Ut'],

                   y = df['Area (km2)'],

                   name = 'Area of the States in (km 2)',

                   marker = dict(color = ('rgb(239,62,91)')),

                   )

    trace2 = go.Bar(x = df['State/Ut'],

                   y = df['ThermGenActual'],

                   name = 'Thermal generation',

                   marker = dict(color = ('rgb(242,98,121)')),

                   xaxis = 'x2',

                   yaxis = 'y2',

                   )

    trace3 = go.Bar(x = df['State/Ut'],

                   y = df['NucGenAct'],

                   name = 'Nuclear generation',

                   marker = dict(color = ('rgb(75,37,109)')),

                   xaxis = 'x3',

                   yaxis = 'y3',

                   )

    trace4 = go.Bar(x = df['State/Ut'],

                   y = df['hydroact'],

                   name = 'HydroElectricity generation',

                   marker = dict(color = ('rgb(92,204,206)')),

                   xaxis = 'x4',

                   yaxis = 'y4',

                   )

    data = [trace1,trace2,trace3,trace4];

    layout = go.Layout( 

             xaxis = dict(domain = [0,0.45]),

             xaxis2 = dict(domain = [0.55,1]),

             xaxis3 = dict(domain = [0,0.45]),

             xaxis4 = dict(domain = [0.55,1]),

             yaxis = dict(domain =  [0,0.45]),

             yaxis2 = dict(domain = [0,0.45],anchor = 'x2'),

             yaxis3 = dict(domain = [0.55,1],anchor = 'x3'),

             yaxis4 = dict(domain = [0.55,1],anchor = 'x4'),

    );

    fig = go.Figure(data=data,layout=layout)

    iplot(fig)
subplots(df)
predictedcols = ['ThermGenEst','NucGenest','hydroEst']
def estimation(df,i,k):

    fig = px.bar(data_frame=df,x = 'State/Ut',y = i,labels={'x':'State/Ut','y':i},color_discrete_sequence=[k],opacity=0.8)

    fig.show()
show_list = ['ThermalGenerationEstimation','NuclearGenerationEstimation','HydroElectrictyGenerationEstimation']
color_list = ['red','aqua','darkmagenta']

for i,k,j in zip(predictedcols,color_list,show_list):

    print(f' stats for the <{j}> are shown below ↓')

    estimation(df,i,k)

    print("="*75)

    
def charts(df,i,k):

    chart = px.pie(df,values=k,names = i, height=600)

    chart.update_traces(textposition = 'inside',textinfo = 'percent+label')

    

    chart.update_layout(title_x = 0.5,

                       geo = dict(showframe = False,

                                 showcoastlines = False))

    chart.show()
show_list = ['ThermalGenerationEstimation','NuclearGenerationEstimation','HydroElectrictyGenerationEstimation']

for i,j  in zip(predictedcols,show_list):

    print(f' stats for the <{j}> are shown below ↓')

    charts(df,'State/Ut',i)

    print('='*75)