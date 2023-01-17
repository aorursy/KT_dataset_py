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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.figure_factory as ff

import plotly.graph_objects as go

%matplotlib inline
data = pd.read_csv(r'/kaggle/input/iris/Iris.csv')

data.head(10)
data['SepalLengthCm'][1]
print("Number of datapoints in the data : ",data.shape[0])

print("Number of features in the data : ",data.shape[1])

print("Features : ",data.columns.values)
data.info()
data.describe()
print("Number of duplicate rows in the dataset :",data.duplicated().sum())
fig = px.scatter(data,x='SepalLengthCm',y='SepalWidthCm',color='Species',template='seaborn')

fig.show()



fig = px.scatter(data,x='PetalLengthCm',y='PetalWidthCm',color='Species')

fig.show()
fig = px.scatter(data,x='SepalLengthCm',y='SepalWidthCm',color='PetalLengthCm',size='PetalLengthCm',hover_data=['PetalWidthCm'],template='seaborn')

fig.show()



fig = px.scatter(data,x='SepalLengthCm',y='SepalWidthCm',color='PetalWidthCm',size='PetalWidthCm',hover_data=['PetalLengthCm'])

fig.show()
fig = px.scatter(data,x='PetalLengthCm',y='PetalWidthCm',color='SepalLengthCm',size='SepalLengthCm',hover_data=['SepalWidthCm'])

fig.show()



fig = px.scatter(data,x='PetalLengthCm',y='PetalWidthCm',color='SepalWidthCm',size='SepalWidthCm',hover_data=['SepalLengthCm'],template='seaborn')

fig.show()
fig = px.line(data,x='SepalLengthCm',y='SepalWidthCm',color='Species')

fig.show()
fig = px.bar(data,x='SepalLengthCm',y='SepalWidthCm',color='Species',template='seaborn',hover_data=['PetalLengthCm','PetalWidthCm'])

fig.show()



fig = px.bar(data,x='PetalLengthCm',y='PetalWidthCm',color='Species',hover_data=['SepalLengthCm','SepalWidthCm'])

fig.show()
fig = px.bar(data,x='SepalLengthCm',y='PetalLengthCm',color='Species',template='seaborn',hover_data=['SepalWidthCm','PetalWidthCm'])

fig.show()



fig = px.bar(data,x='SepalLengthCm',y='PetalWidthCm',color='Species',hover_data=['SepalWidthCm','PetalLengthCm'])

fig.show()
fig = px.bar(data,x='Species',color='SepalLengthCm')

fig.show()



fig = px.bar(data,x='Species',color='SepalWidthCm',template='seaborn')

fig.show()
fig = px.bar(data,x='Species',color='PetalLengthCm')

fig.show()



fig = px.bar(data,x='Species',color='PetalWidthCm',template='seaborn')

fig.show()
fig = px.pie(data,names='Species',title='Species info',color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_traces(textinfo='percent+label',rotation=90,pull=0.02,marker=dict(line=dict(color='black',width=1.2)))

fig.show()
fig = px.pie(data,color='SepalLengthCm',names='Species',template='seaborn',title='Species info with Sepal Length')

fig.update_traces(textinfo='percent+label',rotation=90,pull=0.02,marker=dict(line=dict(color='black',width=1.2)))

fig.show()

fig = px.pie(data,color='SepalWidthCm',names='Species',template='seaborn',title='Species info with Sepal Width')

fig.update_traces(textinfo='percent+label',rotation=90,pull=0.02,marker=dict(line=dict(color='black',width=1.2)))

fig.show()

fig = px.pie(data,color='PetalLengthCm',names='Species',template='seaborn',title='Species info with Petal Length')

fig.update_traces(textinfo='percent+label',rotation=90,pull=0.02,marker=dict(line=dict(color='black',width=1.2)))

fig.show()

fig = px.pie(data,color='PetalWidthCm',names='Species',template='seaborn',title='Species info with Petal Width')

fig.update_traces(textinfo='percent+label',rotation=90,pull=0.02,marker=dict(line=dict(color='black',width=1.2)))

fig.show()
fig = px.sunburst(data,path=['Species','SepalLengthCm'],values='SepalWidthCm',color='SepalLengthCm',color_continuous_scale='RdBu',title='Species with Sepal Length as outer layer')

fig.show()



fig = px.sunburst(data,path=['Species','SepalWidthCm'],values='SepalLengthCm',color='SepalWidthCm',title='Species with Sepal Width as outer layer')

fig.show()



fig = px.sunburst(data,path=['Species','PetalLengthCm'],values='PetalWidthCm',color='PetalLengthCm',color_continuous_scale='gnbu',title='Species with Petal Length as outer layer')

fig.show()



fig = px.sunburst(data,path=['Species','PetalWidthCm'],values='PetalLengthCm',color='PetalWidthCm',template='seaborn',title='Species with Petal Width as outer layer')

fig.show()
fig = px.sunburst(data,path=['Species','SepalLengthCm','SepalWidthCm'],values='PetalLengthCm',color='PetalLengthCm',title='Species with Sepal Length and Sepal Width as outer layer')

fig.show()

fig = px.sunburst(data,path=['Species','PetalLengthCm','PetalWidthCm'],values='SepalWidthCm',template='seaborn',color='SepalWidthCm',title='Species with Petal Length and Petal Width as outer layer')

fig.show()
fig = px.sunburst(data,path=['Species','SepalLengthCm','SepalWidthCm','PetalLengthCm',],color='PetalWidthCm',color_discrete_map={'(?)':'black', 'Lunch':'gold', 'Dinner':'darkblue'},title='Species with Sepal Length , Sepal Width and Petal Length as outer layer')

fig.show()
fig = px.box(data,x='Species',y='SepalLengthCm',points='all')

fig.show()



fig = px.box(data,x='Species',y='SepalWidthCm',points='all',template='seaborn')

fig.show()



fig = px.box(data,x='Species',y='PetalLengthCm',points='all',template='seaborn')

fig.show()



fig = px.box(data,x='Species',y='PetalWidthCm',points='all',template='seaborn')

fig.show()
fig = px.box(data,x='Species',y='SepalLengthCm',points='all',color='SepalWidthCm')

fig.show()



fig = px.box(data,x='Species',y='SepalWidthCm',points='all',template='seaborn',color='PetalLengthCm')

fig.show()



fig = px.box(data,x='Species',y='PetalLengthCm',points='all',template='seaborn',color='PetalWidthCm')

fig.show()



fig = px.box(data,x='Species',y='PetalWidthCm',points='all',template='seaborn',color='SepalLengthCm')

fig.show()
fig = px.histogram(data,x='Species',color='SepalLengthCm')

fig.show()



fig = px.histogram(data,x='Species',color='SepalWidthCm',template='seaborn')

fig.show()
fig = px.histogram(data,x='SepalLengthCm',y='SepalWidthCm',color='Species',template='seaborn',hover_data=['PetalLengthCm','PetalWidthCm'])

fig.show()



fig = px.histogram(data,x='PetalLengthCm',y='PetalWidthCm',color='Species',hover_data=['SepalLengthCm','SepalWidthCm'])

fig.show()
hist_data = [data['SepalLengthCm'],data['SepalWidthCm'],data['PetalLengthCm'],data['PetalWidthCm']]

labels = ['Sepal Length','Sepal Width','Petal Length','Petal Width']

fig = ff.create_distplot(hist_data,labels,bin_size=.2)

fig.show()
hist_data = [data['SepalLengthCm'],data['SepalWidthCm'],data['PetalLengthCm'],data['PetalWidthCm']]

labels = ['Sepal Length','Sepal Width','Petal Length','Petal Width']

fig = ff.create_distplot(hist_data,labels,bin_size=.2,show_hist=False)

fig.show()
fig = px.violin(data,y='SepalLengthCm',box=True,points='all',color='Species',template='seaborn')

fig.show()



fig = px.violin(data,y='SepalWidthCm',box=True,points='all',color='Species')

fig.show()



fig = px.violin(data,y='PetalLengthCm',box=True,points='all',color='Species')

fig.show()



fig = px.violin(data,y='PetalWidthCm',box=True,points='all',color='Species')

fig.show()
fig = px.violin(data,y='SepalWidthCm',x='Species',box=True,points='all',color='Species')

fig.show()



fig = px.violin(data,y='PetalWidthCm',x='Species',box=True,points='all',color='Species')

fig.show()
fig = px.scatter(data, x="SepalLengthCm", y="SepalWidthCm", trendline="ols",color='Species')

fig.show()



fig = px.scatter(data, x="PetalLengthCm", y="PetalWidthCm", trendline="ols",color='Species')

fig.show()
fig = px.scatter_3d(data,x='SepalLengthCm',y='SepalWidthCm',z='PetalLengthCm',color='Species')

fig.show()
fig = px.scatter_3d(data,x='SepalLengthCm',y='SepalWidthCm',z='PetalWidthCm',color='PetalLengthCm',size='PetalWidthCm',template='seaborn',symbol='Species')

fig.show()
fig = px.line_3d(data,x='SepalLengthCm',y='SepalWidthCm',z='PetalWidthCm',color='Species')

fig.show()