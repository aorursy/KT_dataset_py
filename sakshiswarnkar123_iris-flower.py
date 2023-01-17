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
from sklearn.datasets import load_iris



iris_dataset = load_iris()



import numpy as np 

import pandas as pd 



import numpy as np

import matplotlib.pyplot as plt 



import pandas as pd  

import seaborn as sns 



%matplotlib inline
iris = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

iris.head()
iris
iris=sns.load_dataset(iris.csv)

iris.head()


iris['Species'] = iris_dataset.target
iris.head(20)

iris.info()
iris.index
import plotly.graph_objs as go

import plotly.offline as pyoff

import plotly.express as px
fig=px.line(x=["mike","mikel","mum","snoopy","husky"],y=[10,20,30,25,60],title="JUST A LINE GRAPH")

print(fig)

fig.show()
fig=go.Figure()

fig.add_trace(go.Scatter(

    x=iris.index,

    y=iris['sepal length (cm)'],

    mode="lines+markers",

  

))



fig.update_layout(

    title=" sepal length per species ",

    xaxis_title="INDEX",

    yaxis_title="sepal length",

   font_color='black',

   plot_bgcolor="#f8f8f8",

   paper_bgcolor="#f8f8f8"

)



fig.update_xaxes(title_font_family="ARIAL",title_font_size=12)

fig.update_yaxes(title_font_family="ARIAL",title_font_size=12)



fig.show()
fig = px.line(iris,x=iris.index,y=iris['sepal length (cm)'],title="sepal length per species")

fig.update_layout(

    font_family="Courier New",

    font_color="black",

    title_font_family="Times New Roman",

    title_font_color="black",

    title_font_size=20,

    plot_bgcolor='#f8f8f8',

    paper_bgcolor='#f8f8f8',

)

fig.update_xaxes(title_font_family="ARIAL",title_font_size=12,tickangle=-45)

fig.update_yaxes(title_font_family="ARIAL",title_font_size=12)

fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(

    x=iris.index,

    y=iris['sepal length (cm)'],

    mode="lines+markers",

    name="sepal length per species"

))





fig.add_trace(go.Scatter(

    x=iris.index,

    y=iris['petal length (cm)'],

    mode="lines+markers",

    name="petal length per species"

))



fig.update_layout(

    title="CUMMULATIVE PLOT OF LENGTH PER SPECIES ",

    xaxis_title="INDEX",

    yaxis_title="LENGTHS",

    font=dict(

        family="Candal",

        size=20,

        color="green"

    )

)



fig.show()
fig = go.Figure()



fig.add_trace(go.Bar(

    x=iris.index,

    y=iris['sepal length (cm)'],

    name=" sepal length per species"

))





fig.add_trace(go.Bar(

    x=iris.index,

    y=iris['petal length (cm)'],

    name=" petal length per species"

))



fig.update_layout(

    title="CUMMULATIVE PLOT OF LENGTH PER SPECIES",

    xaxis_title="INDEX",

    yaxis_title="LENGTHS",

    font_color="black",

    plot_bgcolor="#f8f8f8",

    barmode="group" #stack

   

)

fig.update_xaxes(title_font_family="ARIAL",title_font_color="black",title_font_size=12)

fig.update_yaxes(title_font_family="ARIAL",title_font_size=12,title_font_color="black")



fig.show()
fig=go.Figure()

fig.add_trace(go.Table(header=dict(values=["NAMES","AGE"],

                                   fill_color="lightpink",

                                   align="center"

                                   ),

                       cells=dict(values=[["AKASH","VIVEK","SIMRAN","KHUSHI"],[10,20,30,40]],

                             fill_color="lightskyblue",

                                   align="center"    

                                   )))

fig.show()
iris.info()
iris['sepal length (cm)']=iris['sepal length (cm)'].astype(int)
iris.info()
fig = px.scatter(iris, x="petal length (cm)", y="petal width (cm)",

        color="Species",animation_frame="sepal length (cm)",

                 animation_group="Species",

                 hover_name="Species",

                 log_x=True, 

                 size_max=100,

                 range_x=[1,10]

                 

                 )

fig.show()