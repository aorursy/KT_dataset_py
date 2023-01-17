# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read csv (comma separated value) into data

data = pd.read_csv('../input/column_2C_weka.csv')

print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')
# to see features and target variable

data.head()
# Well know question is is there any NaN value and length of this data so lets look at info

data.info()
data.describe()
#heatmap

f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidths=0.6,linecolor="blue",fmt=".1f",ax=ax)

plt.show()
data["class"].value_counts().unique
sns.countplot(x="class", data=data)

plt.title("classes", color="green")

plt.show()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
pelvic=data["pelvic_incidence"]

pelvic_tilt=data["pelvic_tilt numeric"]

lumbar=data["lumbar_lordosis_angle"]

sacral_s=data["sacral_slope"]

radius=data["pelvic_radius"]



plt.subplots(figsize=(12,12))



plt.subplot(5,1,1)

plt.title("pelvic_incidence-pelvic_tilt numeric-lumbar_lordosis_angle-sacral_slope-pelvic_radius subplot")

plt.plot(radius,color="b",label="pelvic_radius")

plt.legend()

plt.grid()



plt.subplot(5,1,2)

plt.plot(lumbar,color="g",label="lumbar_lordosis_angle")

plt.legend()

plt.grid()



plt.subplot(5,1,3)

plt.plot(pelvic,color="r",label="pelvic_incidence")

plt.legend()

plt.grid()



plt.subplot(5,1,4)

plt.plot(radius,color="lime",label="pelvic_radius")

plt.legend()

plt.grid()



plt.subplot(5,1,5)

plt.plot(pelvic_tilt,color="b",label="pelvic_tilt numeric")

plt.legend()

plt.grid()





plt.show()
# import graph objects as "go"

import plotly.graph_objs as go



data2=data.copy()



# creating trace1

trace1 =go.Scatter(

                    x = np.arange(0,310),

                    y = data2[data2['class']=='Normal'].sacral_slope,

                    mode = "markers",

                    name = "Normal",

                    marker = dict(color = 'rgba(0, 100, 255, 0.8)'),

                    text= data2['class'])

# creating trace2

trace2 =go.Scatter(

                    x = np.arange(0,310),

                    y = data2[data2['class']=='Abnormal'].sacral_slope,

                    mode = "markers",

                    name = "Abnormal",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= data2['class'])



df = [trace1, trace2]

layout = dict(title = 'sacral_slope',

              xaxis= dict(title= 'index',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)

             )

fig = dict(data = df, layout = layout)

iplot(fig)
# pelvic_radius vs class scatter plot

# import graph objects as "go"

import plotly.graph_objs as go



# creating trace1

trace1 =go.Scatter(

                    x = np.arange(0,310),

                    y = data2[data2['class']=='Normal'].pelvic_radius,

                    mode = "markers",

                    name = "Normal",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= data2['class'])

# creating trace2

trace2 =go.Scatter(

                    x = np.arange(0,310),

                    y = data2[data2['class']=='Abnormal'].pelvic_radius,

                    mode = "markers",

                    name = "Abnormal",

                    marker = dict(color = 'rgba(125, 12, 255, 0.6)'),

                    text= data2['class'])



df = [trace1, trace2]

layout = dict(title = 'pelvic_radius',

              xaxis= dict(title= 'index',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)

             )

fig = dict(data = df, layout = layout)

iplot(fig)
pie1_list=data["class"].value_counts().values

labels = data["class"].value_counts().index

# figure

fig = {

  "data": [

    {

      "values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "class",

      "hoverinfo":"label+percent",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Class Type",

        "annotations": [

            { "font": { "size": 15},

              "showarrow": False,

              "text": "class",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
dataa = data[data['class'] =='Abnormal']

x=dataa["pelvic_incidence"].values.reshape(-1,1)

y=dataa["sacral_slope"].values.reshape(-1,1)



#plot

plt.figure(figsize=(10,10))

plt.scatter(x=x,y=y)

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.title("Abnormal Class")

plt.grid()

plt.show()
#linear regression

#sklearn library

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



#regression

linear_reg=LinearRegression()



#fit

linear_reg.fit(x,y)



#prediction

# we need these values to plot the regression line.

x_=np.linspace(min(x), max(x)).reshape(-1,1) # 

y_head=linear_reg.predict(x_)



#R2 score with LinearRegression library

print("R_square score: ",linear_reg.score(x,y))

# R2 score with sklearn.metrics

print("R_2 score with sklearn.metrics library: ",r2_score(y,linear_reg.predict(x)))



# Plot regression line and scatter

plt.subplots(figsize=(10,10))

plt.plot(x_, y_head, color='green', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.grid()

plt.show()
A=data[data["class"]=="Abnormal"]

N=data[data["class"]=="Normal"]
# pelvic_incidence vs sacral_slope scatter plot in terms of class type 

plt.figure(figsize=(8,8))

plt.scatter(A.pelvic_incidence,A.sacral_slope,color="red",label="abnormal",alpha=0.5)

plt.scatter(N.pelvic_incidence, N.sacral_slope,color="green",label="normal",alpha=0.5)

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.legend()

plt.grid()

plt.show()
# determine the values

data["class"]=[1 if i=="Abnormal" else 0 for i in data["class"]]



y=data["class"].values

x_data=data.drop(["class"],axis=1)
# normalize the values

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
# train test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#knn model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)
prediction
y_test
print("{} nn score: {}".format(3,knn.score(x_test,y_test)))
# find the convenient k value for range (1,21)

score_list=[]

for i in range(1,25):

    knn2=KNeighborsClassifier(n_neighbors=i)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.figure(figsize=(10,10))   

plt.plot(range(1,25),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=21)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)
print("{} nn score: {}".format(21,knn.score(x_test,y_test)))