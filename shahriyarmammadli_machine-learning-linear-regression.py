# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



#import plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/column_2C_weka.csv")
data.head()
data.info()
data.corr()
f,ax = plt.subplots(figsize=(15,8))

sns.heatmap(data.corr(),linewidths=.5,annot=True,fmt=".1f",ax=ax)

plt.show()
# import figure factory

import plotly.figure_factory as ff



# preparing data

data_matrix = data.loc[:,["pelvic_incidence","pelvic_tilt numeric","lumbar_lordosis_angle","sacral_slope","degree_spondylolisthesis"]]

data_matrix["index"] = np.arange(1,len(data_matrix)+1)





fig = ff.create_scatterplotmatrix(data_matrix,diag="histogram",index="index",colormap="Reds",colormap_type="cat",height=900,width=900)

iplot(fig)
sns.countplot(x="class", data=data)

data.loc[:,'class'].value_counts()
# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable

data1 = data[data["class"]=="Abnormal"]

x1 = data1.sacral_slope.values.reshape(-1,1)

y1 = data1.pelvic_incidence.values.reshape(-1,1)



#Scatter

plt.figure(figsize=(10,10))

plt.scatter(x1,y1,color="red")

plt.xlabel("sacral slope")

plt.ylabel("pelvic incidence")

plt.show()
# LinearRegression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

linear_r = LinearRegression()



# for predict

predict_shape = np.linspace(min(x1),max(x1),num=len(x1))

# fit

linear_r.fit(x1,y1)

# predicted

predict_result = linear_r.predict(predict_shape)

# R^2

print("R^2 score: ",linear_r.score(x1,y1))

#print("R^2 score with metrics: ",r2_score(y1,predict_result))



# Plot regression line and scatter

plt.plot(predict_shape,predict_result,color="black",linewidth=4)

plt.scatter(x1,y1,color = "red")

plt.xlabel("sacral slope")

plt.ylabel("pelvic incidence")

plt.show()