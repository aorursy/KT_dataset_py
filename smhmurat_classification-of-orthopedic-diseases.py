# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected = True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_2C = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

data_3C = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')
data_2C.info()
data_3C.info()
data_2C.head()
data_2C.pelvic_incidence
data_3C.columns
data_3C.pelvic_incidence = data_3C.pelvic_incidence.astype(float)

data_3C.lumbar_lordosis_angle = data_3C.lumbar_lordosis_angle.astype(float)
data_3C['class'].head()
data_3C.corr()
#Correlation Matrix

ax=plt.subplots(figsize=(18,18))

ax=sns.heatmap(data_3C.corr(), vmin=-1, vmax=1, center=0, annot=True, linewidths=1, cmap=plt.get_cmap("Spectral", 10))
#Scatter Star Poly

#pelvic_incidence <-> sacral_slope (0.81) (+)

#pelvic_incidence <-> lumbar_lordosis_angle (0.72) (+)

#pelvic_incidence <-> degree_spondylolisthesis (0.64) (+)

#pelvic_incidence <-> pelvic_tilt (0.63) (+)

#sacral_slope <-> pelvic_radius (-0.34) (-)

#pelvic_incidence <-> pelvic_radius (-0.25) (-)

plt.figure(figsize=(20,20))

plt.subplot(321)

x=data_3C.pelvic_incidence

y=data_3C.sacral_slope

z=np.sqrt(x**2+y**2)

plt.scatter(x, y, s=100, c=z, marker='+')

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')



plt.subplot(322)

x=data_3C.pelvic_incidence

y=data_3C.lumbar_lordosis_angle

plt.scatter(x, y, s=100, c=z, marker=(5,0))

plt.xlabel('pelvic_incidence')

plt.ylabel('lumbar_lordosis_angle')



plt.subplot(323)

x=data_3C.pelvic_incidence

y=data_3C.degree_spondylolisthesis

verts = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]])

plt.scatter(x, y, s=100, c=z, marker=verts)

plt.xlabel('pelvic_incidence')

plt.ylabel('degree_spondylolisthesis')



plt.subplot(324)

x=data_3C.pelvic_incidence

y=data_3C.pelvic_tilt

plt.scatter(x, y, s=100, c=z, marker=(5,1))

plt.xlabel('pelvic_incidence')

plt.ylabel('pelvic_tilt')



plt.subplot(325)

x=data_3C.sacral_slope

y=data_3C.pelvic_radius

plt.scatter(x, y, s=100, c=z, marker='+')

plt.xlabel('sacral_slope')

plt.ylabel('pelvic_radius')



plt.subplot(326)

x=data_3C.pelvic_incidence

y=data_3C.pelvic_radius

plt.scatter(x, y, s=100, c=z, marker=(5,2))

plt.xlabel('pelvic_incidence')

plt.ylabel('pelvic_radius')



plt.show()

#Visualization with Joint Plot

pelvic_incidence_normalize = (data_3C.pelvic_incidence) / max(data_3C.pelvic_incidence)

sacral_slope_normalize = (data_3C.sacral_slope) / max(data_3C.sacral_slope)

pelvic_radius_normalize = (data_3C.pelvic_radius) / max(data_3C.pelvic_radius)

pelvic_tilt_normalize = (data_3C.pelvic_tilt) / max(data_3C.pelvic_tilt)

degree_spondylolisthesis_normalize = (data_3C.degree_spondylolisthesis) / max(data_3C.degree_spondylolisthesis)

new_data = pd.concat([pelvic_incidence_normalize, sacral_slope_normalize, pelvic_radius_normalize, pelvic_tilt_normalize, degree_spondylolisthesis_normalize], axis=1)

new_data.sort_values('pelvic_incidence', inplace=True)



#Visualization

g = sns.jointplot(new_data.pelvic_incidence, new_data.sacral_slope, kind='reg', height=10)

plt.savefig('graph.png')

plt.show()
#Visualization with Lm Plot

sns.lmplot(x="pelvic_radius", y="sacral_slope", data=new_data, palette='pal', height=10, logistic=True)

plt.show()
#Visualization with Violin Plot

plt.subplots(figsize=(10,10))

sns.violinplot(data=new_data, palette=sns.color_palette(), inner='points', linewidth=1)

plt.xlabel("Biomechanic Features", color="blue", fontsize=15)

plt.ylabel("Normalize values of features", color="blue", fontsize=15)

plt.title("Frequency Distribution of Biomechanical Properties", color="green", fontsize=15)

plt.show()
#Swarm Plot

sns.swarmplot(x="pelvic_incidence", y="class", hue="class", data = data_3C)

plt.show()
#Pair Plot

sns.pairplot(new_data)

plt.show()
#Plotly

df = data_3C.iloc[:100,:]

import plotly.graph_objs as go

#df.pelvic_incidence = df.pelvic_incidence.astype(int)

#df.sacral_slope = df.sacral_slope.astype(int)

fig = go.Figure()

trace1 = go.Scatter(

                    x = np.linspace(1,140,200),

                    y = df.pelvic_incidence,

                    mode = "lines",

                    name = "pelvic_incidence",

                    marker = dict(color = "rgba(255, 140, 44, 1)"),

                    text = "pelvic_incidence")

#Create a Trace-2

trace2 = go.Scatter(

                    x = np.linspace(1,140,200),

                    y = df.sacral_slope,

                    mode = "lines+markers",

                    name = "sacral_slope",

                    marker = dict(color = "rgba(80, 26, 80, 0.8)"),

                    text = "sacral_slope")

#Data Concat

data = [trace1, trace2]

#Layout (Başlık, x-axis, y-axis name etc.)

layout = dict(title='Relationship between pelvic incidence and sacral slope values of patients', xaxis = dict(title = "Top 100 Features", ticklen = 5, zeroline = False))

fig = dict(data = data, layout = layout)

iplot(fig)
#Scatter Plot

import plotly.figure_factory as ff

dataframe = data_3C[data_3C['class'] == "Normal"]

new_data = dataframe.loc[:,['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope']]

new_data['index'] = np.arange(1, len(new_data)+1)

fig = ff.create_scatterplotmatrix(new_data, diag = 'box', index = 'index', colormap = 'Picnic', colormap_type = 'cat', height = 1000, width = 1000)

iplot(fig)
#Linear Regression

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = data_3C.pelvic_incidence.values.reshape(-1,1)

y = data_3C.sacral_slope.values.reshape(-1,1)



linear_reg.fit(x,y)



#Prediction

b0 = linear_reg.predict([[0]])

print("b0: ",b0)

b0_ = linear_reg.intercept_

print("b0_: ", b0_)

b1 = linear_reg.coef_

print("b1: ", b1)



#Visualization

plt.figure(figsize=(15,15))

plt.scatter(x,y)

arr = np.linspace(26,130,500).reshape(-1,1)

y_head = linear_reg.predict(arr)

plt.plot(arr, y_head, color = "red")

plt.show()
#Multiple Linear Regression

#Correlation

#pelvic_incidence <-> sacral_slope (0.81) (+)

#pelvic_incidence <-> lumbar_lordosis_angle (0.72) (+)

#pelvic_incidence <-> degree_spondylolisthesis (0.64) (+)

#pelvic_incidence <-> pelvic_tilt (0.63) (+)

#from sklearn.linear_model import LinearRegression

#dataframe = data_3C.loc[:,['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','degree_spondylolisthesis']]

#dataframe['index'] = np.arange(1, len(dataframe)+1)



#x = dataframe.iloc[:,[1,4]].values

#y = dataframe.pelvic_incidence.values.reshape(-1,1)



#multiple_linear_regression = LinearRegression()

#multiple_linear_regression.fit(x,y)
dataframe