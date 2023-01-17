# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# seaborn library
import seaborn as sns

# plotly library
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column_2C_weka.csv')
data.head()
data.info()
data['class'].value_counts()
sns.countplot(x="class", data=data)
#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f',ax=ax)
plt.show()
data.columns
sns.boxplot(x="class", y="pelvic_incidence", data=data, palette="PRGn")
plt.show()
sns.boxplot(x="class", y="sacral_slope", data=data, palette="PRGn")
plt.show()
data.head()
data['coloring'] = ['red' if i=='Abnormal' else 'green' for i in data['class']]
data.head()
data.columns
sns.swarmplot(x="class", y="pelvic_incidence", data=data)
plt.show()
sns.swarmplot(x="class", y="sacral_slope", data=data)
plt.show()
# import figure factory
import plotly.figure_factory as ff

data_matrix = data.loc[:,['pelvic_incidence', 'lumbar_lordosis_angle', 'sacral_slope', 'degree_spondylolisthesis']]
data_matrix["index"] = np.arange(1,len(data_matrix)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data_matrix, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 100,
                                       marker = 'points',
                                       edgecolor= "black")
plt.show()
data_abnormal = data[data['class']=='Abnormal']
pd.plotting.scatter_matrix(data_abnormal.loc[:, data_abnormal.columns != 'class'],
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 100,
                                       marker = 'points',
                                       edgecolor= "black")
plt.show()
data_normal = data[data['class']=='Normal']
pd.plotting.scatter_matrix(data_normal.loc[:, data_normal.columns != 'class'],
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 100,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()
data.columns
# x and y axis to be used
data_abnormal = data[data['class']=='Abnormal']
x_abnormal = data_abnormal['pelvic_incidence'].values.reshape(-1,1)
y_abnormal = data_abnormal['sacral_slope'].values.reshape(-1,1)

plt.figure(figsize=(10,10))
plt.scatter(x_abnormal,y_abnormal)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_abnormal, y_abnormal)

# prediction
x_predict = np.linspace(min(x_abnormal), max(y_abnormal)).reshape(-1,1)
y_predict = lin_reg.predict(x_predict)

# visualization
plt.figure(figsize=(10,10))
plt.plot(x_predict, y_predict, color='red', linewidth=3)
plt.scatter(x_abnormal,y_abnormal)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
print('r square score: ', lin_reg.score(x_abnormal,y_abnormal))
# R square using different approach
y_predicted_abnormal = lin_reg.predict(x_abnormal)

from sklearn.metrics import r2_score
print('r square score: ', r2_score(y_abnormal, y_predicted_abnormal))
data.columns
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=2) # 3rd degree polynomial
# let y=b0+b1.x+b2.x^2+b3.x^3
# the code below returns x^0, x^1, x^2 values
x_polynomial = pol_reg.fit_transform(x_abnormal) # y=b0+b1.x+b2.x^2+b3.x^3
x_polynomial
# so that knowing these values we can assume that we have linear regression again
from sklearn.linear_model import LinearRegression
lin_reg2 = LinearRegression()
lin_reg2.fit(x_polynomial, y_abnormal)
y_pol_predict = lin_reg2.predict(x_polynomial)
# visualization
plt.figure(figsize=(10,10))
plt.plot(x_abnormal, y_pol_predict, color='black', linewidth=3, label='polynomial')
plt.legend()
plt.scatter(x_abnormal,y_abnormal)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
data.columns
# x and y axis to be used
data_normal = data[data['class']=='Normal']
x_normal = data_normal['pelvic_tilt numeric'].values.reshape(-1,1)
y_normal = data_normal['sacral_slope'].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_normal, y_normal)

# prediction
x_new = np.linspace(min(x_normal), max(x_normal)).reshape(-1,1)
y_new_predict = tree_reg.predict(x_new)

# visualization
plt.figure(figsize=(10,10))
plt.plot(x_new, y_new_predict, color='red', linewidth=3)
plt.scatter(x_normal,y_normal)
plt.xlabel('pelvic_tilt numeric')
plt.ylabel('sacral_slope')
plt.show()
tree_reg.predict(25) # predicted sacral_slope value for pelvic_tilt numeric=25
tree_reg.predict(250) # predicted sacral_slope value for pelvic_tilt numeric=250
# x and y axis to be used
data_normal = data[data['class']=='Normal']
x_normal = data_normal['pelvic_tilt numeric'].values.reshape(-1,1)
y_normal = data_normal['sacral_slope'].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
ran_for = RandomForestRegressor(n_estimators=100, random_state=42)
ran_for.fit(x_normal, y_normal)

# prediction
x_rf = np.linspace(min(x_normal), max(x_normal)).reshape(-1,1)
y_rf_predict = ran_for.predict(x_rf)

# visualization
plt.figure(figsize=(10,10))
plt.plot(x_rf, y_rf_predict, color='red', linewidth=3)
plt.scatter(x_normal,y_normal)
plt.xlabel('pelvic_tilt numeric')
plt.ylabel('sacral_slope')
plt.show()
print('r square score: ', ran_for.score(x_normal, y_normal))
y_normal_head = ran_for.predict(x_normal)
from sklearn.metrics import r2_score
print('r square score: ', r2_score(y_normal,y_normal_head))