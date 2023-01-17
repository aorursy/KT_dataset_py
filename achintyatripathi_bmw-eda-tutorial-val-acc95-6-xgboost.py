# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# FOR EDA --------------- # 
import matplotlib.pyplot as plt 
%matplotlib inline 
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
# ----------------------- #

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/bmw.csv')
display(data.info(), data.head())
def target_count(data,column):
    trace = go.Bar( x = data[column].value_counts().values.tolist(),
    y = data[column].unique(),
    orientation = 'h',
    text = data[column].value_counts().values.tolist(),
    textfont=dict(size=20),
    textposition = 'auto',
    opacity = 0.5,marker=dict(colorsrc='tealrose',
            line=dict(color='#000000',width=1.5))
    )
    layout = (dict(title= "EDA of {} column".format(column),
                  autosize=True,height=800,))
    fig = dict(data = [trace], layout=layout)
    
    py.iplot(fig)

# --------------- donut chart to show there percentage -------------------- # 

def target_pie(data,column):
    trace = go.Pie(labels=data[column].unique(),values=data[column].value_counts(),
                  textfont=dict(size=15),
                   opacity = 0.5,marker=dict(
                   colorssrc='tealrose',line=dict(color='#000000', width=1.5)),
                   hole=0.6)
                  
    layout = dict(title="Dounat chart to see %age of individual elements")
    fig = dict(data=[trace],layout=layout)
    py.iplot(fig)
# Model 

target_count(data,'model')
target_pie(data,'model')

# Transmition

target_count(data,'transmission')
target_pie(data,'transmission')
# fuelType

target_count(data,'fuelType')
target_pie(data,'fuelType')
#df['model','transmission','fuelType'] = data['model','transmission','fuelType']
#for feat in ['model','transmission','fuelType']:

from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
## model_LE
data["LE_model"] = lb_make.fit_transform(data["model"])

## model_LE
data["LE_transmission"] = lb_make.fit_transform(data["transmission"])

## model_LE
data["LE_fuelType"] = lb_make.fit_transform(data["fuelType"])

## results
data[["model","LE_model","transmission","LE_transmission","fuelType","LE_fuelType"]].head(11)
print(sum(data['engineSize'] == 0))
print(sum(data['tax'] == 0))
# We have simply replaced 0 values with null.
data[["engineSize","tax"]] = data[["engineSize","tax"]].replace(0,np.NaN)
data.isnull().sum()
def find_median(var):
    temp = data[data[var].notnull()]
    temp = data[[var,'model']].groupby('model')[[var]].median().reset_index()
    return temp
# model filling

find_median('tax')
data = data.dropna()
data = data.reset_index(drop=True)
display(data.info)
def correlation_plot():
    #correlation
    correlation = data.corr()
    #label 
    matrix_cols = correlation.columns.tolist()
    #convert to array as it can't take values directly. 
    corr_array = np.array(correlation)
    trace = go.Heatmap(z = corr_array,
                      x=matrix_cols,
                      y=matrix_cols,
                      colorscale='Viridis',
                      colorbar = dict()
                      )
    layout = go.Layout(dict(title='Correlation Matrix for variables provided.',
                          margin = dict(r=0,l=100,
                                       t = 0, b =100,),
                          yaxis = dict(tickfont = dict(size = 9)),
                          xaxis = dict(tickfont = dict(size = 9)),
                          )
                      )
    fig = go.Figure(data = [trace], layout = layout)
    py.iplot(fig)
## So let's start  by finding the correlation between the columns. 

correlation_plot()

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

reg = linear_model.LinearRegression()
X = data[['LE_model', 'LE_transmission', 'LE_fuelType', 'engineSize','year', 'tax','mileage','mpg']]
Y = data['price']
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.2)
reg.fit(train_X,train_Y)
print('Performance Score(GB): %.1f ' %(reg.score(test_X,test_Y)*100))
from sklearn.ensemble import GradientBoostingRegressor
GB=GradientBoostingRegressor(random_state=0)
GB.fit(train_X,train_Y)
print('Performance Score(GB): %.1f ' %(GB.score(test_X,test_Y)*100))
from xgboost import XGBRegressor
XGB=XGBRegressor(random_state=0)
XGB.fit(train_X,train_Y)
print('Performance score(XGB): %.1f ' %(XGB.score(test_X,test_Y)*100))
