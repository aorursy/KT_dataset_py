# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly
from plotly.graph_objects import Figure,Layout,Scatter
import plotly.graph_objects as go
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")
dataset.info()
p=len(dataset)-dataset["Province/State"].count()
p=100*p/len(dataset)
print("Missing data from the column is ",p.round(),"%")
dataset=dataset.drop(["Province/State"],axis=1)
dataset.describe()
columns=["Country/Region","Confirmed","Recovered","Deaths","Date"]
df=dataset[columns].sort_values(by="Confirmed",ascending=False).groupby("Country/Region")
Country_label=list(df.first().sort_values(by="Confirmed",ascending=False).index)[:10]
Confirmed=list(df.first().sort_values(by="Confirmed",ascending=False).Confirmed)[:10]

target=go.Pie(labels=Country_label,values=Confirmed,hole=0.4)
layout=go.Layout(title="Top 10 Countries")
fig=go.Figure(data=target,layout=layout)
fig.show()

Recovered=list(df.first().sort_values(by="Confirmed",ascending=False).Recovered)[:10]
Deaths=list(df.first().sort_values(by="Confirmed",ascending=False).Deaths)[:10]
target1=go.Bar(x=Country_label,y=Recovered,name="Recovered",marker_color="#ADFF96")
target2=go.Bar(x=Country_label,y=Deaths,name="Deaths",marker_color="#26034E")
layout=go.Layout(title="Recovery and Deaths")
fig=go.Figure(data=[target1,target2],layout=layout)
fig.show()
dataset_india=dataset[columns][dataset["Country/Region"]=="India"]
dataset_india=dataset_india.drop(["Country/Region"],axis=1)

dataset_india["Date"]=pd.to_datetime(dataset_india.Date)
dataset_india=dataset_india[dataset_india["Confirmed"]>0]
dataset_india
target1=go.Scatter(x=dataset_india.Date,y=dataset_india["Confirmed"],name="Confirmed",line_color="dimgray",opacity=0.8)
target2=go.Scatter(x=dataset_india.Date,y=dataset_india["Recovered"],name="Recovered",line_color="deepskyblue",opacity=0.8)
target3=go.Scatter(x=dataset_india.Date,y=dataset_india["Deaths"],name="Deaths",line_color="red",opacity=0.8)
layout=go.Layout(title="Coronavirus in India")
fig=go.Figure(data=[target1,target2,target3],layout=layout)
fig.show()

