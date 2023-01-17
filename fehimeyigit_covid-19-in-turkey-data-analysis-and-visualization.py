# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
# read csv (comma separated value) into data
data = pd.read_csv('../input/covid19-in-turkey/covid_19_data_tr.csv')
print(plt.style.available) # look at available plot styles
plt.style.use('ggplot')
# read csv (comma separated value) into data
data1 = pd.read_csv('../input/covid19-in-turkey/time_series_covid_19_confirmed_tr.csv')
print(plt.style.available) # look at available plot styles
plt.style.use('ggplot')
data
# Well know question is is there any NaN value and length of this data so lets look at info
data.info()
data.describe()

plt.figure(figsize=(10,10))
sns.barplot(x=data['Last_Update'], y=data['Confirmed'])
plt.xticks(rotation= 90)
plt.xlabel('Days')
plt.ylabel('Confirmed')

# visualize
f,ax1 = plt.subplots(figsize =(15,15))
sns.pointplot(x=data.Last_Update,y=data.Deaths,color='green',alpha=0.5)
sns.pointplot(x=data.Last_Update,y=data.Confirmed,color='red',alpha=0.5)
plt.text(0,3500,'Confirmed',color='red',fontsize =10,style = 'italic')
plt.text(0,1000,' Deaths',color='green',fontsize = 10,style = 'italic')
plt.xlabel('Last_Update',fontsize = 10,color='black')
plt.ylabel('Values',fontsize = 10,color='black')
plt.xticks(rotation= 90)
plt.title('Deaths  VS   Confirmed',fontsize = 10,color='black')
plt.grid()
sns.lmplot(x="Deaths", y='Confirmed', data=data)
plt.show()
#cubehelix plot
sns.kdeplot(data.Deaths,data.Confirmed, shade=True, cut=1)
plt.show()
data.corr()
#correlation map
data = data.drop(columns="Province/State")

f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
# pair plot
sns.pairplot(data)
plt.show()
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=data.Last_Update,
    y=data.Deaths,
    z=data.Confirmed,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)