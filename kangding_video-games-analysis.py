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
data = pd.read_csv('../input/videogamesales/vgsales.csv')
data.head()
data.describe()
data.info()
data['Year'].fillna(value = 0,axis = 0,inplace = True)
data.dropna(how = 'any',subset = ['Publisher'],inplace = True)
data['Year'] = data['Year'].apply(lambda x: int(x))
import plotly as py
import plotly .graph_objs as go
from plotly.graph_objs import Scatter
py.offline.init_notebook_mode()
Popular_Genre = data.groupby('Genre')['Genre'].count().sort_values(ascending = False)

trace_basic = [go.Bar(x = Popular_Genre.index.tolist(),
                     y = Popular_Genre.values.tolist(),
                     marker = dict(color = ['red','orange','pink','blue','lightblue',
                                           'indianred','darkblue','steelblue','green',
                                         'darkgray','yellow','aqua']),opacity = 0.7)]

layout = go.Layout(title = 'Popularity of each genre',
                  xaxis = {'title':'Genre'},
                  )
figure_basic = go.Figure(data = trace_basic,layout = layout)
figure_basic.show()
#Popularity of Each Genre：
Popular_Genre = pd.DataFrame(data = Popular_Genre)
Popular_Genre = Popular_Genre.rename(columns = {'Genre':'Quantity'}).reset_index()
Popular_Genre['Percent'] = Popular_Genre['Quantity'] / Popular_Genre['Quantity'].sum() * 100

trace_basic = [go.Pie(labels = Popular_Genre['Genre'],
                    values = Popular_Genre['Percent'],
                    hole = 0.3, textfont = dict(size = 12,color = 'white'))]

layout = go.Layout(title = 'Percent of each genre',
                  xaxis = {'title':'Genre'},
                  )
figure_basic = go.Figure(data = trace_basic,layout = layout)
figure_basic.show()
import matplotlib.pyplot as plt
import seaborn as sns
grouped_platform = data.groupby('Platform')['Platform'].count().sort_values(ascending = False)
grouped_platform = pd.DataFrame(data = grouped_platform).rename(columns = {'Platform' : 'Quantity'})

plt.figure(figsize = (12,8),dpi = 300)
sns.barplot(x = grouped_platform.index,
            y = grouped_platform['Quantity'],
            data = grouped_platform)
plt.xlabel('Platform',labelpad = 20,fontsize = 15)
plt.ylabel('Quantity',labelpad = 20,fontsize = 15)
plt.xticks(rotation = 45,fontsize = 10)
plt.show()
#This is another way to plot：sns.countplot（）
plt.figure(figsize = (12,8),dpi = 300)
sns.countplot(x = 'Platform',data = data)
plt.xlabel('Platform',labelpad = 20,fontsize = 15)
plt.ylabel('Quantity',labelpad = 20,fontsize = 15)
plt.xticks(rotation = 45,fontsize = 10)
plt.show()
grouped_publisher = data.groupby('Publisher')[['Publisher','Global_Sales']].sum()
grouped_publisher = grouped_publisher.reset_index().sort_values(by = 'Global_Sales',ascending = False)
grouped_publisher_top20 = grouped_publisher[0:20]

plt.figure(figsize = (12,8),dpi = 300)
sns.barplot(x = 'Publisher',
            y = 'Global_Sales',
             data = grouped_publisher_top20,
           palette="Spectral_r")
plt.xlabel('Publisher',labelpad = 20,fontsize = 15)
plt.ylabel('Revenue',labelpad = 20,fontsize = 15)
plt.xticks(rotation = 90,fontsize = 12)
plt.show()
