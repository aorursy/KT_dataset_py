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
# reading csv



df = pd.read_csv('../input/wall-art-sales/Wall Art sales - Sheet1.csv')

df
df.info()
df.shape
# checking null values



print(df.isnull().sum())
df = df.dropna()
# droping duplicate rows 



df = df.drop_duplicates()

df.shape
# renaming the column



df = df.rename(columns = {'Link':'Product'})

df
df.dtypes
import re



df['Shipping'] = df['Shipping'].apply( lambda x : int(re.findall( '\d+' , x )[0]) )

df['Discount'] = df['Discount'].apply(lambda x : int(re.findall('\d+' , x )[0]))

df['Price'] = df['Price'].apply(lambda x : int(re.sub(',','',x)[1:]))

df['Brand'] = df['Brand'].apply( lambda x: re.sub('by' , '' , x))

df
# 3D Scatter Plot



import plotly.express as px



fig = px.scatter_3d(df, x='Discount', y='Price', z='Shipping', color='Price')

fig.show()
import plotly.express as px



fig = px.scatter_3d(df, x='Price', y='Shipping', z='Discount', color='Discount')

fig.show()
import numpy as np 

import plotly 

import plotly.graph_objects as go 

import plotly.offline as pyo 

from plotly.offline import init_notebook_mode 

  

init_notebook_mode(connected=True) 

  

# generating 150 random integers 

# from 1 to 50 

x = list(df['Brand'])

  



y = list(df['Price'])

  

# plotting scatter plot 

fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict( 

        color=np.random.randn(20), 

        colorscale='Viridis',  

        showscale=True

    ) )) 

  

fig.show() 
import numpy as np 

import plotly 

import plotly.graph_objects as go 

import plotly.offline as pyo 

from plotly.offline import init_notebook_mode 

  

init_notebook_mode(connected=True) 

  

# generating 150 random integers 

# from 1 to 50 

x = list(df['Price'])

  



y = list(df['Shipping'])

  

# plotting scatter plot 

fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict(size=[40, 60, 80, 100],

                color=[0, 1, 2, 3])

    ) )

  

fig.show() 
import plotly.graph_objects as go



fig = go.Figure(data=go.Scatter(

   x = list(df['Price']),

y = list(df['Discount']),

    mode='lines',

))



fig.show()
fig = go.Figure(data=go.Scatter(x=df['Brand'],

                                y=df['Price'],

                                mode='markers',

                                marker_color=df['Price'],

                                text=df['Brand'])) # hover text goes here



fig.update_layout(title='Brand vs Price')

fig.show()
fig = go.Figure(data=go.Scatter(x=df['Brand'],

                                y=df['Discount'],

                                mode='markers',

                                marker_color=df['Discount'],

                                text=df['Brand'])) # hover text goes here



fig.update_layout(title='Brand vs Discount')

fig.show()
# heat map



import matplotlib.pyplot as plt

import seaborn as sns

correlation = df.corr()

plt.figure(figsize = (12 , 12))

sns.heatmap(correlation)