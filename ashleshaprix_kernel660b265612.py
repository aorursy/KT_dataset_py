# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False)

print(os.listdir("/kaggle/input"))

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()



from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import xgboost



df=pd.read_csv("/kaggle/input/daily_milk_rate.csv")
print("dataset contains {} rows and {} columns".format(df.shape[0],df.shape[1]))
df.info()
df.head()
df= pd.read_csv('/kaggle/input/daily_milk_rate.csv')

print("Data shape: ", df.shape)

print("Duplicates: ", df.duplicated().sum())

print()

print(df.isnull().sum())

print()

df.head()


pd.read_csv('/kaggle/input/daily_milk_rate.csv', header=None)

print(df.describe())
pd.read_csv('/kaggle/input/daily_milk_rate.csv', header=None)

# print the first 20 rows of data

print(df.head(10))
np.random.seed(0) 
df.sample(5)
missing_values_count = df.isnull().sum()
missing_values_count[0:10]
total_cells = np.product(df.shape)

total_missing = missing_values_count.sum()



# percent of data that is missing

(total_missing/total_cells) * 100
missing_values_count[0:10]

df.dropna()
columns_with_na_dropped = df.dropna(axis=1)

columns_with_na_dropped.head()
print("Columns in original dataset: %d \n" % df.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])

df.fillna(0)
df.fillna(method = 'bfill', axis=0).fillna(0)
plt.figure(figsize=(10,7))

chains=df['Price'].value_counts()[:10]

sns.barplot(x=chains,y=chains.index,palette='deep')

plt.title("Milk Price in different state")

x=df['Price'].value_counts()

colors = ['#FEBFB3', '#E1396C']



trace=go.Pie(labels=x.index,values=x,textinfo="value",

            marker=dict(colors=colors, 

                           line=dict(color='#000000', width=4)))

layout=go.Layout(title="Accepting vs not accepting online orders",width=500,height=500)

fig=go.Figure(data=[trace],layout=layout)

py.iplot(fig, filename='pie_chart_subplots')

    

    