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
nRowsRead = 1000 # specify 'None' if want to read whole file

df1 = pd.read_csv('../input/cusersmarildownloadsdeathscsv/deaths.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df1.dataframeName = 'deaths.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
categorical_cols = [cname for cname in df1.columns if

                    df1[cname].nunique() < 10 and 

                    df1[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df1.columns if 

                df1[cname].dtype in ['int64', 'float64']]
print(categorical_cols)
print(numerical_cols)
# for data visualzation

import seaborn as sns

import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(19.7,8.27)})



sns.heatmap(df1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Fields not needed to our problem

to_drop = ["Unnamed: 1","Unnamed: 2"]

    

# Drop selected fields in place

df1.drop(to_drop, inplace=True, axis=1)
df1.dropna(inplace=True)

df1.shape
sns.distplot(df1["lower"])
sns.scatterplot(x='lower',y='median.1',data=df1)
sns.countplot(df1["upper.1"])
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf1 = df1.groupby('lower').size()/df1['lower'].count()*100

labels = lowerdf1.index

values = lowerdf1.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
#catplot room type and price

plt.figure(figsize=(10,6))

sns.catplot(x="lower", y="upper.1", data=df1);

plt.ioff()