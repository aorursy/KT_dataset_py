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
import pandas as pd

df=pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
df.describe()
df.fillna(df.mean())
df=df.drop(['CustomerID'],axis=1)

import numpy as np

data=np.asanyarray(df)

data
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

le.fit(['Female','Male'])

data[:,0]=le.transform(data[:,0])

data

import matplotlib.pyplot as plt

j=1

for i in ['Gender','Age','Annual Income (k$)']:

    plt.figure(1 , figsize = (15 , 7))

    plt.subplot(3,3,j)

    plt.scatter(data[:,j-1],data[:,3])

    plt.xlabel(i)

    plt.ylabel('Spending Score (1-100)')

    plt.show()

    j=j+1

from sklearn.cluster import KMeans

km=KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300,tol=0.0001,  random_state= 111  , algorithm='elkan')

km.fit(data)
import seaborn as sns 

import plotly.graph_objs as go

import plotly as py

df['label'] =  km.labels_

trace = go.Scatter3d(

    x= df['Age'],

    y= df['Spending Score (1-100)'],

    z= df['Annual Income (k$)'],

    mode='markers',

     marker=dict(

        color = df['label'], 

        size= 50,

        line=dict(

            color= df['label'],

            width= 20

        ),

        opacity=0.7

     )

)

arr = [trace]

layout = go.Layout(

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'Spending Score'),

            zaxis = dict(title  = 'Annual Income')

        )

)

fig = go.Figure(data=arr, layout=layout)

py.offline.iplot(fig)