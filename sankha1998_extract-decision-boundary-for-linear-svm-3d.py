# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

s=StandardScaler()

import plotly.graph_objects as go

from sklearn.svm import LinearSVC

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
df.shape
df.head()
df['species'].value_counts()
df['species'].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2},inplace=True)
df['species'].value_counts()
def plot(x,y,cvar,Scale):

  if Scale == 'Yes': ## if scaling is needed or not

    x=s.fit_transform(x)

  else:

    pass

 

  model = LinearSVC(C=cvar, loss='hinge') ## Linear SVM Model with hyper parameter tuning

  clf = model.fit(x, y)

  Z = lambda X,Y: (-clf.intercept_[0]-clf.coef_[0][0]*X-clf.coef_[0][1]*Y) / clf.coef_[0][2] 

    # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.

    # Solve for w3 (z)

    

  trace1 = go.Mesh3d(x = x[:,0], y = x[:,1], z = Z(x[:,0],x[:,1])) ## for separating plane

  trace2 = go.Scatter3d(x=x[:,0], y=x[:,1], z=x[:,2], mode='markers',

                        marker = dict(size = 12,color = y,colorscale = 'Viridis')) ## for vector plots

  data=[trace1,trace2]

  fig = go.Figure(data=data,layout={})

  fig.show()
p = df.iloc[:,[2,3,0]].values

q = (df.species!=0).astype(np.float64)
plot(p,q,0.1,'No')
plot(p,q,0.1,'Yes')
m = df.iloc[:,[2,3,0]].values

n = (df.species!=1).astype(np.float64)
plot(m,n,10,'No') ## Without Scaling
plot(m,n,10,'Yes') ## With Scaling
a=df.iloc[:,[2,3,0]].values

b = (df.species!=0).astype(np.float64)

plot(a,b,0.01,'Yes')
plot(a,b,100,'Yes')