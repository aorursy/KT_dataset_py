# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

s=StandardScaler()

import plotly.graph_objects as go

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import make_gaussian_quantiles# Construct dataset

X1, y1 = make_gaussian_quantiles(cov=1.,

                                 n_samples=1000, n_features=2,

                                 n_classes=2, random_state=1)

x1 = pd.DataFrame(X1,columns=['x','y'])

y1 = pd.Series(y1)

x1=x1.values
trace = go.Scatter(x=x1[:,0],y=x1[:,1],mode='markers',marker = dict(size = 12,color = y1,colorscale = 'Viridis'))

data=[trace]



layout = go.Layout()

fig = go.Figure(data=data,layout=layout)

fig.show()
r = np.exp(-(x1 ** 2).sum(1)* 0.3)    ## exp(-gamma|x1-x2|**2) here gamma 0.3
trace1 = go.Scatter3d(x=x1[:,0], y=x1[:,1],z=r,mode='markers',marker = dict(size = 3,color = y1,colorscale = 'Viridis')) 

data=[trace1]

fig = go.Figure(data=data,layout={})

fig.show()
x1 = np.insert(x1,2,r,axis=1)
model = LinearSVC(C=1.0, loss='hinge')

clf = model.fit(x1, y1)



Z = lambda X,Y: (-clf.intercept_[0]-clf.coef_[0][0]*X-clf.coef_[0][1]*Y) / clf.coef_[0][2]

# The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.

# Solve for w3 (z)



trace1 = go.Mesh3d(x = x1[:,0], y = x1[:,1], z = Z(x1[:,0],x1[:,1])) ## for separating plane

trace2 = go.Scatter3d(x=x1[:,0], y=x1[:,1],z=x1[:,2],mode='markers',marker = dict(size = 3,color = y1,colorscale = 'Viridis')) ## for vector plots

data=[trace1,trace2]

fig = go.Figure(data=data,layout={})

fig.show()