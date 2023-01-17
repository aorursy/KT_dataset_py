import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pandas as pd

import cufflinks as cf

import sklearn

from sklearn import svm, preprocessing 

import seaborn as sns

import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import os
df = pd.read_csv('../input/diamonds.csv')

df.head()
sns.FacetGrid(df, hue = 'cut', height = 6).map(sns.distplot, 'price').add_legend()

plt.plot()
cut_dict = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Premium' : 4, 'Ideal' : 5}

clarity_dict ={ 'I1' : 1, 'SI2' : 2, 'SI1' : 3, 'VS2' : 4, 'VS1' : 5, 'VVS2' : 6, 'VVS1' : 7 , 'IF' : 8}

color_dict = {'D':7, 'E':6, 'F':5, 'G':4, 'H':3, 'I':2, 'J':1}
df['cut'] = df['cut'].map(cut_dict)

df['clarity'] = df['clarity'].map(clarity_dict)

df['color'] = df['color'].map(color_dict)
df = df.drop('Unnamed: 0', axis = 1)

df.head()
# sns.pairplot(df[['carat','price', 'cut', 'color', 'clarity', 'depth']], hue = 'cut', height = 3)
df.isnull().any()
df = sklearn.utils.shuffle(df, random_state = 42)

X = df.drop(['price'], axis = 1).values

X = preprocessing.scale(X)

y = df['price'].values

y = preprocessing.scale(y)
test_size = 200

X_train = X[: -test_size]

y_train = y[: -test_size]

X_test = X[-test_size :]

y_test =  y[-test_size :]
from sklearn.neighbors import KNeighborsRegressor

score = []

for k in range(1,20):   # running for different K values to know which yields the max accuracy. 

    clf = KNeighborsRegressor(n_neighbors = k,  weights = 'distance', p=1)

    clf.fit(X_train, y_train)

    score.append(clf.score(X_test, y_test ))    
trace0 = go.Scatter(

    y = score,

    x = np.arange(1,len(score)+1), 

    mode = 'lines+markers', 

    marker = dict(

        color = 'rgb(150, 10, 10)'

    )

)

layout = go.Layout(

    title = '', 

    xaxis = dict(

        title = 'K value', 

        tickmode = 'linear'

    ),

    yaxis = dict(

        title = 'Score',

#         range = [0, 10000]

    )

)

fig = go.Figure(data = [trace0], layout = layout)

iplot(fig, filename='basic-line')
k_max = score.index(max(score))+1

print( "At K = {}, Max Accuracy = {}".format(k_max, max(score)*100))
clf = KNeighborsRegressor(n_neighbors = k_max,  weights = 'distance', p=1)

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test ))   

y_pred = clf.predict(X_test)

trace0 = go.Scatter(

    y = y_test,

    x = np.arange(200), 

    mode = 'lines', 

    name = 'Actual Price',

    marker = dict(

    color = 'rgb(10, 150, 50)')

)



trace1 = go.Scatter(

    y = y_pred,

    x = np.arange(200), 

    mode = 'lines', 

    name = 'Predicted Price',

    line = dict(

        color = 'rgb(110, 50, 140)',

        dash = 'dot'

    )

)





layout = go.Layout(

    xaxis = dict(title = 'Index'), 

    yaxis = dict(title = 'Normalized Price')

)



figure = go.Figure(data = [trace0, trace1], layout = layout)

iplot(figure)