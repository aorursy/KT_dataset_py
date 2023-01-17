# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import cufflinks as cf

import sklearn

from sklearn import svm, preprocessing 



import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/diamonds.csv')

df.info()
df.head()

df = df.drop('Unnamed: 0', axis = 1)

df.head()
df['clarity'].unique()
df.groupby('cut').count()['carat'].plot.bar()
cut_dict = {'Fair' : 1, 'Good' : 2, 'Very Good' : 3, 'Premium' : 4, 'Ideal' : 5}

clarity_dict ={ 'I1' : 1, 'SI2' : 2, 'SI1' : 3, 'VS2' : 4, 'VS1' : 5, 'VVS2' : 6, 'VVS1' : 7 , 'IF' : 8}

color_dict = {'D':7, 'E':6, 'F':5, 'G':4, 'H':3, 'I':2, 'J':1}
df['cut'] = df['cut'].map(cut_dict)

df['clarity'] = df['clarity'].map(clarity_dict)

df['color'] = df['color'].map(color_dict)
df.head()
df.isnull().any()
df = sklearn.utils.shuffle(df)

X = df.drop(['price'], axis = 1).values

X = preprocessing.scale(X)

y = df['price'].values

y = preprocessing.scale(y)
test_size = 200

X_train = X[: -test_size]

y_train = y[: -test_size]

X_test = X[-test_size :]

y_test =  y[-test_size :]
clf = svm.SVR(kernel = 'linear')

clf.fit(X_train, y_train)
clf.score(X_test, y_test)
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