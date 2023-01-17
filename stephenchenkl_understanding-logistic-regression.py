%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris()  #Load dataset iris
X = iris.data[:, :2]
y = (iris.target != 0) * 1
plt.figure(figsize=(4, 3))
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend()
class myLogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr  # learning rate
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        #weights initialization
        self.weight = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            #forward propagation
            z = np.dot(X, self.weight)
            h = self.__sigmoid(z)
            
            #calculate the gradient from h-y
            gradient = np.dot(X.T, (h - y)) / y.size
            
            #update weights using gradient and learning rate
            self.weight -= self.lr * gradient
            
            #update the prediction h, and calculate the loss with latest weights
            z = np.dot(X, self.weight)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
            
            #if verbose is True, display training info    
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} at {i} iteration\t')
                
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)    
        return self.__sigmoid(np.dot(X, self.weight))
    
    def predict(self, X):
        return self.predict_prob(X).round()
model = myLogisticRegression(lr=0.1, num_iter=300000, fit_intercept=False, verbose=True)
%time history = model.fit(X, y)
model.weight
preds = model.predict(X)
(preds == y).mean()
plt.figure(figsize=(10, 6))
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max()
x2_min, x2_max = X[:,1].min(), X[:,1].max()
#xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
xx1, xx2 = np.meshgrid(np.linspace(0, x1_max), np.linspace(0, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
model_2 = myLogisticRegression(lr=0.1, num_iter=300000, fit_intercept=True, verbose=True)
%time model_2.fit(X, y)
model_2.weight
preds = model_2.predict(X)
(preds == y).mean()
plt.figure(figsize=(10, 6))
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max()
x2_min, x2_max = X[:,1].min(), X[:,1].max()
#xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
xx1, xx2 = np.meshgrid(np.linspace(0, x1_max), np.linspace(0, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model_2.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
import seaborn as sns
import pandas as pd
import sys
import os
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display
%matplotlib inline

import plotly.offline as py
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split

col=['sepal_length','sepal_width','petal_length','petal_width','type']

# convert a Scikit-learn dataset to a Pandas dataset
iris=datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
y = (iris.target != 0) * 1
df['type'] = y

df_name=df.columns

df.head()
df.describe()
g = sns.pairplot(df, hue="type", palette="husl")
def plotHist(df,nameOfFeature):
    cls_train = df[nameOfFeature]
    data_array = cls_train
    hist_data = np.histogram(data_array)
    binsize = .5
    
    '''

    trace1 = go.Histogram(
        x=data_array,
        histnorm='count',
        name=nameOfFeature,
        autobinx=False,
        xbins=dict(
            start=df[nameOfFeature].min()-1,
            end=df[nameOfFeature].max()+1,
            size=binsize
        )
    )
    '''
    trace1 = go.Histogram(
        x = data_array,
        name = nameOfFeature,
        autobinx = False)

    trace_data = [trace1]
    layout = go.Layout(
        bargroupgap=0.3,
         title='The distribution of ' + nameOfFeature,
        xaxis=dict(
            title=nameOfFeature,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Number of labels',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
    fig = go.Figure(data=trace_data, layout=layout)
    py.iplot(fig)
plotHist(df,'sepal length (cm)')
plotHist(df,'sepal width (cm)')
from scipy.stats import skew
from scipy.stats import kurtosis
def plotBarCat(df,feature,target):
    
    
    
    x0 = df[df[target]==0][feature]
    x1 = df[df[target]==1][feature]

    trace1 = go.Histogram(
        x=x0,
        opacity=0.75
    )
    trace2 = go.Histogram(
        x=x1,
        opacity=0.75
    )

    data = [trace1, trace2]
    layout = go.Layout(barmode='overlay',
                      title=feature,
                       yaxis=dict(title='Count'
        ))
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='overlaid histogram')
    
    def DescribeFloatSkewKurt(df,target):
        """
            A fundamental task in many statistical analyses is to characterize
            the location and variability of a data set. A further
            characterization of the data includes skewness and kurtosis.
            Skewness is a measure of symmetry, or more precisely, the lack
            of symmetry. A distribution, or data set, is symmetric if it
            looks the same to the left and right of the center point.
            Kurtosis is a measure of whether the data are heavy-tailed
            or light-tailed relative to a normal distribution. That is,
            data sets with high kurtosis tend to have heavy tails, or
            outliers. Data sets with low kurtosis tend to have light
            tails, or lack of outliers. A uniform distribution would
            be the extreme case
        """
        print('-*-'*25)
        print("{0} mean : ".format(target), np.mean(df[target]))
        print("{0} var  : ".format(target), np.var(df[target]))
        print("{0} skew : ".format(target), skew(df[target]))
        print("{0} kurt : ".format(target), kurtosis(df[target]))
        print('-*-'*25)
    
    DescribeFloatSkewKurt(df,target)
plotBarCat(df,df_name[0],'type')
plotBarCat(df,df_name[1],'type')
def PlotPie(df, nameOfFeature):
    labels = [str(df[nameOfFeature].unique()[i]) for i in range(df[nameOfFeature].nunique())]
    values = [df[nameOfFeature].value_counts()[i] for i in range(df[nameOfFeature].nunique())]

    trace=go.Pie(labels=labels,values=values)

    py.iplot([trace])
PlotPie(df, 'type')
model = myLogisticRegression(lr=0.1, num_iter=300000, fit_intercept=True, verbose=True)
X =  df[df_name[0:4]]
Y = df[df_name[4]]
X_train, X_test, y_train, y_test =train_test_split(X,Y,
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=df['type'])
X_train.head()
y_train.head()
%time model.fit(X_train, y_train)
model.weight