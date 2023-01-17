import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import ipywidgets as widgets
# reading the dataset

dataset = pd.read_csv('../input/Advertising.csv')

dataset.head()
columns=dataset.columns

columns
dataset=dataset[['TV', 'Radio', 'Newspaper', 'Sales']]

dataset.head()
dataset.describe()
import matplotlib.pyplot as plt

import seaborn as sns



sns.pairplot(dataset , plot_kws = {'alpha': 0.7, 's': 40, 'edgecolor': 'k', 'color':'red'})
corr=dataset.corr()

corr[corr==1]=np.nan



sns.heatmap(corr, cmap='coolwarm', linewidths=2, annot=True)
import statsmodels.formula.api as smf



linear_regression_smf = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=dataset)

fitted_model_smf = linear_regression_smf.fit()

fitted_model_smf.summary()
from sklearn.linear_model import LinearRegression





linear_regression_skl = LinearRegression()

linear_regression_skl.fit(dataset[['Radio', 'Newspaper', 'TV']], dataset['Sales'])

print(linear_regression_skl.coef_)

print(linear_regression_skl.intercept_)
# defining the gradient descent model

import random



def random_w(p):

    return np.array([np.random.normal() for j in range(p)])



def hypothesis(X,w):

    return np.dot(X,w)



def loss(X,w,y):

    return hypothesis(X,w) -y



def squared_loss(X,w,y):

    return loss(X,w,y)**2



def gradient(X,w,y):

    gradients= list()

    n=float(len(y))

    for j in range(len(w)):

        gradients.append(np.sum(loss(X,w,y)*X[:,j])/n)

    return gradients



def update(X,w,y, alpha = 0.001):

    return [t - alpha*g for t,g in zip(w,gradient(X,w,y))]



def optimize(X, y, alpha = 0.001, eta = 10**-12, iterations=1000):

    w=random_w(X.shape[1])

    path = list()

    for k in range(iterations):

        SSL=np.sum(squared_loss(X,w,y))

        new_w=update(X,w,y, alpha=alpha)

        new_SSL=np.sum(squared_loss(X,new_w,y))

        w=new_w

        if k >= 5 and (new_SSL - SSL <= eta and new_SSL-SSL > -eta):

            path.append(new_SSL)

            return w, path

        if k % (iterations / 20)==0:

            path.append(new_SSL)

    return w, path
#preparing the variables and standardizing it

from sklearn.preprocessing import StandardScaler



X=dataset[['Radio', 'Newspaper', 'TV']]

observations = len(dataset)

standarddization=StandardScaler()

Xst = standarddization.fit_transform(X)

original_means=standarddization.mean_

original_stds=standarddization.var_**0.5

Xst = np.column_stack((Xst, np.ones(observations)))

y = dataset['Sales']
#using the gradient descent

alpha = 0.02

w, path = optimize(Xst, y, alpha, eta = 10**-12, \

iterations = 20000)

print ("These are our final standardized coefficients: " + ', '.join(map(lambda x: "%0.4f" % x, w)))
#unstandardizing the coefficients

unstandardized_betas = w[:-1]/original_stds

unstandardized_bias=w[-1]-np.sum(original_means/original_stds*w[:-1])

print(unstandardized_betas)

print(unstandardized_bias)