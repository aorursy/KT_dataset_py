# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_boston

import seaborn as sns

from sklearn import linear_model



boston = load_boston()

dataset=pd.DataFrame(boston.data, columns=boston.feature_names)

dataset['target']=boston.target
sns.pairplot(dataset)

plt.show()
sns.pairplot(dataset[['target', 'LSTAT', 

                     'INDUS','NOX']])
g = sns.PairGrid(dataset[['target', 'LSTAT', 

                     'INDUS','NOX']])

g = g.map_upper(plt.scatter)

g = g.map_lower(sns.kdeplot, cmap="Blues_d")

g = g.map_diag(sns.kdeplot, lw=3, legend=False)
corr=dataset.corr()

corr[corr==1]=np.nan



for i in range(len(corr)):

    for j in range(len(corr)):

        if i<=j:

            corr.iloc[i,j]=np.nan

            

            

plt.figure(figsize=(20,10))



sns.heatmap(corr,linewidths=3, annot=True, cmap='coolwarm')

plt.show()
from sklearn.preprocessing import StandardScaler

X=dataset.iloc[:,1:-1]

observations = len(dataset)

variables = dataset.columns

standarddization=StandardScaler()

Xst = standarddization.fit_transform(X)

original_means=standarddization.mean_

original_stds=standarddization.var_**0.5

Xst = np.column_stack((Xst, np.ones(observations)))

y = dataset['target']
import random

def random_w( p ):

    return np.array([np.random.normal() for j in range(p)])

def hypothesis(X,w):

    return np.dot(X,w)

def loss(X,w,y):

    return hypothesis(X,w) - y

def squared_loss(X,w,y):

    return loss(X,w,y)**2

def gradient(X,w,y):

    gradients = list()

    n = float(len( y ))

    for j in range(len(w)):

        gradients.append(np.sum(loss(X,w,y) * X[:,j]) / n)

    return gradients

def update(X,w,y, alpha=0.01):

    return [t - alpha*g for t, g in zip(w, gradient(X,w,y))]

def optimize(X,y, alpha=0.01, eta = 10**-12, iterations = 1000):

    w = random_w(X.shape[1])

    path = list()

    for k in range(iterations):

        SSL = np.sum(squared_loss(X,w,y))

        new_w = update(X,w,y, alpha=alpha)

        new_SSL = np.sum(squared_loss(X,new_w,y))

        w = new_w

        if k>=5 and (new_SSL - SSL <= eta and \

        new_SSL - SSL >= -eta):

            path.append(new_SSL)

            return w, path

        if k % (iterations / 20) == 0:

            path.append(new_SSL)

    return w, path



alpha = 0.02

w, path = optimize(Xst, y, alpha, eta = 10**-12, \

iterations = 20000)

print ("These are our final standardized coefficients: " + ', '.join(map(lambda x: "%0.4f" % x, w)))
unstandardized_betas = w[:-1]/original_stds

unstandardized_bias=w[-1]-np.sum(original_means/original_stds*w[:-1])

print(unstandardized_betas)

print(unstandardized_bias)
linear_regression = linear_model.LinearRegression(normalize=False, fit_intercept=True)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

standardization = StandardScaler()

Stand_Coeff = make_pipeline(standardization, linear_regression)



linear_regression.fit(X,y)



for coef, var in sorted(zip(map(abs,linear_regression.coef_), dataset.columns), reverse=True):

    print("%6.3f %s" %(coef,var))
from sklearn.metrics import r2_score

linear_regression = linear_model.LinearRegression(normalize=False, fit_intercept=True)



def r2_est(X,y):

    return r2_score(y, linear_regression.fit(X,y).predict(X))



print('Baseline: ', r2_est(X,y))
from sklearn.preprocessing import PolynomialFeatures

createinteractions=PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)


Xi=createinteractions.fit_transform(X)

Xi=X

Xi['interaction']=X['RM']*X['LSTAT']
print('R2 of a model with RM*LSTAT interaction: %0.3f' %r2_est(Xi,y))
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score



linear_regression = linear_model.LinearRegression(normalize=False, fit_intercept=True)

create_cubic = PolynomialFeatures(degree=3, interaction_only=False)

create_highdegree = PolynomialFeatures(degree=7, interaction_only=False, include_bias=False)



linear_predictor = make_pipeline(linear_regression)

cubic_predictor=make_pipeline(create_cubic, linear_regression)
predictor = 'LSTAT'

x=dataset.LSTAT.values.reshape((len(dataset.LSTAT.values),1))



xt=np.arange(0,50,0.1).reshape((500,1))

x_range = [dataset[predictor].min(), dataset[predictor].max()] 

y_range = [dataset['target'].min(), dataset['target'].max()] 



scatter=dataset.plot(kind='scatter', x=predictor, y='target', xlim=x_range, ylim=y_range)

regression_line=scatter.plot(xt, linear_predictor.fit(x,y).predict(xt), color='red', linewidth=3)

scatter=dataset.plot(kind='scatter', x=predictor, y='target', xlim=x_range, ylim=y_range)

regression_line=scatter.plot(xt, cubic_predictor.fit(x,y).predict(xt), color='red', linewidth=3)
i=0

plt.style.use('ggplot')

fig=plt.figure(figsize=(20,15))

for d in [1,2,3,5,15]:

    i+=1

    create_pynomial = PolynomialFeatures(degree=d, interaction_only=False, include_bias=False)

    poly = make_pipeline(create_pynomial, StandardScaler(), linear_regression)

    model = poly.fit(x,y)

    print('R2 degree - %2i polynomial :%0.3f' %(d, r2_score(y, model.predict(x))))

    plt.subplot(2,3,i)

    plt.scatter(x,y, color='gray')

    plt.plot(xt, model.fit(x,y).predict(xt), linewidth=3)

    plt.xlim(x_range)

    plt.ylim(y_range)

    plt.title('R2 degree - %2i polynomial :%0.3f' %(d, r2_score(y, model.predict(x))))

    

plt.show()