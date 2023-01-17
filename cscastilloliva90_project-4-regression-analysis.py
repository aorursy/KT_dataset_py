import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

import folium
dataset=pd.read_csv('../input/Advertising.csv')

dataset.head()
dataset.iloc[:, 1:].describe()
plt.style.use('seaborn')

dataset[['TV','Radio', 'Newspaper']].boxplot()

plt.show()
corr_matrix=dataset.iloc[:,1:].corr()

plt.figure(figsize=(10,8))



plt.subplot2grid((1,1),(0,0))

sns.heatmap(corr_matrix, cmap='PuOr', annot=True, linewidths=3)

plt.title('Correlation Matrix including response variable', fontsize=20)

plt.show()
sns.pairplot(dataset.iloc[:,1:])

plt.show()
# Scatter plot TV and Sales

plt.figure(figsize=(20,8))

plt.tight_layout



plt.subplot(1,3,1)

plt.scatter(dataset['TV'], dataset['Sales'], alpha=0.5, color='red')

plt.title('TV and Sales', fontsize=18, fontweight='bold')



plt.subplot(1,3,2)

plt.scatter(dataset['Radio'], dataset['Sales'], alpha=0.5, color='green')

plt.title('Radio and Sales', fontsize=18, fontweight='bold')



plt.subplot(1,3,3)

plt.scatter(dataset['Newspaper'], dataset['Sales'], alpha=0.5)

plt.title('Newspaper and Sales', fontsize=18, fontweight='bold')



plt.show()
# preparation of the data to our purposes

y=dataset['Sales']

X=dataset[['TV']]

X=sm.add_constant(X)
model1=sm.regression.linear_model.OLS(y,X)

linear_regression=model1.fit()

linear_regression.summary()
coefficients=np.array(linear_regression.params)

coefficients
y_predicted=linear_regression.predict(X)
plt.figure(figsize=(15,10))



plt.subplot2grid((2,3),(0,0),colspan=2,rowspan=2)

plt.scatter(X.iloc[:,1],y, color='gray', label='Observations')

plt.plot(X.iloc[:,1], y_predicted, color='red', linewidth=3, label="y= %.2f + %.2fx" %(coefficients[0], coefficients[1]))

plt.legend(fontsize=14)

plt.title('TV vs Sales', fontsize=18, fontweight='bold')



plt.subplot2grid((2,3),(0,2),colspan=2)

plt.scatter(X.iloc[:,1],linear_regression.resid_pearson)

plt.plot(X.iloc[:,1], [3]*len(X.iloc[:,1]), color='red', linestyle='--', alpha=0.5)

plt.plot(X.iloc[:,1], [-3]*len(X.iloc[:,1]), color='red', linestyle='--',alpha=0.5)

plt.plot(X.iloc[:,1], [0]*len(X.iloc[:,1]), color='red', linestyle='--', alpha=0.5)

plt.title('Residuals', fontsize=18)



plt.subplot2grid((2,3),(1,2))

plt.hist(linear_regression.resid_pearson, alpha=0.6)

plt.title('Residuals Distributions', fontsize=18)





plt.show()
from sklearn import linear_model
X=dataset['TV']

X=np.array(X).reshape(len(X),1)

y=dataset['Sales']



linear_regression=linear_model.LinearRegression()

linear_regression=linear_regression.fit(X,y)

print(linear_regression.coef_,linear_regression.intercept_, sep='   ')
print('The first 5 predicted values are:', linear_regression.predict(X)[:5])
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

X=dataset[['TV']]

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
unstandardized_betas = w[:-1]/original_stds

unstandardized_bias=w[-1]-np.sum(original_means/original_stds*w[:-1])

print(unstandardized_betas)

print(unstandardized_bias)
import ipywidgets as widgets

from IPython.display import Javascript, display



a=widgets.Dropdown(

    options=list(dataset.columns)[1:-1],

    value=list(dataset.columns)[1:-1][0],

    description='Predictor:',

    disabled=False,

)

display(a)



def run_all(ev):

    display(Javascript('IPython.notebook.execute_cell_range(IPython.notebook.get_selected_index()+1, IPython.notebook.ncells())'))



button = widgets.Button(description="Run")

button.on_click(run_all)

display(button)
y=dataset['Sales']

X=dataset[[a.value]]

X=sm.add_constant(X)



model1=sm.regression.linear_model.OLS(y,X)

linear_regression=model1.fit()



y_predicted=linear_regression.predict(X)

coefficients=np.array(linear_regression.params)



plt.figure(figsize=(15,10))



plt.subplot2grid((2,3),(0,0),colspan=2,rowspan=2)

plt.scatter(X.iloc[:,1],y, color='gray', label='Observations')

plt.plot(X.iloc[:,1], y_predicted, color='red', linewidth=3, label="y= %.2f + %.2fx" %(coefficients[0], coefficients[1]))

plt.legend(fontsize=14)

plt.title(a.value + ' vs Sales', fontsize=18, fontweight='bold')



plt.subplot2grid((2,3),(0,2),colspan=2)

plt.scatter(X.iloc[:,1],linear_regression.resid_pearson)

plt.plot(X.iloc[:,1], [3]*len(X.iloc[:,1]), color='red', linestyle='--', alpha=0.5)

plt.plot(X.iloc[:,1], [-3]*len(X.iloc[:,1]), color='red', linestyle='--',alpha=0.5)

plt.plot(X.iloc[:,1], [0]*len(X.iloc[:,1]), color='red', linestyle='--', alpha=0.5)

plt.title('Residuals', fontsize=18)



plt.subplot2grid((2,3),(1,2))

plt.hist(linear_regression.resid_pearson, alpha=0.6)

plt.title('Residuals Distributions', fontsize=18)





plt.show()