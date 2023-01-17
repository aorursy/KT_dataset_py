# Import libraries

import pandas as pd

import numpy as np

import random

from numpy import linalg

from sklearn.linear_model import LinearRegression
# Generate random numbers

data = pd.DataFrame(np.random.randint(low=-3, high=3, size=(3, 3)))



data.columns = ['num 1', 'num 2', 'num 3']

data.head()
# Get data-frame values into numpy array

num_array = data.values

num_array
# We can call linalg.norm function to compute norm

# ord=1 represents L(p=1) norm

linalg.norm(num_array, ord=1, axis=1)
# Import few more necessary libraries

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams

import seaborn as sns

%matplotlib inline
# Generate some random data-points

data = pd.DataFrame(np.random.randint(low=1, high=15, size=(5,2)))

data.columns = ['x', 'y']

data.head()
# Plot (x,y) value

# fit_reg=False doesn't fix a regression line

sns.lmplot('x',

           'y',

           data=data,

           fit_reg=False,

           scatter_kws={'s': 100})



plt.title('Plotted data-points')



plt.xlabel('x')

plt.ylabel('y')
# Get data-frame values into numpy array

xy_array = data.values

xy_array
# Compute L(p=2) norm

linalg.norm(xy_array, ord=2, axis=1)
from sklearn.linear_model import Ridge

# Define matplotlib figure size to draw

rcParams['figure.figsize'] = 10, 7



# Generate an array that contains necessary data-points to draw a cosinr curve

x = np.array([r*np.pi/180 for r in range(70,300,7)])



#Define cosine curve range for plotting

m_plt = {1:231,3:232,6:233,9:234,12:235}

alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}



# Reproducability seed

np.random.seed(100)



# Adding some random noise

nz = np.random.normal(0,0.15,len(x))

y = np.cos(x) + nz



# Set x,y value as coordinate

cs = np.column_stack([x,y])



# Convert column_stack plotted value into data frame

data = pd.DataFrame(cs, columns=['x','y'])



# Plot generated data-points

plt.plot(data['x'],data['y'],'.')
# Linear regression generic function

# This function returns Residual Sum of Squares and the number of estimated coefficients

#Import Linear Regression model from scikit-learn.

from sklearn.linear_model import LinearRegression

def lr(d, p, mp):

    

    # Initialize prediction variable

    prd=['x']

    if p>=2:

        prd.extend(['x_%d'%i for i in range(2,p+1)])

    linreg = LinearRegression(normalize=True)

    linreg.fit(d[prd],d['y'])

    y_pred = linreg.predict(d[prd])

    

    if p in mp:

        plt.subplot(mp[p])

        plt.tight_layout()

        plt.plot(d['x'],y_pred)

        plt.plot(d['x'],d['y'],'.')

        plt.title('%d'%p)

    

    # construct predefined format

    res = sum((y_pred-data['y'])**2)

    

    et = [res]

    et.extend([linreg.intercept_])

    et.extend(linreg.coef_)

    

    return et
# Generate data

for i in range(2,12):

    data['x_%d'%i] = data['x']**i

    

print(data.head())
# Save generated result and plot

data_column = ['rss','intercept'] + ['x%d'%i for i in range(1,12)]

data_index = ['power of %d'%i for i in range(1,12)]

c_mat = pd.DataFrame(index=data_index, columns=data_column)



# Visualize results

for i in range(1,12):

    c_mat.iloc[i-1,0:i+2] = lr(data, p=i, mp=m_plt)

    

# Display coefficient data table

pd.options.display.float_format = '{:,.3g}'.format



c_mat
# Generic l2 regularization method

def l2_norm(d, pred, a, mp={}):

    # fitting data

    l2n = Ridge(alpha=a,normalize=True)

    l2n.fit(d[pred],d['y'])

    y_pred = l2n.predict(d[pred])

    

    #Check alpha value and plot 

    if a in mp:

        plt.subplot(mp[a])

        

        plt.tight_layout()

        plt.plot(d['x'],y_pred)

        plt.plot(d['x'],d['y'],'.')

        plt.title('Lambda: %.3g'%a)

    

    #Return the result in pre-defined format

    res = sum((y_pred-data['y'])**2)

    

    x = [res]

    x.extend([l2n.intercept_])

    x.extend(l2n.coef_)

    

    return x
# Prediction variable initialization

p=['x']

p.extend(['x_%d'%i for i in range(2,12)])



# Store coefficients

data_column = ['rss','intercept'] + ['x%d'%i for i in range(1,12)]

data_index = ['lambda=%.2g'%alpha_ridge[i] for i in range(0,10)]

mat_l2 = pd.DataFrame(index=data_index, columns=data_column)



for i in range(10):

    mat_l2.iloc[i,] = l2_norm(data, p, alpha_ridge[i], models_to_plot)

    

# Display coefficient table

pd.options.display.float_format = '{:,.2g}'.format



mat_l2