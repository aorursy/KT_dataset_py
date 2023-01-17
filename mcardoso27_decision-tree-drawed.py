import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor



# Generates 50 points between 0 and 1

x = np.linspace(0,1, num=100)



# Generates random points from a uniform distribution between -0.2 and 0.2, and added the x axis number. 

y = x + np.random.uniform(-0.2,0.2,x.shape)



plt.scatter(x,y)
# xx = np.linspace(0,1, num=10)

# print(xx.shape)

# print(np.random.permutation(len(xx)))

# print(np.random.permutation(len(xx))[:int(0.7*len(xx))])

# print(sorted(np.random.permutation(len(xx))[:int(0.7*len(xx))]))
# train, validation set split



# Take random indexs

idx_train = sorted(np.random.permutation(len(x))[:int(0.7*len(x))])



#Take the leaft indexs to validation

idx_test = [i for i in range(0,len(x)) if i not in idx_train]



x_trn, x_val = x[idx_train, None], x[idx_test, None]

y_trn, y_val = y[idx_train, None], y[idx_test, None]



# fit a model

m = DecisionTreeRegressor(max_depth=2).fit(x_trn, y_trn)
plt.scatter(x_val,m.predict(x_val),color='blue',label='Prediction')

plt.scatter(x_val,y_val,color='red',label='Actual')

plt.scatter(x_trn,y_trn,color='red')

plt.scatter(x_trn,m.predict(x_trn),color='blue')

plt.legend(loc='upper left')
# Fit a linear regression model

from sklearn.linear_model import LinearRegression



l = LinearRegression().fit(x_trn, y_trn)
# Let's plot the result of two approachs



plt.figure(figsize=(20,10))

plt.scatter(x,y,color='red',label='Actual')

plt.scatter(x,l.predict(x[:,None]),color='green',label='Prediction Linear')

plt.scatter(x,m.predict(x[:,None]),color='blue',label='Prediction DT')

plt.legend(loc='upper left')
from sklearn.tree import export_graphviz

import IPython, graphviz, re, math
def draw_tree(t, col_names, size=9, ratio=0.5, precision=3):

    """ Draws a representation of a random forest in IPython.

    Parameters:

    -----------

    t: The tree you wish to draw

    df: The data used to train the tree. This is used to get the names of the features.

    """

    s=export_graphviz(t, out_file=None, feature_names=col_names, filled=True,

                      special_characters=True, rotate=True, precision=precision)

    IPython.display.display(graphviz.Source(re.sub('Tree {',

       f'Tree {{ size={size}; ratio={ratio}',s)))

col_names =['X']

draw_tree(m, col_names, precision=3)