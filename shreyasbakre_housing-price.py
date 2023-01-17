# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#import matplotlib.pyplot as pyplot

import matplotlib

import matplotlib.pyplot

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

#from matplotlib import cm

from sklearn import preprocessing, manifold, linear_model, metrics, model_selection, ensemble

import seaborn as sns



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

prices = pd.DataFrame({"Number of houses sold at a certain price":train["SalePrice"]})

prices.hist(bins=100)

plt.xlabel('Price')
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

prices = pd.DataFrame({"Number of houses sold at a certain price":train["YearBuilt"]})

prices.hist(bins=100)

plt.xlabel('Price')
matplotlib.pyplot.plot(train["YearBuilt"], train["SalePrice"], 'ro')

#plt.axis([0, 6, 0, 20])

matplotlib.pyplot.show()
import seaborn as sn

import pandas as pd

import numpy as np
curve = pd.read_csv( "curve.csv" )

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sn
x = [round( 1 + np.random.random(), 3 ) for _ in range(0, 100)]

y = list( map( lambda x: round( np.math.sin(6*x), 3 ), x ) )
y[0:10]
noise = [round( np.random.random()/2, 3 ) for _ in range(0, 100)]



y_rand = list( map( lambda a, b: round( a + b, 3 ), y, noise ) )

xy_df = pd.DataFrame( { 'x': x, 'y':y_rand } )

xy = xy_df.copy()
sn.lmplot( "x", "y", data=xy, fit_reg=False, size = 5 )

for i in range( 2, 20 ):

  xy_df[ 'x'+ str( i ) ] = list(map( lambda a:np.math.pow( a, i ),xy_df.x ))
xy_df
xy_df = xy_df[['x', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',

             'x8', 'x9', 'x10', 'x11','x12', 'x13', 'x14',

             'x15', 'x16', 'x17', 'x18', 'x19', 'y']]

xy_df.iloc[:,:2]


from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

lreg.fit( xy_df.iloc[:,:2], xy_df.y )

lreg_predict_y = lreg.predict( xy_df.iloc[:,:2] )
lreg_predict_y
lreg.coef_
import pandas as pd

admission =pd.DataFrame({'admit':0,'gre':380,'gpa':3.61,'rank':3},

{'admit':1,'gre':660,'gpa':3.67,'rank':3},

{'admit':1,'gre':800,'gpa':4.00,'rank':1},

{'admit':1,'gre':640,'gpa':3.19,'rank':4},

{'admit':0,'gre':520,'gpa':2.93,'rank':4})
admission
admission.admit == 0