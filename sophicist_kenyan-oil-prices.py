# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


from IPython.core.display import display, HTML
display(HTML("""<a href="https://wordpress.com/customize/identity/pythonkenya.wordpress.com">Kenyan Data Online</a>"""))
# for more on this data click on the ctr + link below 
Oil = pd.read_csv('../input/oil2.csv')

Oil.head()
# to make visualizations we need to convert the date column into a time stamp as follows

H = []
import datetime
for time in Oil.Time:
    v = datetime.datetime.strptime(time, '%Y-%m-%d')
    H.append(v)
Oil['Time'] = H
#Now we can draw the visualizations


    
    

filter1 = ['PMS', 'AGO', 'Kero', 'Price Per Barell']
filter2 = ['Int Price in KSH','Local Price in KSH']
Oil1 = Oil[filter1]
Oil2 = Oil[filter2]
Oil.head()

import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
plt.plot(Oil.Time, Oil.PMS)
plt.plot(Oil.Time, Oil.AGO)
plt.plot(Oil.Time, Oil.Kero)
plt.plot(Oil.Time, Oil['Price Per Barell'],'r-o')
plt.legend()
plt.title('Oil Prices Movements in Kenya \n vs International barell price  \n 2010-2018')
plt.xlabel('Time in Years')
plt.ylabel('Price')
plt.show()
# a more rigorous analysis of the price movements would be a correlation analysis as follows
#note the very high positive correlation coefficients
Oil1.corr()
#Having combined the 'Ago,PMS and kero' into 1 local price and coverted the international price per barell into int price.
#we can compare the correlation on both to see very high positive correlation
Oil2.corr()
# we can use heat maps fron the seaborn python module to visualize the corelation and better understand them.
Oil1corr = Oil1.corr()
Oil2corr = Oil2.corr()
import seaborn as sn
plt.figure(figsize = (11,5))
plt.subplot(221)
sn.heatmap(Oil1corr,cmap="YlGnBu")
plt.title('Correlation heat maps')
plt.subplot(223)
sn.heatmap(Oil2corr,cmap="Greens")
plt.show()

#Data preperations
#create dataset for machine learning training and testing: You need an X matrix and a Y-matrix/vector for this to work
filter3 = [ 'PMS', 'AGO', 'Kero', 'Price Per Barell','Exrates', ]
X = Oil[filter3]
filter4 = ['PMS','AGO','Kero']
Y  = X[filter4].shift(periods =-1) # where last months prices predict current months prices

#Split the data between 'Training data' and 'Testing Data'...test-size is the percentage of the data that should be used for testing.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.3)
print(X.head())
print(Y.head())
#Decision tree regressor model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#fit the model:
X_train = X_train.fillna(0)
Y_train = Y_train.fillna(0)
DTR = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best')
DTR.fit(X_train,Y_train)
X_test = X_test.fillna(0)
Y_test = Y_test.fillna(0)
predictions = DTR.predict(X_test)
import math
math.sqrt(mean_squared_error( Y_test ,predictions))

#predict the X_test values and compare to Y_test
X_test = X_test.fillna(0)
Y_test = Y_test.fillna(0)
predictions = DTR.predict(X_test)
import math
print(r2_score(Y_test ,predictions))# this metric measures how well the data fits the model; a score of 85% is not a bad one
math.sqrt(mean_squared_error( Y_test ,predictions))# the lower the measure the better,4 is a good measure
#r2_score(Y_test,predictions)
#use linear regression 
from sklearn.linear_model import LinearRegression
LR= LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
LR.fit(X_train,Y_train)
pred = LR.predict(X_test)
print(r2_score(Y_test ,pred))# this metric measures how well the data fits the model; a score of 73% is not a bad one
import math
math.sqrt(mean_squared_error( Y_test ,pred)) 
G =pd.DataFrame(predictions,columns=['PMS1','AGO1','Kero1'],)
GG =pd.DataFrame(pred,columns=['PMS2','AGO2','Kero2'],)
V = pd.concat((G,GG),axis = 1)
print(Y_test.head())# the actual data
print(V.head())# two predictions with the first from 'Decision Tree' and the second from 'LinearRegression', the second seems more accurate

# the second machine learning method seems like a better fit because it has more accurate predictions.






