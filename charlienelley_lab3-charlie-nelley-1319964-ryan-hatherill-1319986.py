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
data = pd.read_csv('../input/housing.csv')

data = data.fillna(0)

data.head()



data.info()
ocean_proximity = {'NEAR BAY':1, '<1H OCEAN': 2, 'INLAND':3, 'NEAR OCEAN':4, 'ISLAND':5}

data.ocean_proximity = [ocean_proximity[i] for i in data.ocean_proximity]

data.head()

from sklearn.model_selection import train_test_split



x = data.loc[:, list(data.columns[0:9])] 

y = data.iloc[:, 9] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1319964)

print(x_train.shape, y_train.shape)
from sklearn.metrics import mean_absolute_error 

from matplotlib import pyplot as plt 

import seaborn as sns



def run_reg(regressor, x_train, x_test, y_train, y_test):

    

    regressor.fit(x_train, y_train)# train the classifier using the train data    

    pred = regressor.predict(x_test)# compute predictions for the test data

    # set all predictions that are smaller than 15000, to 15000

    # set all predictions that are larger than 500000, to 500000

    pred[pred<15000] = 15000

    pred[pred>500000] = 500000

    mae = mean_absolute_error(y_test, pred) # compute the MAE (mean_absolute_error) for the test data

    acc=regressor.score(x_test, y_test)

    print("the accuracy percentage is " + str(acc*100) +"%")

    # scatterplot the true test targets vs. predictions (show MAE in the plot as "title")

    plt.title('MAE = ' + str(mae))

    plt.scatter(y_test, pred)

    plt.show()

    plt.plot(regressor.coef_)# plot the coeffients of the model

    plt.show()

    return mae #return the MAE
#using ridge with an alpha of 0.1

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha = 0.1, solver = "cholesky")

run_reg(ridge_reg, x_train, x_test, y_train, y_test)

from sklearn.linear_model import Ridge #using ridge with an alpha of 0.01

ridge_reg = Ridge(alpha = 0.001, solver = "cholesky")

run_reg(ridge_reg, x_train, x_test, y_train, y_test)
from sklearn.linear_model import Lasso #using Lasso with an alpha of 0.1

lasso_reg = Lasso(alpha=0.1)

run_reg(lasso_reg, x_train, x_test, y_train, y_test)
from sklearn.linear_model import Lasso #using Lasso with an alpha of 0.01

lasso_reg = Lasso(alpha=0.001)

run_reg(lasso_reg, x_train, x_test, y_train, y_test)
from sklearn.linear_model import ElasticNet #using elastic net with an alpha of 0.1

elastic_net = ElasticNet(alpha = 0.1)

run_reg(elastic_net, x_train, x_test, y_train, y_test)
from sklearn.linear_model import ElasticNet #using elastic net with an alpha of 0.01

elastic_net = ElasticNet(alpha = 0.001)

run_reg(elastic_net, x_train, x_test, y_train, y_test)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(3) #generating polynomial features with a value of 3

x_test_poly3 = poly.fit_transform(x_test) #fitting the poly features to x_test 

x_train_poly3 = poly.fit_transform(x_train)

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha = 0.1, solver = "cholesky")

run_reg(ridge_reg, x_train_poly3, x_test_poly3, y_train, y_test) #using poly to run predictions again on each model
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha = 0.001, solver = "cholesky")

run_reg(ridge_reg, x_train_poly3, x_test_poly3, y_train, y_test)
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)

run_reg(lasso_reg, x_train_poly3, x_test_poly3, y_train, y_test)
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.001)

run_reg(lasso_reg, x_train_poly3, x_test_poly3, y_train, y_test)
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha = 0.1)

run_reg(elastic_net, x_train_poly3, x_test_poly3, y_train, y_test)
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha = 0.001)

run_reg(elastic_net, x_train_poly3, x_test_poly3, y_train, y_test)