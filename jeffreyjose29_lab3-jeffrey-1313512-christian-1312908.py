# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_error

import seaborn as sns

import statsmodels.formula.api as smf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/housing.csv") #load data

data = data.fillna(0) #fill empty values with zero

data.head() #show first 5 fields
data.info() #display information about the data
#Convert the categorical ocean_proximity to numerical values

ocean_proximity = {'NEAR BAY': 1, '<1H OCEAN': 2, 'INLAND': 3, 'NEAR OCEAN': 4, 'ISLAND': 5}

data.ocean_proximity = [ocean_proximity[item] for item in data.ocean_proximity]



#show the first few values of the dataset

data.head()
#x is all the columns except for ocean proximity

x = data.iloc[:, 0:9] 

#I have set y to the ocean_proximity

y = data.iloc[:,9]





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1313512)



#Print the x and y columns

#x , y





#print the the size of the train and tests

print (x_train.shape, y_train.shape)

print (x_test.shape, y_test.shape)
def run_reg(regressor, x_train, x_test, y_train, y_test):

   

    #fit the model to the x and y trains

    regressor.fit(x_train, y_train)

    #make a prediction using the model

    prediction = regressor.predict(x_test)

    

    #check accuracy of prediction

    acc = regressor.score(x_test, y_test)

    print("The prediction accuracy is: {:0.2f}%".format(acc * 100))

    

    #set all predictions within a range of 15000 to 500000

    prediction[prediction < 15000] = 15000

    prediction[prediction > 500000] = 500000

    

    #calculate mae

    MAE = mean_absolute_error(y_test, prediction)

    #plot the predictions

    plt.scatter(y_test, prediction)

    plt.title('MAE = ' + str(MAE))

    plt.show()  

    

    #plot the coefficients of the model

    plt.plot(regressor.coef_)

    plt.title("Coefficient Of The Model")

    plt.show()

    

    return MAE

#information about x_train for debugging purposes

x_train.info()
#Ridge with 0.1 alpha

from sklearn.linear_model import Ridge

ridge_reg_1 = Ridge(alpha = 0.1, solver = 'cholesky')

run_reg(ridge_reg_1, x_train, x_test, y_train, y_test)
#Ridge with 0.001 alpha

from sklearn.linear_model import Ridge

ridge_reg_2 = Ridge(alpha = 0.001, solver = 'cholesky')

run_reg(ridge_reg_2, x_train, x_test, y_train, y_test)
#Lasso model with 0.1 alpha

from sklearn.linear_model import Lasso

lasso_reg_1 = Lasso(alpha = 0.1)

run_reg(lasso_reg_1, x_train, x_test, y_train, y_test)
#Lasso model with 0.001 alpha

from sklearn.linear_model import Lasso

lasso_reg_2 = Lasso(alpha = 0.001)

run_reg(lasso_reg_2, x_train, x_test, y_train, y_test)
#ElasticNet with 0.1 alpha

from sklearn.linear_model import ElasticNet

elastic_net_1 = ElasticNet(alpha=0.1, random_state=1313512)

run_reg(elastic_net_1, x_train, x_test, y_train, y_test)
#ElasticNet with 0.001 alpha

from sklearn.linear_model import ElasticNet

elastic_net_2 = ElasticNet(alpha=0.001, random_state=1313512)

run_reg(elastic_net_2, x_train, x_test, y_train, y_test)
# Modify X_train + X_test using the PolynomialFeatures class with degree=3

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(3)

x_train_poly3 = poly.fit_transform(x_train)

x_test_poly3 = poly.fit_transform(x_test)



run_reg(ridge_reg_1, x_train_poly3, x_test_poly3, y_train, y_test)
run_reg(ridge_reg_2, x_train_poly3, x_test_poly3, y_train, y_test)
run_reg(lasso_reg_1, x_train_poly3, x_test_poly3, y_train, y_test)
run_reg(lasso_reg_2, x_train_poly3, x_test_poly3, y_train, y_test)
run_reg(elastic_net_1, x_train_poly3, x_test_poly3, y_train, y_test)
run_reg(elastic_net_2, x_train_poly3, x_test_poly3, y_train, y_test)