# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
if __name__ == "__main__":

    dataset = pd.read_csv('/kaggle/input/whr2017.csv')

    print(dataset.describe())



    X = dataset[['gdp', 'family', 'lifexp', 'freedom' , 'corruption' , 'generosity', 'dystopia']]

    y = dataset[['score']]



    print(X.shape)

    print(y.shape)



    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)


    modelLinear = LinearRegression().fit(X_train, y_train)

    y_predict_linear =  modelLinear.predict(X_test)



    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)

    y_predict_lasso = modelLasso.predict(X_test)



    modelRidge = Ridge(alpha=1).fit(X_train, y_train)

    y_predict_ridge = modelRidge.predict(X_test)



    linear_loss = mean_squared_error(y_test, y_predict_linear)

    print("Linear Loss:", linear_loss)



    lasso_loss = mean_squared_error(y_test, y_predict_lasso)

    print("Lasso Loss: ", lasso_loss)



    ridge_loss = mean_squared_error(y_test, y_predict_ridge)

    print("Ridge Loss: ", ridge_loss)
    print("="*32)

    print("Coef LASSO")

    print(modelLasso.coef_)

    

    print("="*32)

    print("Coef RIDGE")

    print(modelRidge.coef_)