# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode, download_plotlyjs, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print()

print("The files in the dataset are:-")

from subprocess import check_output

print(check_output(['ls','../input']).decode('utf'))



# Any results you write to the current directory are saved as output.
# Importing the dataset.

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv('../input/housing.csv', delim_whitespace=True, names=names)
df.head()
df.info()
df.corr().iplot(kind='heatmap', )
# Importing of Useful libraries from sklearn library.

from sklearn.preprocessing import StandardScaler   # For Scaling the dataset

from sklearn.model_selection import train_test_split    # For Splitting the dataset

from sklearn.linear_model import LinearRegression      # For Linear regression

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
# Let us Create Feature matrix and Target Vector.

x_train = df.iloc[:,:-1].values

y_train = df.iloc[:,-1].values
sc_X=StandardScaler()

x_train=sc_X.fit_transform(x_train)
from sklearn.decomposition import PCA

pca = PCA(n_components=None)

x_train = pca.fit_transform(x_train)



explained_variance = pca.explained_variance_ratio_

explained_variance

print(f"The sum of initial 5 values is \t {0.47+0.11+0.09+0.06+0.06} , which is very good." )

print("So we will choose 5 number of features and reduce our training feature matrix to 5 features/columns. ")

pca = PCA(n_components=5)

x_train = pca.fit_transform(x_train)
def all_models():    

    # Multi-linear regression Model. 

    regressor_multi = LinearRegression()

    regressor_multi.fit(x_train,y_train)

    # Let us check the accuray

    accuracy = cross_val_score(estimator=regressor_multi, X=x_train, y=y_train,cv=10)

    print(f"The accuracy of the Multi-linear Regressor Model is \t {accuracy.mean()}")

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    print()

    

    # Polynomial Regression

    from sklearn.preprocessing import PolynomialFeatures

    poly_reg=PolynomialFeatures(degree=4) #These 3 steps are to convert X matrix into X polynomial

    x_poly=poly_reg.fit_transform(x_train) #matrix. 

    regressor_poly=LinearRegression()

    regressor_poly.fit(x_poly,y_train)

    # Let us check the accuray

    accuracy = cross_val_score(estimator=regressor_poly, X=x_train, y=y_train,cv=10)

    print(f"The accuracy of the Polynomial Regression Model is \t {accuracy.mean()}")

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    print()

    

    # Random Forest Model

    regressor_random = RandomForestRegressor(n_estimators=100,)

    regressor_random.fit(x_train,y_train)

    # Let us check the accuray

    accuracy = cross_val_score(estimator=regressor_random, X=x_train, y=y_train,cv=10)

    print(f"The accuracy of the Random Forest Model is \t {accuracy.mean()}")

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    print()

    

    # SVR 

    regressor_svr = SVR(kernel='rbf')

    regressor_svr.fit(x_train, y_train)

    # Let us check the accuracy

    accuracy = cross_val_score(estimator=regressor_svr, X=x_train, y=y_train,cv=10)

    print(f"The accuracy of the SVR Model is \t {accuracy.mean()}")

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    print()

    

    # Decision Tress Model

    regressor_deci = DecisionTreeRegressor()

    regressor_deci.fit(x_train, y_train)

    # Let us check the accuracy

    accuracy = cross_val_score(estimator=regressor_deci, X=x_train, y=y_train,cv=10)

    print(f"The accuracy of the Decision Tree Model is \t {accuracy.mean()}")

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    

    



    
# Let us run all models together. If we have large dataset then we will not run all models together.

# Then we will run one model at a time, otherwise your processor will struck down.

all_models()