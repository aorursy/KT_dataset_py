# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/position-salaries/Position_Salaries.csv')
data
X=data['Level']
y=data['Salary']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
%matplotlib inline
plotting_xs=np.linspace(X.min(),X.max(),1000)
for i in range(1,11):
    # Applying Train Test Split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    
    # Creating polynomial object
    poly=PolynomialFeatures(degree=i)
    
    # Trasforming the training and testing data to required polynomial degree
    X_train=poly.fit_transform(X_train.values.reshape(X_train.shape[0],1))
    X_test=poly.fit_transform(X_test.values.reshape(X_test.shape[0],1))
    
    # Model
    regressor=LinearRegression()
    
    # Fitting
    regressor.fit(X_train,y_train)
    
    # Predicting
    y_pred=regressor.predict(X_test)
    
    
    # Generating predictions for 1000 points which will help in plotting the curve of regression
    xs=poly.fit_transform(plotting_xs.reshape(plotting_xs.shape[0],1))
    plotting_ys=regressor.predict(xs)
    
    # Plotting
    plt.title('Degree: '+str(i)+', R2 score: '+str(r2_score(y_pred,y_test)))
    
    # Plotting the training data 
    plt.scatter(X,y,label='Training Data')
    
    # Plotting the curve of regression
    plt.scatter(plotting_xs,plotting_ys,label='Curve of Regression')
    plt.legend()
    plt.show()
