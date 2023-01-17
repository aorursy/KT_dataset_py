# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression



#read the files

nRowsRead=1000

data = pd.read_csv('/kaggle/input/continuous_factory_process.csv', delimiter=',')



#stage 1

regressor = LinearRegression() 

file_name = "regression_report.txt"

k=1

with open(file_name, 'w') as x_file:

    i=42

    while(i<72 ):

        j=i+1

        x=data.iloc[:,i].values.reshape(-1, 1)

        y=data.iloc[:,j].values.reshape(-1, 1)

        

    

        #training the algorithm

        regressor.fit(x, y) 

        y_pred = regressor.predict(x)

        #plotting the regression line 

        #plt.plot(y_pred, x, color = "g")

        #plt.xlabel('x') 

        #plt.ylabel('y') 

    

    

    

        #To retrieve the intercept:

        print('Stage 1')

        print('Regression Intercept',regressor.intercept_)

        #For retrieving the slope:

        print('Regression Coefficient',regressor.coef_)

        x_file.write('CASE {}:********'.format(k))

        x_file.write('\n')

        x_file.write('Regression Intercept {}'.format(regressor.intercept_))

        x_file.write('\n')

        x_file.write('Regression Coefficient {}'.format(regressor.coef_))

        x_file.write('\n')

        x_file.write('\n')

        print('Measurement at output location', k,'\n')

        k=k+1

        i=j+1

        print(i,j)

#stage 2

regressor = LinearRegression() 

file_name = "regression_report.txt"

k=1

with open(file_name, 'w') as x_file:

    i=86

    while(i<116):

        j=i+1

        x=data.iloc[:,i].values.reshape(-1, 1)

        y=data.iloc[:,j].values.reshape(-1, 1)

        

    

        #training the algorithm

        regressor.fit(x, y) 

        y_pred = regressor.predict(x)

        #plotting the regression line 

        #plt.plot(y_pred, x, color = "g")

        #plt.xlabel('x') 

        #plt.ylabel('y') 

    

    

        print(i,j)

        #To retrieve the intercept:

        print('Regression Intercept',regressor.intercept_)

        #For retrieving the slope:

        print('Regression Coefficient',regressor.coef_)

        x_file.write('CASE {}:********'.format(k))

        x_file.write('\n')

        x_file.write('Regression Intercept {}'.format(regressor.intercept_))

        x_file.write('\n')

        x_file.write('Regression Coefficient {}'.format(regressor.coef_))

        x_file.write('\n')

        x_file.write('\n')

        print('Measurement at output location', k)

        print('\n')

        k=k+1

        i=j+1
