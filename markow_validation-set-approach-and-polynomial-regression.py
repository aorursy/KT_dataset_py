# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#grid needed for plotting charts: 101 equidistant points between -20 and 20

grid = np.linspace(-20,20,101) 

#artificial input values x match the grid

x = np.linspace(-20,20,101) 

#artifical output values y: polynomial of degree 5 plus a random error term (normally distributed with mean 0 and standard deviation 2, using a fixed seed for reproducible results)

#So the best estimator for y should be a polynom of degree 5

np.random.seed(111)

y = 0.000005*x**5 + 2*np.random.randn(len(x))
import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('default')

plt.plot(x,y,'o', label = 'Output y') # Artificial Datapoints

plt.plot(x,0.000005*x**5,'--',lw=4, label = 'Underlying model') #'Real' underlying model

plt.title('Artificial data points compared to underlying model')

plt.xlabel('x')

plt.legend(loc ='lower right',frameon=False)

plt.show()

from numpy.polynomial import polynomial as P



#Create an empty array to store the fitted polynomial functions

p = [] 



#Fit polynomials of orders 0 to 30 and store them in the array

for i in range(0,31):

    p.append(P.polyfit(x,y,i)) 

    

#Plot selected polynomials and compare them to data points as well as the known underlying relationship

plt.plot(x,y,'o',label = 'Output y') # Datapoints

plt.plot(x,0.000005*x**5,'--',lw=4, label = 'Underlying model') #'Real' underlying model

plt.plot(grid,P.polyval(grid,p[0]),'-', label ='Fit Order 0') #Flat line (polynomial of order 0)

plt.plot(grid,P.polyval(grid,p[1]),'-', label = 'Fit Order 1') #Linear regression (1st order polynom)

plt.plot(grid,P.polyval(grid,p[3]),'-', label = 'Fit Order 3') #3rd order polynomial

plt.plot(grid,P.polyval(grid,p[5]),'-', label = 'Fit Order 5') #5th order polynomial

plt.plot(grid,P.polyval(grid,p[29]),'-', label = 'Fit Order 29') #29th order polynomial

plt.legend(loc ='lower right',frameon=False, ncol=2)

plt.title('Datapoints and fits based on the entire data set')

plt.xlabel('x')

plt.show()
from sklearn.metrics import mean_squared_error

yfit = []

mse =[]



for i in range(0,31):

    #y values of the polynomials evaluated at the input points x

    yfit.append(P.polyval(x,p[i])) 

    #mean of squared differences between the datapoints y and the fitted y-values

    mse.append(mean_squared_error(y,yfit[i]))



#display overview about mse's

plt.bar(range(0,31),mse, label = 'mse') # Artificial Datapoints

plt.title('Accuracy against complexity of model (entire data set)')

plt.xlabel('Order of fitted polynomial function')

plt.ylabel('Mean squared error')

plt.show()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4)



#Array of polynomial functions fitted only on the training data

p_train = [] 

for i in range(0,31):

    p_train.append(P.polyfit(x_train,y_train,i)) 

    

#Plot selected polynomials and compare to the train data points as well as the known underlying relationship

plt.plot(x_train,y_train,'o',label = 'y_train') # Training data set

plt.plot(grid,0.000005*x**5,'--',lw=4, label = 'Underlying model') #'Real' underlying model

plt.plot(grid,P.polyval(grid,p_train[0]),'-', label ='Fit Order 0') #Flat line (polynomial of order 0)

plt.plot(grid,P.polyval(grid,p_train[1]),'-', label = 'Fit Order 1') #Linear regression (1st order polynomial)

plt.plot(grid,P.polyval(grid,p_train[3]),'-', label = 'Fit Order 3') #3rd order polynomial

plt.plot(grid,P.polyval(grid,p_train[5]),'-', label = 'Fit Order 5') #5th order polynomial

plt.plot(grid,P.polyval(grid,p_train[29]),'-', label = 'Fit Order 29') #29th order polynomial

plt.legend(loc ='lower right',frameon=False, ncol=2)

plt.title('Training data and fits based only on the training data')

plt.xlabel('x_train')

plt.ylim(bottom=-25,top = 25)

plt.show()
yfit_train = []

mse_train =[]



for i in range(0,31):

    #y values of the polynomials evaluated at the input points x_train.

    yfit_train.append(P.polyval(x_train,p_train[i])) 

    #mean of squared differences between the datapoints y and the fitted y-values

    mse_train.append(mean_squared_error(y_train,yfit_train[i]))



#display overview about mse's

plt.bar(range(0,31),mse_train, label = 'mse based on training set') # Artificial Datapoints

plt.title('Accuracy against complexity of model (training data set)')

plt.xlabel('Order of fitted polynomial function')

plt.ylabel('Mean squared error')

plt.show()
#Plot selected polynomials and compare to test data points as well as the known underlying relationship

plt.plot(x_test,y_test,'o',label = 'y_test') # Test data set

plt.plot(grid,0.000005*x**5,'--',lw=4, label = 'Underlying model') #'Real' underlying model

plt.plot(grid,P.polyval(grid,p_train[0]),'-', label ='Fit Order 0') #Flat line (polynomial of order 0)

plt.plot(grid,P.polyval(grid,p_train[1]),'-', label = 'Fit Order 1') #Linear regression (1st order polynomial)

plt.plot(grid,P.polyval(grid,p_train[3]),'-', label = 'Fit Order 3') #3rd order polynomial

plt.plot(grid,P.polyval(grid,p_train[5]),'-', label = 'Fit Order 5') #5th order polynomial

plt.plot(grid,P.polyval(grid,p_train[29]),'-', label = 'Fit Order 29') #29th order polynomial

plt.legend(loc ='lower right',frameon=False, ncol=2)

plt.title('Test data against fits based on the training data')

plt.xlabel('x_test')

plt.ylim(bottom=-25,top = 25)

plt.show()
yfit_test = []

mse_test =[]



for i in range(0,31):

    #y values of the polynomials evaluated at the input points x_test.

    yfit_test.append(P.polyval(x_test,p_train[i])) 

    #mean of squared differences between the datapoints y and the fitted y-values

    mse_test.append(mean_squared_error(y_test,yfit_test[i]))



#display overview about mse's

plt.bar(range(0,31),mse_test, label = 'mse based on test set') # Artificial Datapoints

plt.title('Accuracy against complexity of model (test data set)')

plt.xlabel('Order of fitted polynomial function')

plt.ylabel('Mean squared error')

plt.ylim(top=100)

plt.show()
mse_df = pd.DataFrame(data=mse_test) #Store mse results in a data frame in order to display them 

mse_df.index.name='Order'

mse_df.columns=['mse']

mse_df.head(16).T #Display transposed dataframe
#extended grid needed for plotting chart including extrapolation: 101 equidistant points between -30 and 30

grid_extended = np.linspace(-30,30,101) 



plt.style.use('default')

plt.plot(x,y,'o', label = 'Data') # Artificial Datapoints

plt.plot(grid_extended,0.000005*grid_extended**5,'--',lw=4, label = 'Underlying model') #'Real' underlying model

plt.plot(grid_extended,P.polyval(grid_extended,p_train[5]), '-', lw=3, label = 'Selected model Order 5')

plt.plot(grid_extended,P.polyval(grid_extended,p_train[9]),'-', label = 'Model Order 9') 

plt.plot(grid_extended,P.polyval(grid_extended,p_train[29]),'-', label = 'Model Order 29') 

 

     

plt.title('Using the fitted polynomial for extrapolation')

plt.xlabel('x')

plt.ylim(top=50, bottom = -50)

plt.legend(loc ='lower center',frameon=False)

plt.show()
