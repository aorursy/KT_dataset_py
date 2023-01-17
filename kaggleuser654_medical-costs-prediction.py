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
#read in csv file

insurance = pd.read_csv('/kaggle/input/insurance/insurance.csv')
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
insurance.head()
insurance.info()
categorical_vars = ['sex', 'smoker', 'region']



for var in categorical_vars:

    dummies = pd.get_dummies(insurance[var], prefix = var)

    insurance = pd.concat([insurance, dummies], axis=1)

    

insurance.info()
#list of features for model

features = insurance.columns.tolist()

#removing categoricals 'sex', 'smoker', 'region' as we have created dummy variables for them; also remove target variable 'charges'

for f in ['sex','smoker','region','charges']:

    features.remove(f)

#check

features
insurance[['age', 'bmi', 'children', 'charges']].describe()
for f in ['sex_male','smoker_no']:

    features.remove(f)
#pairwise correlation between features 'age','bmi','children', and the target variable 'charges'

corr = insurance[['age','bmi','children','charges']].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, cmap="YlOrRd", annot=True)

plt.show()
#scatter plots

fig, axs = plt.subplots(3,3, figsize= (20,20))



f = 0

for i in range(3):

    for j in range(3):

        axs[i,j].scatter(insurance[features[f]], insurance['charges'], marker = 'x')

        axs[i,j].set_title('Charges and '+ features[f])

        plt.xlabel(features[f])

        plt.ylabel('charges')

        f += 1

    

plt.show()
#splitting the insurance data set into train and test datasets

from sklearn.model_selection import train_test_split



train_X, test_X, train_y, test_y = train_test_split(insurance[features], insurance['charges'], test_size = 0.2, random_state=0)
#scale features and target

for f in ['age', 'bmi', 'children']:

    train_X[f + '_scaled'] = (train_X[f]- train_X[f].mean())/(train_X[f].std())

    test_X[f + '_scaled'] = (test_X[f]- test_X[f].mean())/(test_X[f].std())

    features.append(f + '_scaled')

    features.remove(f)

    

    

train_y = (train_y - train_y.mean())/(train_y.std())

test_y = (test_y - test_y.mean())/(test_y.std())
print(features)

print(train_X.columns)

print(test_X.columns)
#vectorised cost function



def CostFunction(X, y, theta):

    

    m = len(y)

    cost = (1/(2*m))*np.sum((X@theta - y)**2)

    

    return cost


def GradientDescent(X,y,theta,alpha, precision):

  

    m = len(y)

    cost_history = []

    

    previous_cost = CostFunction(X, y, theta)

    cost_history.append(previous_cost)

    cost = 0

    iter = 1

    

    while (previous_cost - cost) > precision:

        if iter == 1:

            pass

        else:

            previous_cost = cost

        theta_temp = np.copy(theta)

        for j in range(len(theta)):

            theta[j] = theta[j] - (alpha/m)*np.sum((X@theta_temp - y)*X[:,j])

        cost = CostFunction(X,y,theta)

        cost_history.append(cost)

        iter +=1

    

    return theta, cost_history
#store target variable in y as numpy array

y = np.array(train_y)

m = len(y)



#X is matrix with features data and a columns of ones as the first column

array_train = np.array(train_X[features])

X = np.insert(array_train, 0, 1, axis=1)
from time import perf_counter 



time_start = perf_counter()

opt_theta, cost_hist_1 = GradientDescent(X, y, theta = np.zeros(X.shape[1]), alpha = 0.01, precision = 0)

time_stop = perf_counter()

print('Convergence time: {:.2f} seconds'.format(time_stop - time_start))

print('Converged at cost = {:.4f}'.format(cost_hist_1[-1]))
plt.figure(figsize = (9.6, 7.2))

plt.plot(cost_hist_1)

plt.xlabel('Number of iterations')

plt.ylabel('Mean Squared Error')

plt.title('Cost history with learning rate = 0.01')

plt.show()



print('Function converged after {} iterations.'.format(len(cost_hist_1)))
#learning rate increased to 0.1

time_start = perf_counter()

opt_theta, cost_hist_2 = GradientDescent(X, y, theta = np.zeros(X.shape[1]), alpha = 0.1, precision = 0)

time_stop = perf_counter()

print('Convergence time: {:.2f} seconds'.format(time_stop - time_start))

print('Converged at cost = {:.4f}'.format(cost_hist_2[-1]))
plt.figure(figsize = (9.6, 7.2))

plt.plot(cost_hist_2)

plt.xlabel('Number of iterations')

plt.ylabel('Mean Squared Error')

plt.title('Cost history with learning rate = 0.1')

plt.show()



print('Function converged after {} iterations.'.format(len(cost_hist_2)))
print('Increasing the learning rate from 0.01 to 0.1 reduced the number of iterations necessary for the function to converge by {:.2f}%'.format(

    ((len(cost_hist_1) - len(cost_hist_2))/(len(cost_hist_1)))*100))
from sklearn.metrics import mean_squared_error



#convert test_X into numpy array with initial column of ones

array_test = np.array(test_X[features])

X_test = np.insert(array_test, 0, 1, axis=1)

#calculate predictions on test data

predictions_test = X_test@opt_theta

#mean squared error

mse_gd = CostFunction(X_test, test_y, opt_theta)

print('The mean squared error of the model on test data is {:.3f}.'.format(mse_gd))