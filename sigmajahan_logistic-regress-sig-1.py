import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt



def plot_reg_withoutbeta(X, y): 

    ''' 

    function to plot decision boundary 

    '''

    # labelled observations 

    x_0 = X[np.where(y == 0.0)] 

    x_1 = X[np.where(y == 1.0)] 

      

    # plotting points with diff color for diff label 

    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0') 

    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1') 

      

    plt.xlabel('x1') 

    plt.ylabel('x2') 

    plt.legend() 

    plt.show()



def plot_reg(X, y, beta): 

    ''' 

    function to plot decision boundary 

    '''

    # labelled observations 

    x_0 = X[np.where(y == 0.0)] 

    x_1 = X[np.where(y == 1.0)] 

      

    # plotting points with diff color for diff label 

    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0') 

    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1') 

      

    # plotting decision boundary 

    x1 = np.arange(0, 1, 0.1) 

    x2 = -(beta[0,0] + beta[0,1]*x1)/beta[0,2] 

    plt.plot(x1, x2, c='k', label='reg line') 

  

    plt.xlabel('x1') 

    plt.ylabel('x2') 

    plt.legend() 

    plt.show()



def logistic_func(beta, X):

    return 1.0/(1 + np.exp(-np.dot(X, beta.T)))



def cost_func(beta, X, y): 

    ''' 

    cost function, J 

    '''

    log_func_v = logistic_func(beta, X) 

    y = np.squeeze(y) 

    step1 = y * np.log(log_func_v) 

    step2 = (1 - y) * np.log(1 - log_func_v) 

    final = -step1 - step2 

    return np.mean(final)





def log_gradient(beta, X, y): 

    ''' 

    logistic gradient function 

    '''

    first_calc = logistic_func(beta, X) - y.reshape(X.shape[0], -1) 

    final_calc = np.dot(first_calc.T, X) 

    return final_calc 





def grad_desc(X, y, beta, lr=.01, converge_change=.001): 

    ''' 

    gradient descent function 

    '''

    cost = cost_func(beta, X, y) 

    change_cost = 1

    num_iter = 1

      

    while(change_cost > converge_change): 

        old_cost = cost 

        beta = beta - (lr * log_gradient(beta, X, y)) 

        cost = cost_func(beta, X, y) 

        change_cost = old_cost - cost 

        num_iter += 1

      

    return beta, num_iter



def pred_values(beta, X): 

    ''' 

    function to predict labels 

    '''

    pred_prob = logistic_func(beta, X) 

    pred_value = np.where(pred_prob >= .5, 1, 0) 

    return np.squeeze(pred_value) 





df = pd.read_csv('/kaggle/input/Logistic_regression_dataset.csv', skiprows = 0)

a = np.array(df)



np.shape(a)
dt=pd.DataFrame(a)
dt.head()
# a=[1,2,3,4]
# a[0:-2]
X = a[:,0:-1]

Y = a[:,-1]

X_scaled = MinMaxScaler().fit_transform(X)

one = np.ones(len(X))

one_reshaped = one.reshape(len(X), 1)

X_total = np.c_[one_reshaped, X_scaled]

print(X_total)
plot_reg_withoutbeta(X_total, Y)
beta = [1, 10, 10]

beta = np.matrix(beta)
beta.shape




plot_reg(X_total, Y, beta)

beta, num_iter = grad_desc(X_total, Y, beta, lr=.01, converge_change=.001)

print(beta)



print(num_iter)



plot_reg(X_total, Y, beta)
pred_values(beta,50)