import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/bigcar_matlab.csv')
print(data.head())

print(data.info())
data = data.drop(data.index[data['Horsepower'].isnull()])

print(data.info())
plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2)

plt.xlabel('Horsepower');plt.ylabel('Weight')

plt.show()
# construct input and target values

data_arr = np.array(data); n = len(data_arr)

horsepower = data_arr[:,0].reshape(n,1)

weight = data_arr[:,1].reshape(n,1)



# normalization of feature and target

x = np.c_[np.ones((n,1)),weight/np.max(weight)]

t = horsepower/np.max(horsepower)



# closed form least square solution

Weight_closed = np.linalg.inv(np.transpose(x).dot(x)).dot(np.transpose(x)).dot(t)



# map weight vector to orginal space , before normalization

Weight_closed[0] = max(horsepower)*Weight_closed[0]

Weight_closed[1] = (max(horsepower)*Weight_closed[1])/(max(weight))

print('weight vectors:',Weight_closed)



# plotting data with model

y_plot_closed = Weight_closed[0] + Weight_closed[1]*(weight)

plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2,label='data')

plt.plot(data['Weight'],y_plot_closed,c='b',label='closed form linear model')

plt.legend()

plt.xlabel('Weight');plt.ylabel('Horsepower')

plt.show()
from sklearn import linear_model

reg_lin = linear_model.LinearRegression()

reg_lin.fit(weight,horsepower)

print ('weight vectors: ',reg_lin.intercept_,reg_lin.coef_)



plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2,label='data')

plt.plot(weight,reg_lin.predict(weight),c='b',label='scikit-learn closed form solution')

plt.legend()

plt.xlabel('Weight');plt.ylabel('Horsepower')

plt.show()
# assing initial parameters 

w_gd = np.random.randn(2,1); lr = 0.5; misfit = 10; max_iteration = 5000

cos_func_val=np.zeros((max_iteration,1))



for i in range(max_iteration):

    thetax = x.dot(w_gd)

    #gradient of cost function

    grad = (1/n)*(np.transpose(x).dot(thetax-t))

    #update weight vectors

    w_gd = w_gd - lr*grad

    #calculate cost function during iteration for plotting

    cos_func_val[i] = (thetax-t).T.dot(thetax-t)

    

    # early stopping based on misfit barely changes between iterations

    if i>0:

        misfit = abs(cos_func_val[i]-cos_func_val[i-1])

    if misfit <0.00001:

        break



# map weight vector to orginal space , before normalization

w_gd[0] = max(horsepower)*w_gd[0]

w_gd[1] = (max(horsepower)*w_gd[1])/(max(weight))

print('weight vectors:',w_gd)



plt.plot(cos_func_val[0:i])

plt.xlabel('iteration number');plt.ylabel('cost function value')

plt.show()
plt.figure(figsize=(20,10))

y_plot_closed = Weight_closed[0] + Weight_closed[1]*(weight)

y_plot_gd = w_gd[0] + w_gd[1]*(weight)

plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2,label='data')

plt.plot(data['Weight'],y_plot_closed,c='b',label='closed form linear model')

plt.plot(data['Weight'],y_plot_gd,c='k',label='gradient descent linear model')

plt.legend()

plt.xlabel('Weight');plt.ylabel('Horsepower')

plt.show()



print('weight vectors from closed form:',Weight_closed)

print('weight vectors from gradient descent:',w_gd)
sgd_lin = linear_model.SGDRegressor(loss='squared_loss',penalty=None,max_iter=5000,eta0=0.5,tol=0.0001)

sgd_lin.fit(weight,horsepower.ravel())

print ('weight vectors: ',sgd_lin.intercept_,sgd_lin.coef_)
sgd_lin = linear_model.SGDRegressor(loss='squared_loss',penalty=None,max_iter=5000,eta0=0.5,tol=0.0001)

sgd_lin.fit(weight/max(weight),horsepower.ravel()/max(horsepower))

sgd_lin.intercept_ = max(horsepower)*sgd_lin.intercept_

sgd_lin.coef_ = (max(horsepower)*sgd_lin.coef_)/(max(weight))

print ('weight vectors: ',sgd_lin.intercept_,sgd_lin.coef_)
plt.figure(figsize=(20,10))

y_plot_sgd = sgd_lin.intercept_ + sgd_lin.coef_*(weight)

plt.plot(weight,reg_lin.predict(weight),c='b',label='skitlearn closed form solution')

plt.scatter(data['Weight'],data['Horsepower'],marker='o',c='r',s=2,label='data')

plt.plot(data['Weight'],y_plot_sgd,c='k',label='skitlearn SDG solution')

plt.legend()

plt.xlabel('Weight');plt.ylabel('Horsepower')

plt.show()