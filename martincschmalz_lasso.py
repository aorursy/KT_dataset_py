from sklearn.linear_model import Lasso #gain access to the functions and variables to the module Lasso from sklearn library's linear_model feature

model = Lasso(alpha=0) #build the model with choice of penaltry alpha

X_train = [[-1],[0],[1]] #define training x 

Y_train = [-1,0,1] #define training y

model.fit(X_train,Y_train) #fit the lasso model of training x and training y

print(model.intercept_) #output is the intercept, which is the beta0 in the formula: y=beta1*x+beta0

print(model.coef_) #output is the coefficient, which is the beta1 in the formula: y=beta1*x+beta0

model.predict([[4]]) #testing a particular value of x; output is prediction
import matplotlib.pyplot as plt

plt.scatter(X_train,Y_train)

plt.plot(X_train, model.coef_*X_train + model.intercept_, linestyle='--')
#The following reiterates the first example, but adding a second feature.



from sklearn.linear_model import Lasso 

model = Lasso(alpha=0.1) 

X_train = [[3,-1],[0,0],[1,1]] 

Y_train = [-1,0,1] 

model.fit(X_train,Y_train) 

print(model.intercept_) 

print(model.coef_) 

model.predict([[0,1]]) 