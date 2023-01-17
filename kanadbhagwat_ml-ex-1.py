# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/kaggle/input/ex1data1/ex1data1.txt', header = None)
df.head()
# Label the columns and add a column of ones for X(0)
df.columns = ['Population', 'Profit']
df.insert(0, 'x_zero' , np.ones(len(df)), True)  
df.head()
sns.scatterplot(x = 'Population', y = 'Profit', data = df)
# Convert dataframe columns to numpy arrays X and Y for easy manipulation and computation
x = np.array([df.x_zero, df.Population]).T
y = np.array([df.Profit]).T # create output vector Y with third column of df. Transpose to make it 96 x 1 vector
print('X is of dimension: ', x.shape, '\nY is of dimension: ', y.shape) # print dimensions of X and Y to confirm they are of the correct shape (\n is used to print Y in next line)
# Initialise parameter vector theta
theta = np.zeros([2,1])
print(theta)
def Compute_Cost(x, y, theta):
    m = len(y)
    squared_error = np.square(np.dot(x, theta) - y)
    J = np.sum(squared_error) / (2 * m)
    return J
#theta = np.array([[-1], [2]])
Compute_Cost(x, y,theta)
# Implement Gradient Descent

def Grad_Descent (x, y, theta, alpha, iterations):
    ## This functions runs gradient descent
   
    m = len(y) # initialise m = no of training examples
    J = np.zeros([iterations,1])
    for i in range(iterations):
        temp1 = np.dot(x, theta) - y # error = h_theta - y
        temp2 = np.multiply(temp1,x) # (h_theta - y).* x
        temp3 = (alpha / m) * np.sum(temp2, axis = 0, keepdims = True) # sum over m training examples, keepdims=true to prevent 0D array
        theta = theta - temp3.T # simultaneous update of theta
        J[i] = Compute_Cost(x, y, theta)
    
    return theta, J
init_theta = np.array([[0],[0]]) # Initialise theta to [0,0]
alpha = 0.01
iterations = 1500
theta_final, J = Grad_Descent (x, y, init_theta, alpha, iterations)
print('Theta by Gradient Descent = ', theta_final.T)
print('Minimum Cost with Gradient Descent = ', J[1499])
plt.plot(J, 'r-')
plt.xlabel('Iterations')
plt.ylabel('Cost (J)')
plt.show()
# Linear Regression model
regr = LinearRegression().fit(x,y)
theta_regr = regr.coef_.T
J_regr = Compute_Cost(x, y, theta_regr)
print('Theta using Scikit model =  ', theta_regr)
print('Cost using Scikit model = ', J_regr)
print(profit_gd_predict.shape)
print(profit_sk_predict.shape)
population = np.array(df['Population'])
population.shape = (97,1)
print(population.shape)
profit_gd_predict = np.dot(x, theta_final)
profit_sk_predict = np.dot(x, theta_regr)
plt.plot(population, profit_gd_predict,'r')
plt.plot(population, profit_sk_predict,'g')
plt.scatter(population,y)

# SGD (Stochastic Gradient Descent) Regressor 
sgd_reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter = 1500, tol = 1e-3))
sgd_reg.fit(x,y)
