#Now we are switching from regression problems to classification problems. 
#Don't be confused by the name "Logistic Regression"; it is named that way for 
#historical reasons and is actually an approach to classification problems, not regression problems.

#for more info on naming convention 
# Goto: https://www.graphpad.com/guides/prism/7/curve-fitting/index.htm?reg_the_term_logistic.htm
import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import zipfile
import matplotlib.pyplot as plt

import plotly
plotly.tools.set_credentials_file(username='enter_your_details_here', api_key='8tUzNDbRW7G2wlH4js1P')
import plotly.plotly as py
import plotly.graph_objs as go
df = pd.read_csv('../input/week_3-ex_2.txt',header=None,names=('Exam 1 score','Exam 2 score','Status'))
print(df.head())
print(df.keys())
print(df.shape)
m, n = df.shape
X = df.iloc[:,0:2]
y = df.iloc[:,2]
print(X.head())
print(y.head())
print(type(X))
print(type(y))
admitted = np.where(y[:] == 1)
not_admitted = np.where(y[:] == 0)

print(type(admitted))
print(admitted)
# Create a trace
trace1 = go.Scatter(
    x = X['Exam 1 score'].iloc[admitted],
    y = X['Exam 2 score'].iloc[admitted],
    name = 'Admitted',  
    marker={'color': 'blue', 'symbol': 100},
    mode = 'markers',
)

trace2 = go.Scatter(
    x = X['Exam 1 score'].iloc[not_admitted],
    y = X['Exam 2 score'].iloc[not_admitted],
    name = 'Not Admitted',   
    marker={'color': 'red', 'symbol': 104},
    mode = 'markers'
   
)
#data = [trace1,trace2]

# Plot and embed in ipython notebook!
#py.iplot(data, filename='basic-scatter')


data=go.Data([trace1,trace2])
layout=go.Layout(title="Admission statistics", xaxis={'title':'Exam 1 score'}, yaxis={'title':'Exam 2 score'})
figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='pyguide_1')
def sigmoid(z):
    return 1/(1+np.exp(-z))

# np.exp is already vectorized function
# for more info 
# Goto: https://stackoverflow.com/questions/42594695/how-to-apply-a-function-map-values-of-each-element-in-a-2d-numpy-array-matrix?rq=1
X = np.column_stack((X['Exam 1 score'],X['Exam 2 score']))
X = np.hstack((np.full([m,1],1), X))
y = y[:,np.newaxis]
# Goto: https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
print(type(X))
print(X.shape)

print(type(y))
print(y.shape)

theta = np.full([(n-1)+1, 1],0)
print(theta.shape)
theta = np.array([[-24], [0.2], [0.2]])
def computeCost(X, y, theta):    
    h = sigmoid(np.dot(X, theta))    
    return (-1)*(np.dot(y.T,np.log(h)) + np.dot(1-y.T,np.log(1-h)))/m
# initial cost
computeCost(X,y,theta)
def gradientDescent(theta, X, y, alpha, iterations):
    
    # theta_vals, J_vals stores the intermediate values of theta and J during optimiztion usng gradient descent
    theta_vals=np.full([1,n],0)
    J_vals = np.full([1],computeCost(X,y,theta))
    
    for _ in range(iterations):        
        theta = theta - (alpha/m)*(np.dot(X.T, (sigmoid(np.dot(X, theta)) - y)))
        theta_vals = np.vstack((theta_vals,theta.T))
        J_vals = np.vstack((J_vals,computeCost(X, y, theta)))
    return [theta_vals,J_vals]

(theta_vals,J_vals) = gradientDescent(theta, X, y, 0.1,400000)
theta = np.reshape(theta_vals[-1],[n,1])
print("Theta: ",theta)
print(computeCost(X, y, theta))
# print the decision boundary
positives = np.where(y[:,0] == 1);
negatives = np.where(y[:,0] == 0);
plt.scatter(X[positives,1], X[positives,2], marker='+', label='Admitted')
plt.scatter(X[negatives,1], X[negatives,2], marker='o', label='Not admitted')
px = np.array([np.min(X[:,1])-2, np.max(X[:,2])+2])
py = (-1 / theta[2]) * (theta[1]*px + theta[0])
plt.plot(px, py)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()
a=np.array([1,45,85])
b=sigmoid(np.dot(a,theta))
b
