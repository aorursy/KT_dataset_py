import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.tools import FigureFactory as ff

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/data.csv')
data.head()
data = data[['Video Uploads', 'Subscribers', 'Video views']]
def plot_line(X, y, theta):
    p1 = go.Scatter(x = X, y = y,mode='markers', marker=dict(color='black'))
    p2 = go.Scatter(x=np.array([0, max(X)]), 
                    y=np.array([[1,0],[1,max(X)]]).dot(theta),
                    mode='lines', line=dict(color='blue', width=3))
    fig = go.Figure(data=[p1, p2])
    py.iplot(fig)
    
def plot(X, y, mode='markers', title='Plot title', x_axis='X axis', y_axis='Y axis'):
    p1 = go.Scatter(x = X, y = y,mode=mode, marker=dict(color='black'))
    layout = go.Layout(
                    title=title,
                    xaxis=dict(title=x_axis),
                    yaxis=dict(title=y_axis)
                )
    fig = go.Figure(data=[p1], layout=layout)
    py.iplot(fig)
    
def plot_scatter_matrix(data):
    fig = ff.create_scatterplotmatrix(data, height=1000, width=1000, title='Scatterplot Matrix')
    py.iplot(fig)
plot_scatter_matrix(data)
data = data[['Subscribers','Video views']].apply(pd.to_numeric, errors='coerce').dropna()
X_train, y_train= data['Subscribers'], data['Video views']
def normalize(matrix):
    means = matrix.mean(axis=0)
    maxs = matrix.max(axis=0)
    mins = matrix.min(axis=0)
    return ((matrix - means) / (maxs - mins),means, maxs,mins)

def normalize_data(matrix, means, maxs, mins):
    return (matrix - means) / (maxs - mins)
matrix_train = np.column_stack((X_train,y_train))
matrix_train_norm, means, maxs, mins = normalize(matrix_train)
X_train = matrix_train_norm[:,0]
y_train = matrix_train_norm[:,1]
X_train.shape, y_train.shape 
def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * sum(np.square((np.dot(X,theta) - y)))    
def gradient_descent(X, y, alpha=0.1, num_iters=100):
    theta = np.zeros(2)
    X = np.column_stack((np.ones(X.shape[0]),X))
    m = len(y)
    J_hist = np.zeros(num_iters)
    theta_hist = np.zeros((num_iters,2))
    for i in range(0,num_iters):
        prediction = X.dot(theta)
        
        theta = theta - alpha*1/m * X.T.dot((prediction-y))
        J_hist[i] = cost_function(X, y, theta)
        theta_hist[i] = theta.T
        diff = 1 if i == 0 else J_hist[i-1] - J_hist[i]
    return (theta, J_hist, theta_hist)
theta_grad, J_hist, theta_hist = gradient_descent(X_train, y_train, 0.2,3000)
theta_grad_large, J_hist_large, theta_hist_large = gradient_descent(X_train, y_train,10,19)
theta_grad_small, J_hist_small, theta_hist_small = gradient_descent(X_train, y_train, 0.02,5000)

print(theta_grad)
plot(np.arange(len(J_hist)),J_hist,mode='lines'
     , title='Cost historic with learning rate = 0.2'
     , x_axis='number of iteration'
     , y_axis='Cost')
plot(np.arange(len(J_hist_large)),J_hist_large,mode='lines'
     , title='Cost historic with learning rate = 10'
     , x_axis='number of iteration'
     , y_axis='Cost')
plot(np.arange(len(J_hist_small)),J_hist_small,mode='lines'
     , title='Cost historic with learning rate = 0.02'
     , x_axis='number of iteration'
     , y_axis='Cost')
def normal_equation(X, y):
    X = np.column_stack((np.ones(X.shape[0]),X))
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
theta_norm = normal_equation(X_train,y_train)
print(theta_norm)
from sklearn import linear_model
regr = linear_model.LinearRegression(normalize=True)

regr.fit(X_train.reshape(-1, 1), y_train)

theta_sklearn = np.array([regr.intercept_,regr.coef_])
print('Theta for gradient descent : ',theta_grad)
print('Theta for normal equation  : ',theta_norm)
print('Theta for sklearn          : ',theta_sklearn)
plot_line(X_train,y_train,theta_grad)
plot_line(X_train,y_train,theta_norm)
plot_line(X_train,y_train,theta_sklearn)