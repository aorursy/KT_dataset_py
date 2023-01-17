#importar librerias

import numpy as np

import plotly.express as px

from matplotlib import pyplot as plt

import pandas as pd

import plotly.graph_objects as go
#importar documento excel

dataset = pd.read_csv('../input/admission-predictcsv/Admission_Predict.csv')

dataset
def linear_cost(X, y, theta):

    m, _ = X.shape

    h = np.matmul(X, theta)

    sq = (y - h) ** 2

    return sq.sum() / (2 * m)

#REGULARIZADA

def linear_cost_reg(X, y, theta, regParam_lambda):

    m, _ = X.shape

    h = np.matmul(X, theta)

    sq = (y - h) ** 2

    reg = theta ** 2

    return (sq.sum() + np.sum(regParam_lambda * reg))/ (2 * m) 

def linear_cost_derivate(X, y, theta):

    h = np.matmul(X, theta)

    m, _ = X.shape

    return np.matmul((h - y).T, X).T / m

#REGULARIZADA

def linear_cost_derivate_reg(X, y, theta, regParam_lambda):

    h = np.matmul(X, theta)

    m, _ = X.shape

    reg = (regParam_lambda / m) * theta.sum()

    return ((np.matmul((h - y).T, X).T) + reg) / m
def gradient_descent(

        X,

        y,

        theta_0,

        cost,

        cost_derivate,

        alpha,

        treshold,

        max_iter):

    theta, i = theta_0, 0

    costs = []

    gradient_norms = []

    while np.linalg.norm(cost_derivate(X, y, theta)) > treshold and i < max_iter:

        theta -= alpha * cost_derivate(X, y, theta)

        i += 1

        costs.append(cost(X, y, theta))

        gradient_norms.append(cost_derivate(X, y, theta))

    return theta, costs, gradient_norms

def gradient_descent_regularizado(

        X,

        y,

        theta_0,

        cost,

        cost_derivate,

        dato_lambda,

        alpha,

        treshold,

        max_iter):

    theta, i = theta_0, 0

    costs = []

    gradient_norms = []

    while np.linalg.norm(cost_derivate(X, y, theta, dato_lambda)) > treshold and i < max_iter:

        theta -= alpha * cost_derivate(X, y, theta, dato_lambda)

        i += 1

        costs.append(cost(X, y, theta, dato_lambda))

        gradient_norms.append(cost_derivate(X, y, theta, dato_lambda))

    return theta, costs, gradient_norms
# extraccion de datos

gre = dataset.iloc[:, 1].values

toefl = dataset.iloc[:, 2].values

uni_rating = dataset.iloc[:, 3].values

sop = dataset.iloc[:, 4].values

lor = dataset.iloc[:, 5].values

cgpa = dataset.iloc[:, 6].values

research = dataset.iloc[:, 7].values



Y = dataset.iloc[:, 8].values

Y.shape = (Y.size,1)
x_data = dataset.iloc[:, 6].values

y_data = dataset.iloc[:, 8].values

fig1 = px.scatter(x=x_data, y=y_data, labels={'x':'CGPA Score', 'y':'Chance of Admit'})

fig1.show()
x_data = dataset.iloc[:, 3].values

y_data = dataset.iloc[:, 8].values

fig2 = px.scatter(x=x_data, y=y_data, labels={'x':'University Rating', 'y':'Chance of Admit'})

fig2.show()
x_data = dataset.iloc[:, 3].values

y_data = dataset.iloc[:, 6].values

fig2 = px.scatter(x=x_data, y=y_data, labels={'x':'University Rating', 'y':'CGPA Score'})

fig2.show()
# datos en matriz

gre_mtx = np.array(gre)

gre_mtx.shape = (gre_mtx.size,1)



toefl_mtx = np.array(toefl)

toefl_mtx.shape = (toefl_mtx.size,1)



uni_mtx = np.array(uni_rating)

uni_mtx.shape = (uni_mtx.size,1)



sop_mtx = np.array(sop)

sop_mtx.shape = (sop_mtx.size,1)



lor_mtx = np.array(lor)

lor_mtx.shape = (lor_mtx.size,1)



cgpa_mtx = np.array(cgpa)

cgpa_mtx.shape = (cgpa_mtx.size,1)



research_mtx = np.array(research)

research_mtx.shape = (research_mtx.size,1)
# datos en matrix con columna 1

one = np.ones(gre_mtx.size)

one.shape = (gre_mtx.size,1)

one



gre_ready = np.hstack((one, gre_mtx))

toefl_ready = np.hstack((one, toefl_mtx))

uni_rating_ready = np.hstack((one, uni_mtx))

sop_ready = np.hstack((one, sop_mtx))

lor_ready = np.hstack((one, lor_mtx))

cgpa_ready = np.hstack((one, cgpa_mtx))

research_ready = np.hstack((one, research_mtx))
cgpa_ready = np.hstack((one, cgpa_mtx))
m, n = cgpa_ready.shape



theta_0 = np.random.rand(n, 1)

theta, costs, gradient_norms = gradient_descent(

    cgpa_ready,

    Y,

    theta_0,

    linear_cost,

    linear_cost_derivate,

    alpha=0.01,

    treshold=0.001,

    max_iter=200000

)



print ('THETA:', theta)



# Plot training data

#plt.scatter(cgpa_ready[:, 1], Y)

#plt.plot(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='red')



# GRAFICA DE CODO

plt.plot(np.arange(len(costs)), costs)



plt.show()
# Plot training data

plt.scatter(cgpa_ready[:, 1], Y)

plt.scatter(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='red')
m, n = cgpa_ready.shape

dato_lambda = 0.5





theta_0 = np.random.rand(n, 1)

theta, costs, gradient_norms = gradient_descent_regularizado(

    cgpa_ready,

    Y,

    theta_0,

    linear_cost_reg,

    linear_cost_derivate_reg,

    dato_lambda,

    alpha=0.01,

    treshold=0.001,

    max_iter=200000

)



print ('THETA:', theta)



# Plot training data

#plt.scatter(cgpa_ready[:, 1], Y)

#plt.plot(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='red')



# GRAFICA DE CODO

plt.plot(np.arange(len(costs)), costs)



plt.show()
# Plot training data

plt.scatter(cgpa_ready[:, 1], Y)

plt.scatter(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='brown')
cgpa_ready = np.hstack((one, cgpa_mtx, (cgpa_mtx ** 2) ))
m, n = cgpa_ready.shape



theta_0 = np.random.rand(n, 1)

theta, costs, gradient_norms = gradient_descent(

    cgpa_ready,

    Y,

    theta_0,

    linear_cost,

    linear_cost_derivate,

    alpha=0.0001,

    treshold=0.001,

    max_iter=200000

)



print ('THETA:', theta)



# Plot training data

#plt.scatter(cgpa_ready[:, 1], Y)

#plt.plot(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='red')



# GRAFICA DE CODO

plt.plot(np.arange(len(costs)), costs)



plt.show()
# Plot training data

plt.scatter(cgpa_ready[:, 1], Y)

plt.scatter(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='red')
m, n = cgpa_ready.shape

dato_lambda = 10



theta_0 = np.random.rand(n, 1)

theta, costs, gradient_norms = gradient_descent_regularizado(

    cgpa_ready,

    Y,

    theta_0,

    linear_cost_reg,

    linear_cost_derivate_reg,

    dato_lambda,

    alpha=0.0001,

    treshold=0.001,

    max_iter=200000

)



print ('THETA:', theta)



# Plot training data

#plt.scatter(cgpa_ready[:, 1], Y)

#plt.plot(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='red')



# GRAFICA DE CODO

plt.plot(np.arange(len(costs)), costs)



plt.show()
# Plot training data

plt.scatter(cgpa_ready[:, 1], Y)

plt.scatter(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='brown')
cgpa_u_ready = np.hstack((one, cgpa_mtx, (uni_mtx ** 2), (uni_mtx ** 3) ))
m, n = cgpa_u_ready.shape



theta_0 = np.random.rand(n, 1)

theta, costs, gradient_norms = gradient_descent(

    cgpa_u_ready,

    Y,

    theta_0,

    linear_cost,

    linear_cost_derivate,

    alpha=0.00056,

    treshold=0.001,

    max_iter=200100

)



print ('THETA:', theta)



# Plot training data

#plt.scatter(cgpa_ready[:, 1], Y)

#plt.plot(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='red')



# GRAFICA DE CODO

plt.plot(np.arange(len(costs)), costs)



plt.show()
# Plot training data

plt.scatter(cgpa_ready[:, 1], Y)

plt.scatter(cgpa_ready[:, 1], np.matmul(cgpa_u_ready, theta), color='red')
cgpa_u_ready = np.hstack((one, cgpa_mtx, (uni_mtx ** 2), (uni_mtx ** 3) ))
m, n = cgpa_u_ready.shape

dato_lambda = 150



theta_0 = np.random.rand(n, 1)

theta, costs, gradient_norms = gradient_descent_regularizado(

    cgpa_u_ready,

    Y,

    theta_0,

    linear_cost_reg,

    linear_cost_derivate_reg,

    dato_lambda,

    alpha=0.00056,

    treshold=0.001,

    max_iter=200100

)



print ('THETA:', theta)



# Plot training data

#plt.scatter(cgpa_ready[:, 1], Y)

#plt.plot(cgpa_ready[:, 1], np.matmul(cgpa_ready, theta), color='red')



# GRAFICA DE CODO

plt.plot(np.arange(len(costs)), costs)



plt.show()
# Plot training data

plt.scatter(cgpa_ready[:, 1], Y)

plt.scatter(cgpa_ready[:, 1], np.matmul(cgpa_u_ready, theta), color='red')