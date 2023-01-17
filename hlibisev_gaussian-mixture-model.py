# Importing libraries

import numpy as np

import pandas as pd

from numpy.linalg import det,inv

import pystan as ps

from scipy.stats import multivariate_normal
def gmm(X,k,n_iter = 20):

    # Initial approximation

    N = X.shape[0] # Number of data

    m = X.shape[1] # Numper of features

    w = np.ones(k)/k # Coef of distributions

    µ = np.ones([k,m]) # Mathematical expectation

    S = np.array([np.eye(m) for _ in range(k)]) # Matrix of covariation

    G = np.ones([N,k]) 

    

    for j in range(k):

        µ[j] = X[np.random.choice(N)]

    

    

    # E-step

    for _ in range(n_iter):

        for j in range(k):

            G[:,j] = w[j]*multivariate_normal.pdf(X,µ[j],S[j])

        G = G/(G.sum(axis=1,keepdims=True))

        n = G.sum(axis = 0)



    # M-step

        w = n/N

        µ = G.T@X/n[:,None]

        for j in range(k):

            delta = X - µ[j,:] # N x m

            Rdelta = np.expand_dims(G[:,j], -1) * delta

            S[j] = Rdelta.T.dot(delta) / n[j]

    return(µ,S,w)



def gmm_predict(X,k,n_iter = 20):

    N = len(X)

    Class = np.zeros(N)

    µ,S,w = gmm(X,k,n_iter)

    G = np.ones([N,k])

    

    for j in range(k):

        G[:,j] = w[j]*multivariate_normal.pdf(X,µ[j],S[j])

    

    Class = G.argmax(axis=1)

    return(Class)
import numpy as np

import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture

from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice

import plotly.graph_objects as go



# Making data

X, y = datasets.make_blobs(n_samples=2000)



# Сlustering

color = gmm_predict(X,3,20)



# Сhart

fig = go.Figure(data=go.Scatter(

    x=X.T[0],

    y=X.T[1],

    mode='markers',

    marker=dict(color=color)))



fig.show()
# Toy data

X = np.concatenate([np.random.randn(100)*2+12,np.random.randn(300)*5-5])
import plotly.express as px

fig = go.Figure(data=[go.Histogram(x=X,histnorm='probability',opacity=0.5)])

fig.show()
eccmodel = """

data {

    int<lower=1> m; // number of mixture components

    int<lower=1> N; // number of data points

    real y[N]; // observations

}

parameters {

    vector[m] mu;

    vector<lower=0>[m] o;

    simplex[m] w;

    vector[m] lambda;

}



model {

    o ~ cauchy(0, 5);

    w ~ dirichlet(lambda);

    mu ~ normal(0, 20);

    for (n in 1:N) {

        target += log_mix(w[1],

            normal_lpdf(y[n] | mu[1], o[1]),

            normal_lpdf(y[n] | mu[2], o[2]));

    }

}

"""

sm1 = ps.StanModel(model_code=eccmodel)
data = {"m":2,"N":len(X),"y":X}

fit = sm1.sampling(data=data, iter=1000, chains=1)
fit.plot()
def stan_predict(X,fit):

    N = len(X)

    Class = np.zeros(N)

    µ,S,w = fit["mu"].mean(axis=0),fit["o"].mean(axis=0),fit["w"].mean(axis=0)

    G = np.ones([N,2])

    for j in range(2):

        G[:,j] = w[j]*multivariate_normal.pdf(X,µ[j],S[j])

    

    Class = G.argmax(axis=1)

    return(Class)
color = stan_predict(X,fit);
df = pd.DataFrame([X,color]).T

fig = px.histogram(df, x=0, color=1)

fig.show()