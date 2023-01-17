# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
from scipy import optimize

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Read CSV
df = pd.read_csv("../input/countries of the world.csv", decimal = ',').fillna(0)

# Region encoding
le = preprocessing.LabelEncoder()
df['Region'] = le.fit_transform(df['Region'])

# Scaling data
scaler = MinMaxScaler()
df[df.columns[2:]] = scaler.fit_transform(df[df.columns[2:]])

# Prepare Dataset
X = df[df.columns[2:]].values
y = df[df.columns[1]].values
# Principal Component Analysis

# Compute empirical mean values
mean = X.mean(axis=0)

# Compute Covariance Matrix
SIGMA = np.ones((len(mean),len(mean)))
for i in range(0,len(mean)):
    for j in range(0,len(mean)):
        SIGMA[i,j] = (np.dot(X[:,i],X[:,j]) / len(X)) - (mean[i]*mean[j])
        
# w, v = LA.eig(SIGMA)

args=('vTappo',SIGMA)
def J_values(w, *args):
    tappo,s=args
    lambdaI = w * np.identity(len(w))
    return np.square(np.square(LA.det(s - lambdaI)))

w0 = np.random.randn(len(mean))
w_ast = optimize.fmin_cg(J_values,w0,args=args,gtol=1.0e-10)

def J_vectors(v, *args):
    w_ast_i,SIGMA=args
    W = SIGMA - (w_ast_i * np.identity(len(v)))
    return np.square(np.sum(np.dot(W,v)))
V = []
w_ast[::-1].sort()

for i in range(0, len(w_ast)):
    v0 = np.random.randn(len(mean))
    args = (w_ast[i], SIGMA)
    V.append(optimize.fmin_cg(J_vectors,v0,args=args))
# Mapping into new 2 dimensional space
W = np.hstack((V[0].reshape(len(v0),1),V[1].reshape(len(v0),1)))

red_X = np.dot(X, W)
# Plot
import matplotlib.pyplot as plt
palette = ['red', 'green', 'blue', 'gray', 'violet', 'black', 'yellow', 'brown', 'coral', 'darkgreen', 'cyan']
colors = list(map(lambda x: palette[x], y))
fig = plt.figure(figsize=(10, 10)) 
ax = fig.subplots()
ax.scatter(red_X[:,0], red_X[:,1], c = colors)

for i, text in enumerate(df['Country']):
    ax.annotate(text, (red_X[i,0], red_X[i,1]))
    
plt.show()
