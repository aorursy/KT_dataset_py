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
# Words 

W = np.array([0, 1, 2, 3, 4])



# D := remplacement des mots par un id 0:4

X = np.array([

    [0, 0, 1, 2, 2],

    [0, 0, 1, 1, 1],

    [0, 1, 2, 2, 2],

    [4, 4, 4, 4, 4],

    [3, 3, 4, 4, 4],

    [3, 4, 4, 4, 4]

])



W = np.array([0, 1, 2, 3, 4, 5, 6])

X = np.array([



    [0,0,0,1,1,1,1],

    [1,1,0,1,0,2,0],

    [0,1,1,0,0,1,2],

    [2,1,1,0,0,2,2],

    [3,2,3,4,3,2,4],

    [3,3,3,4,4,4,4],

    [3,2,2,3,3,3,4],

    [4,4,4,4,3,3,2]



])



N_D = X.shape[0]  # num of docs 

N_W = W.shape[0]  # num of words 

N_K = 2  # num of topics
X
# Dirichlet priors 

alpha = 1

gamma = 1



# Z := word topic assignment 

Z = np.zeros(shape=[N_D, N_W])



for i in range(N_D):

    for l in range(N_W):

        Z[i, l] = np.random.randint(N_K)  # randomly assign word's topic

# Pi := document topic distribution 

Pi = np.zeros([N_D, N_K])



for i in range(N_D):

    Pi[i] = np.random.dirichlet(alpha*np.ones(N_K))



# B := word topic distribution 

B = np.zeros([N_K, N_W])



for k in range(N_K):

    B[k] = np.random.dirichlet(gamma*np.ones(N_W))

print(Z)

for i in range(N_D):# for each doc

        for v in range(N_W):# for each word

            # Calculate params for Z             

            # HERE WE USE X !

            p_iv = np.exp(np.log(Pi[i]) + np.log(B[:, X[i, v]]))

            p_iv /= np.sum(p_iv)



            # Resample word topic assignment Z             

            Z[i, v] = np.random.multinomial(1, p_iv).argmax()

print(Z)

print(Pi)

# Sample from full conditional of Pi     # ----------------------------------     

# intermediate step, NOT USING X

for i in range(N_D):

    m = np.zeros(N_K)



    # Gather sufficient statistics         

    for k in range(N_K):

        m[k] = np.sum(Z[i] == k)



    # Resample doc topic dist.         

    Pi[i, :] = np.random.dirichlet(alpha + m)

print(Pi)
print(B)

# Sample from full conditional of B     # ---------------------------------     

for k in range(N_K):

    n = np.zeros(N_W)



    # Gather sufficient statistics         

    for v in range(N_W):

        for i in range(N_D):

            for l in range(N_W):

                # WE USE X HERE !

                n[v] += (X[i, l] == v) and (Z[i, l] == k)



    # Resample word topic dist.         

    B[k, :] = np.random.dirichlet(gamma + n)

print(B)

nb_iters = 2000

record_every = 100

B_rec = np.zeros(shape=[nb_iters//record_every,N_W])

B_rec
for it in range(2000):

    # Sample from full conditional of Z     # ---------------------------------     

    for i in range(N_D):# for each doc

        for v in range(N_W):# for each word

            # Calculate params for Z             

            p_iv = np.exp(np.log(Pi[i]) + np.log(B[:, X[i, v]]))

            p_iv /= np.sum(p_iv)



            # Resample word topic assignment Z             

            Z[i, v] = np.random.multinomial(1, p_iv).argmax()



    # Sample from full conditional of Pi     # ----------------------------------     

    for i in range(N_D):

        m = np.zeros(N_K)



        # Gather sufficient statistics         

        for k in range(N_K):

            m[k] = np.sum(Z[i] == k)



        # Resample doc topic dist.         

        Pi[i, :] = np.random.dirichlet(alpha + m)



    # Sample from full conditional of B     # ---------------------------------     

    for k in range(N_K):

        n = np.zeros(N_W)



        # Gather sufficient statistics         

        for v in range(N_W):

            for i in range(N_D):

                for l in range(N_W):

                    n[v] += (X[i, l] == v) and (Z[i, l] == k)



        # Resample word topic dist.         

        B[k, :] = np.random.dirichlet(gamma + n)

    if it % 100 == 0:

        B_rec[it//100,:] = B[1,:]
index = ['iter'+str(i)+'00' for i in range(len(B_rec))]

df = pd.DataFrame(B_rec, index=index)

df = df.reset_index()

df.head()
import plotly.graph_objects as go
df.columns = ['niter']+['w'+str(i) for i in range(7)]

df
fig = go.Figure(data=[go.Scatter(x=df.niter, y=df.w0),

                     go.Scatter(x=df.niter, y=df.w1),

                     go.Scatter(x=df.niter, y=df.w2),

                     go.Scatter(x=df.niter, y=df.w3),

                     go.Scatter(x=df.niter, y=df.w4),

                     go.Scatter(x=df.niter, y=df.w5),

                     go.Scatter(x=df.niter, y=df.w6)])

fig.show()
B
Z
Pi