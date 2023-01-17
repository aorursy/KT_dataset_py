import os

from copy import deepcopy

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (16,9)

plt.style.use('ggplot')



# Read Data

data = pd.read_csv("../input/data_export_indonesia.csv")

data.head()

# Any results you write to the current directory are saved as output.
# Prepare Data and Plotting

f1 = data["TujuanNetto(Ton)"].values

f2 = data["NilaiFOB(RibuUS$)"].values

X = np.array(list(zip(f1, f2)))

plt.scatter(f1, f2, c='black')
# Euclidean Distance Caculator

def dist(a, b, ax=1):

    return np.linalg.norm(a - b, axis=ax)
k = 3

C = X[0:3]

plt.scatter(f1, f2, c="#050505")

plt.scatter(C[:,0], C[:,1], marker="*", s=100, c='g')
C_old = np.zeros(C.shape)



clusters = np.zeros(len(X))



error = dist(C, C_old, None)



count = 1

while (error !=0):

    for i in range(len(X)):

        distances = dist(X[i],C)

        cluster = np.argmin(distances)

        clusters[i] = cluster

    

    C_old = deepcopy(C)

    

    for i in range(k):

        points = [X[j] for j in range(len(X)) if clusters[j]==i]

        C[i] = np.mean(points, axis=0)

    error = dist(C, C_old, None)
colors = ['r', 'g', 'b', 'y', 'c', 'm']

fig, ax = plt.subplots()

for i in range(k):

        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])

        ax.scatter(points[:, 0], points[:, 1], s=100, c=colors[i])

ax.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='#050505')



hasil = np.array(list(zip(data["Negara"], clusters)))
#%% Output

hasil = [('Negara', data["Negara"].values),

         ('Cluster', clusters)]

output = pd.DataFrame.from_items(hasil)

output