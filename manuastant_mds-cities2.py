import pandas as pd

from sklearn import manifold

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
cities = pd.read_csv('../input/cities2.txt', sep=" ")

names = cities.iloc[:,0]

cities = cities.iloc[:,1:19]



for i in range(len(cities.index)):

    for j in range(i,len(cities.index)):

        cities.iloc[i,j] = cities.iloc[j,i]



print(cities.head())
loc2 = manifold.MDS(n_components=2, dissimilarity='precomputed')

pos = loc2.fit(cities).embedding_

plt.scatter(pos[:,0],pos[:,1])



for i, txt in enumerate(names):

    plt.annotate(txt, (pos[i,0], pos[i,1]))



plt.show()
loc3 = manifold.MDS(n_components=3, dissimilarity='precomputed')

pos = loc3.fit(cities).embedding_

x = pos[:,0]

y = pos[:,1] 

z = pos[:,2]



fig = plt.figure()

ax = fig.gca(projection='3d')

ax.scatter(x,y,z)



for xlab, ylab, zlab, name in zip(x,y,z,names):

    ax.text(xlab, ylab, zlab, name)



ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')



plt.show()
loc22 = manifold.smacof(cities,n_components=2)

pos = loc22[0]

plt.scatter(pos[:,0],pos[:,1])



for i, txt in enumerate(names):

    plt.annotate(txt, (pos[i,0], pos[i,1]))



plt.show()
loc32 = manifold.smacof(cities, n_components=3)

pos = loc32[0]

x = pos[:,0]

y = pos[:,1] 

z = pos[:,2]



fig = plt.figure()

ax = fig.gca(projection='3d')

ax.scatter(x,y,z)



for xlab, ylab, zlab, name in zip(x,y,z,names):

    ax.text(xlab, ylab, zlab, name)



ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')



plt.show()
stress2d = loc22[1]

stress3d = loc32[1]

ratio = stress2d/stress3d

print(f'Stress sin normalizar:\n2D: {stress2d}\n3D: {stress3d}\nRatio: {ratio}')