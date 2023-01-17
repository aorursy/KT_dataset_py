import pandas as pd

from sklearn import manifold

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
datos = pd.read_csv('../input/similaridades-promediadas-entre-colores/color2.txt', sep=" ")



colores = datos.iloc[:,0]

datos = datos.iloc[:,1:15]



for i in range(len(datos.index)):

    for j in range(len(datos.index)):

            if(i != j):

                datos.iloc[i,j] = 1-datos.iloc[i,j]



print(datos.head())

loc2 = manifold.smacof(datos, n_components=2, metric=True)

pos = loc2[0]



x = pos[:,0]

y = pos[:,1]



fig = plt.figure()

ax = fig.gca(projection='rectilinear')

ax.scatter(x,y)



for xlab, ylab, color in zip(x,y,colores):

    ax.text(xlab, ylab, color)



ax.set_xlabel('x')

ax.set_ylabel('y')



plt.show()




loc3 = manifold.smacof(datos,n_components=3, metric=True)

pos = loc3[0]

x = pos[:,0]

y = pos[:,1] 

z = pos[:,2]



fig = plt.figure()

ax = fig.gca(projection='3d')

ax.scatter(x,y,z)



for xlab, ylab, zlab, color in zip(x,y,z,colores):

    ax.text(xlab, ylab, zlab, color)



ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')



plt.show()