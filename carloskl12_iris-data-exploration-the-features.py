# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from itertools import combinations
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv") #load the dataset
iris.drop('Id',axis=1,inplace=True) # Se elimina la columna no requerida

clases=list(iris.Species.unique()) #Se extraen las diferentes clases
colores=['orange', 'blue', 'green']
descriptores= list(iris)[:-1] #Se obtienen los diferentes descriptores
combinacionesD= list(combinations(descriptores,2))
nColumnas= 2
nFilas= len(combinacionesD)/nColumnas
fig = plt.figure(1,figsize=(15,4.5*nFilas))
for d in range(len(combinacionesD)):
    ax = fig.add_subplot(nFilas,nColumnas,d+1)
    ds=combinacionesD[d]#Par de descriptores
    for i in range( len(clases)):
        iris[iris.Species==clases[i]].plot(kind='scatter',x=ds[0],y=ds[1],ax=ax,color=colores[i], label=clases[i])
plt.show()
from scipy.stats import norm
IrisSetosa= iris[(iris['Species']=='Iris-setosa')]
IrisVersicolor= iris[(iris['Species']=='Iris-versicolor')]
IrisVirginica= iris[(iris['Species']=='Iris-virginica')]
dt=(IrisSetosa, IrisVersicolor,IrisVirginica)
lSigma=[]
lMu=[]
lMin=[]
lMax=[]
# Para cada dataFrame se halla sus máximos
for df in dt:
    lSigma.append(list(df.std()))
    lMu.append(list(df.mean()))
    lMin.append(list(df.min()))
    lMax.append(list(df.max()))
# Se procede a realizar las gŕaficas
fig = plt.figure(1,figsize=(15,9))
for d in range(len(descriptores)):
    sigma=[x[d] for x in lSigma]
    mu= [x[d] for x in lMu]
    maximo=[x[d] for x in lMax]
    maximo=max(maximo)
    minimo=[x[d] for x in lMin]
    minimo=min(minimo)
    rng = np.linspace(minimo,maximo,200)
    ax = fig.add_subplot(2,2,d+1)
    for i in range(len(mu)):
        dist=norm(mu[i],sigma[i])
        ax.plot(rng, (2*dist.pdf(rng)),label=clases[i])
    legend = ax.legend(loc='upper right', shadow=True, fontsize=12)
    ax.set_title('Descriptor: %s'%descriptores[d])
plt.show()
list(zip([1, 2], ('a' , 'b')))
clases=list(iris.Species.unique()) #Se extraen las diferentes clases
descriptores= list(iris)[:-1] #Se obtienen los diferentes descriptores
irisN= iris.copy()
descrip=irisN.describe()
for d , i in zip( descriptores, range(len (descriptores))):
    vMin=descrip[d]['min']
    vMax=descrip[d]['max']
    irisN[d]=(irisN[d]-vMin)/(vMax-vMin)

Ia= irisN[(iris['Species']=='Iris-setosa')]
Ib= irisN[(iris['Species']=='Iris-versicolor')]
Ic= irisN[(iris['Species']=='Iris-virginica')]
dt=(Ia, Ib, Ic)
lSigma=[]
lMu=[]
# Para cada dataFrame se halla sus máximos
for df in dt:
    lMu.append(list(zip(list(df.mean()), list(df.std()), descriptores)))
#Ordena los datos de varianza y media
#lMu.sort()
# Se procede a realizar las gŕaficas
fig = plt.figure(1,figsize=(12,9))
styles = (('r-','r.','r--'),('g-','g.','g--'),('b-','b.','b--'),('k-','k.','k--'))
rng = np.linspace(0.1,1,100)
#print (lMu[0])
for i, clase in enumerate(clases):
    ax = fig.add_subplot(3,1,i+1)
    mu=lMu[i]
    mu.sort()
    for k, d in enumerate(descriptores):
        dist=norm(mu[k][0],mu[k][1])
        lbStr='%s'%d
        ax.plot(rng, (2*dist.pdf(rng)),label=lbStr)
    legend = ax.legend(loc='upper right', shadow=True, fontsize=12)
    ax.set_title('Clase: %s'%clase)

plt.show()