import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data=pd.read_csv('../input/prmlassignment/Dataset3.csv',header=None)

#data=pd.read_csv('../input/prml-dataset/Dataset3.csv')

data.shape

data.head()
data.columns=['x','y']

data.plot.scatter(x='x',y='y');

data.head()
#finding mean

m=np.mean(data)
#centering the data

centeredData=data-m

print(centeredData)
centeredData.plot.scatter(x='x' , y='y');

centeredData.mean(),m
#covarianceMatrix=np.cov(centeredData.T)

#find covariance matrix

covarianceMatrix=np.cov(data.T)

print(covarianceMatrix)
#find eigen values and eigen vectoes

from numpy.linalg import eig

evalues,evectors=eig(covarianceMatrix)

print(evectors)

print(evalues)
evectors=evectors.T

i = np.argsort(evalues)[::-1]

evalues=evalues[i]

evectors=evectors[i]

print(evalues,evectors)
ProjectData1=np.dot(data,evectors[0].T).reshape(1000,1)

ProjectData1=ProjectData1*evectors[0]

print(ProjectData1)
ProjectData2=np.dot(data,evectors[1].T).reshape(1000,1)

ProjectData2=ProjectData2*evectors[1]

print(ProjectData2)
ProjectData1.shape, centeredData.shape, centeredData.shape, evectors[0].shape
plt.xlabel('x axis')

plt.ylabel('y axis')

plt.scatter(x=ProjectData1[:,0], y=ProjectData1[:,1]);

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.scatter(x=ProjectData2[:,0], y=ProjectData2[:,1]);

#varvariance=np.var(centeredData)

variance=np.var(data)

variance1=np.var(ProjectData1)

variance2=np.var(ProjectData2)

Pvar1=(variance1/variance)*100

Pvar2=(variance2/variance)*100

print(Pvar1)

print(Pvar2)