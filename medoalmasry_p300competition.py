import matplotlib.pyplot as plt

from scipy.io import loadmat

import numpy as np

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

import seaborn as sns

%matplotlib inline
data = loadmat('../input/P300_NonP300_A_Train_Mod.mat')

data.keys()

Pdata = data['P300'][0]

NPdata = data['NonP300'][0]

print(Pdata.shape , NPdata.shape)
CARP300 = np.array([(mat.transpose()-mat.mean(axis=1)).transpose() for mat in Pdata])

CARNonP300 = np.array([(mat.transpose()-mat.mean(axis=1)).transpose() for mat in NPdata])

print(CARP300.shape , CARNonP300.shape)
P300Data = []

for m in CARP300:

    P300Data += [i.reshape(240,15,order='F').mean(axis=1) for i in np.split(m[:,10],2)]

P300Data = np.array(P300Data).transpose()



NonP300Data = []

for m in CARNonP300:

    NonP300Data += [i.reshape(240,15,order='F').mean(axis=1) for i in np.split(m[:,10],10)]

NonP300Data = np.array(NonP300Data).transpose()



print(P300Data.shape , NonP300Data.shape) 
DataTrain = np.concatenate((P300Data[:,:136],NonP300Data[:,:136]),axis=1).transpose()

DataTest = np.concatenate((P300Data[:,136:170],NonP300Data[:,136:170]),axis=1).transpose()

print(DataTrain.shape , DataTest.shape)
from sklearn.decomposition import PCA

pca = PCA()

proj = pca.fit_transform(DataTrain)

pcs = pca.components_.transpose()

projTest = np.dot(DataTest,pcs)

print(pcs.shape , proj.shape , projTest.shape)
gnb = GaussianNB()

scores = []

for numpcs in range(1,241):

    X_train = proj[:,:numpcs]

    X_test = projTest[:,:numpcs]

    y_train = np.concatenate((np.ones(136),np.ones(136)*(-1)),axis=0)

    y_test = np.concatenate((np.ones(34),np.ones(34)*(-1)),axis=0)

    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    scores += [accuracy_score(y_pred,y_test)]

plt.plot(scores)