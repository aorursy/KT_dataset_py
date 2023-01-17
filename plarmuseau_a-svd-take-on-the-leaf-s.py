# train 990 leaves

# test 594 samples



import numpy as np

import pandas as pd

from numpy.linalg import inv



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

labels = train.species

labelid = train.id

testid =test.id

train = train.drop(['species', 'id'], axis=1)

test = test.drop(['id'], axis=1)

#A=train

A=train.transpose()



# singular value decomposition

U,s,V=np.linalg.svd(A,full_matrices=False)

# reconstruct

S=np.diag(s)

Q=test.transpose()

iS=inv(S)

US=np.dot(U,iS)

Qtemp=np.dot(Q.transpose(),US)

simila=np.dot(Qtemp,V)/np.dot(np.abs(Qtemp),np.abs(V))

for xya in range(0,594):

  for xyz in range (len(V)):

    simila=np.dot(Qtemp[xya,:],V[:,xyz])/np.dot(np.abs(Qtemp[xya,:]),np.abs(V[:,xyz]))*100

    if simila>70:

      print(testid[xya],labels[xyz],labelid[xyz], round(simila,1),"%" ) 
import matplotlib

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

matplotlib.style.use('fivethirtyeight')

from scipy import ndimage as ndi



matplotlib.rcParams['font.size'] = 5

matplotlib.rcParams['figure.figsize'] = (11,8)   



# learned from Selfisch Gene.

numImages = 1584

shapesMatrix = np.zeros((2,numImages))

listOfImages = []

for k in range(numImages):

    imageFilename = '../input/images/' + str(k+1) + '.jpg'

    currImage = mpimg.imread(imageFilename)

    shapesMatrix[:,k] = np.shape(currImage)

    listOfImages.append(currImage)



# create a large 3d array with all images

maxShapeSize = shapesMatrix.max(axis=1)

for k in range(len(maxShapeSize)):

    if maxShapeSize[k] % 2 == 0:

        maxShapeSize[k] += 311

    else:

        maxShapeSize[k] += 310

    

fullImageMatrix3D = np.zeros(np.hstack((maxShapeSize,

                                        np.shape(shapesMatrix[1]))).astype(int),dtype=np.dtype('u1'))

destXc = (maxShapeSize[1]+1)/2; destYc = (maxShapeSize[0]+1)/2

for k, currImage in enumerate(listOfImages):

    Yc, Xc = ndi.center_of_mass(currImage)

    Xd = destXc - Xc; Yd = destYc - Yc

    fullImageMatrix3D[round(Yd):round(Yd)+np.shape(currImage)[0],

                      round(Xd):round(Xd)+np.shape(currImage)[1],k] = currImage



k=1    

for xya in range(0,594):

  for xyz in range (len(V)):

    simila=np.dot(Qtemp[xya,:],V[:,xyz])/np.dot(np.abs(Qtemp[xya,:]),np.abs(V[:,xyz]))*100

    if simila>70:

        plt.subplot(10,6,k); plt.imshow(fullImageMatrix3D[:,:,testid[xya]-1], cmap='gray'); plt.axis('off'); plt.title('search id '+str(testid[xya]))

        plt.subplot(10,6,k+1); plt.imshow(fullImageMatrix3D[:,:,labelid[xyz]-1], cmap='gray'); plt.axis('off'); plt.title(labels[xyz]+str(labelid[xyz]))

        k+=2        

plt.tight_layout()       
import matplotlib.pylab as plt

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_aspect('equal')

plt.imshow(train, interpolation='nearest', cmap=plt.cm.ocean)

plt.colorbar()

plt.show()



fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_aspect('equal')

plt.imshow(test, interpolation='nearest', cmap=plt.cm.ocean)

plt.colorbar()

plt.show()