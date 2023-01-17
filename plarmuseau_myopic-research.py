#load with pandas, manipulate with numpy, plot with matplotlib

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



#ML - we will classify using a naive xgb with stratified cross validation

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss







#filenames

inputFolder = "../input/"

trainSet = 'train.json'

#testSet = 'test.json'

subName = 'iceberg-svd-xgb-3fold.csv'

#load data

trainDF = pd.read_json(inputFolder+trainSet)

#testDF = pd.read_json(inputFolder+testSet)
trainDF.head(15)
#get numpy arrays for train/test data, prob there is a more pythonic approach

band1 = trainDF['band_1'].values

im1 = np.zeros((len(band1),len(band1[0])))

for j in range(len(band1)):

    im1[j,:]=np.asarray(band1[j])

    

band2 = trainDF['band_2'].values

im2 = np.zeros((len(band2),len(band2[0])))

for j in range(len(band2)):

    im2[j,:]=np.asarray(band2[j])

    

import cv2

from skimage import filters

from skimage import data, exposure





image=np.reshape(im1[13,:],(75,75))

hsobel_text = filters.sobel_h(image)

camera_equalized = exposure.equalize_hist(image)

val = filters.threshold_otsu(image)



print(val+5)

fig, ax = plt.subplots(1,2) 

ax[0].imshow(image)

ax[1].imshow(image, cmap='nipy_spectral', interpolation='nearest')



fig, ax = plt.subplots(1,2) 

ax[0].imshow(hsobel_text, cmap='gray', interpolation='nearest')

ax[1].imshow(hsobel_text, cmap='nipy_spectral', interpolation='nearest')



fig, ax = plt.subplots(1,2) 

ax[0].imshow(camera_equalized, cmap='gray', interpolation='nearest')

ax[1].imshow(camera_equalized, cmap='nipy_spectral', interpolation='nearest')



fig, ax = plt.subplots(1,2) 

print(image.std(),image.mean(),image.var())

ax[0].imshow( image*(image > (1+image.std()/image.mean()*2.5) * image.mean() ) , cmap='nipy_spectral', interpolation='nearest')

ax[1].imshow(image>val+4, cmap='gray', interpolation='nearest')
U1,s1,V1  = np.linalg.svd(im1,full_matrices = 0)

#U2,s2,V2  = np.linalg.svd(im2,full_matrices = 0)

print(U1[:,:100].shape,V1.shape)
from sklearn.metrics.pairwise import cosine_similarity





for rank in range(3,50,3):

    im1cs=cosine_similarity(U1[:,:rank],V1[:rank,:].T)

    image=np.reshape(im1cs[13,:],(75,75))

    fig, ax = plt.subplots(1,3) 

    ax[0].imshow(image)

    ax[1].imshow(image, cmap='nipy_spectral', interpolation='nearest')

    ax[2].imshow(image, cmap='gray', interpolation='nearest')



print(np.reshape(im1[13,:],(75,75)))

im1ce = exposure.equalize_hist(im1)

U1,s1,V1  = np.linalg.svd(im1ce,full_matrices = 0)
for rank in range(3,50,3):

    im1cs=cosine_similarity(U1[:,:rank],V1[:rank,:].T)

    image=np.reshape(im1cs[13,:],(75,75))

    fig, ax = plt.subplots(1,3) 

    ax[0].imshow(image)

    ax[1].imshow(image, cmap='nipy_spectral', interpolation='nearest')

    ax[2].imshow(image, cmap='gray', interpolation='nearest')
from sklearn.preprocessing import normalize

def distanc(X,Y):

    Z=X

    for yi in range(0,len(X)):

        Z[yi]=angle_between((X[yi],Y[yi],0),(1,0,0))

    return Z #np.reshape(Z,(75,75))



def unit_vector(vector):

    """ Returns the unit vector of the vector.  """

    return vector / np.linalg.norm(vector)



def angle_between(v1, v2):

    """ Returns the angle in radians between vectors 'v1' and 'v2'::



            >>> angle_between((1, 0, 0), (0, 1, 0))

            1.5707963267948966

            >>> angle_between((1, 0, 0), (1, 0, 0))

            0.0

            >>> angle_between((1, 0, 0), (-1, 0, 0))

            3.141592653589793

    """

    v1_u = unit_vector(v1)

    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



ima=im1

for xi in range(0,len(im1)):

    xi1=np.reshape(im1[xi,:],(75,75))

    xi2=np.reshape(im2[xi,:],(75,75))

    ima[xi]=distanc(im1[xi,:],im2[xi,:])
U1,s1,V1  = np.linalg.svd(ima,full_matrices = 0)



for rank in range(3,50,3):

    im1cs=cosine_similarity(U1[:,:rank],V1[:rank,:].T)

    image=np.reshape(im1cs[13,:],(75,75))

    fig, ax = plt.subplots(1,3) 

    ax[0].imshow(image)

    ax[1].imshow(image, cmap='nipy_spectral', interpolation='nearest')

    ax[2].imshow(image, cmap='gray', interpolation='nearest')