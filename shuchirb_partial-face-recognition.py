# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

"""

Created on Fri Dec 29 17:01:06 2017



@author: Shuchi Rawat

"""



import numpy as np

import matplotlib.pyplot as plt



from sklearn.datasets import fetch_olivetti_faces

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV



def loadData():

   # facesDSAll=fetch_olivetti_faces()

   # targets=facesDSAll.target

   facesDSAll=np.load("../input/olivetti_faces.npy") 

   targets=np.load("../input/olivetti_faces_target.npy") 

   return facesDSAll,targets



def prepDS(facesDSAll):

    #flattenFacesDSAll= facesDSAll.images.reshape(facesDSAll.images.shape[0], facesDSAll.images.shape[1] * facesDSAll.images.shape[2])     # 64 X 64 = 4096

    flattenFacesDSAll= facesDSAll.reshape(facesDSAll.shape[0], facesDSAll.shape[1] * facesDSAll.shape[2])     # 64 X 64 = 4096

    return flattenFacesDSAll     



def prepTrainDS(flattenFacesDSAll,targets):

    train=flattenFacesDSAll[targets<30]

    colsTrain=train.shape[1]

    upperFaceTrain=train[:,:(colsTrain+1)//2]

    lowerFaceTrain=train[:,(colsTrain//2):]

    return upperFaceTrain,lowerFaceTrain





def randomTestDS(test):

    numFaces = test.shape[0]//10

    print(numFaces)

    faceIds = np.random.randint(0 , 100, size =numFaces)

    print(faceIds)

    print(test.shape)

    print(test)

    rndTest = test[faceIds, :] 

    return rndTest

    

def prepTestDS(flattenFacesDSAll,targets):

    test=flattenFacesDSAll[targets>=30]

    rndTest=randomTestDS(test)

    colsTest=rndTest.shape[1]

    upperFaceTest=rndTest[:,:(colsTest+1)//2]

    lowerFaceTest=rndTest[:,(colsTest//2):]

    return upperFaceTest,lowerFaceTest,rndTest





def createEstimator():

    ESTIMATORS = {

        "Extra trees": ExtraTreesRegressor(n_estimators=10,

                                           max_features=32,     # Out of 20000

                                           random_state=0),

        "K-nn": KNeighborsRegressor(),                          # Accept default parameters

        "Linear regression": LinearRegression(),

        "Ridge": RidgeCV(),

    }

    return ESTIMATORS

    

def predict(ESTIMATORS,upperFaceTrain,lowerFaceTrain,upperFaceTest):

    lowerFaceTestpredict = dict()

    for name, estimator in ESTIMATORS.items():

        estimator.fit(upperFaceTrain, lowerFaceTrain)

        lowerFaceTestpredict[name] = estimator.predict(upperFaceTest)

    return lowerFaceTestpredict   



    



def plotPredictions(upperFaceTest,lowerFaceTest,lowerFaceTestpredict,rndTest):

    imageShape=(64,64)

    numFaces=10

    numRows=10

    numCols = 1 + len(ESTIMATORS) # 1 original + 4 predictions

    plt.figure(figsize=(12,12))

    plt.suptitle("Partial Face Recognition", size=16)

    for i in range(numFaces):

        origFace = rndTest[i]

        if i!=0:

            sub = plt.subplot(numRows, numCols, i * numCols + 1)

        else:

            sub = plt.subplot(numRows, numCols, i * numCols+1,

                              title="Origianl Face")

        sub.axis("off")

        sub.imshow(origFace.reshape(imageShape),

                   cmap=plt.cm.gray,

                   interpolation="nearest")

        for j, est in enumerate(sorted(ESTIMATORS)):

            predictedFace = np.hstack((upperFaceTest[i], lowerFaceTestpredict[est][i]))

    

            if i!=0:

                sub = plt.subplot(numFaces, numCols, i * numCols + 2 + j)

    

            else:

                sub = plt.subplot(numFaces, numCols, i * numCols + 2 + j,

                                  title=est)

    

            sub.axis("off")

            sub.imshow(predictedFace.reshape(imageShape),

                       cmap=plt.cm.gray,

                       interpolation="nearest")



    plt.show()

      

facesDSAll,targets=loadData()

flattenFacesDSAll=prepDS(facesDSAll)

upperFaceTrain,lowerFaceTrain=prepTrainDS(flattenFacesDSAll,targets)

upperFaceTest,lowerFaceTest,rndTest=prepTestDS(flattenFacesDSAll,targets)

ESTIMATORS=createEstimator()

lowerFaceTestpredict=predict(ESTIMATORS,upperFaceTrain,lowerFaceTrain,upperFaceTest)

plotPredictions(upperFaceTest,lowerFaceTest,lowerFaceTestpredict,rndTest)
