import scipy.signal as ss

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import cv2

from sklearn.ensemble import RandomForestClassifier



from skimage import feature

#read input data

trainData = pd.read_csv("../input/train.csv")

testData = pd.read_csv("../input/test.csv")

trainY = trainData.iloc[:,0].values

trainX = trainData.iloc[:,1:].values

testX = testData.iloc[:,:].values

#define scharr filters

scharr = np.zeros((3,3,2))

scharr[:,:,0] = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]

scharr[:,:,1] = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]

#edge detection

edges = np.zeros((len(trainX),30,30,3))

for k in range(0,len(trainX)):

    for i in range(0,2):

       edges[k,:,:,i] = ss.convolve2d(trainX[k],scharr[:,:,i])

    edges[k,:,:,2] = np.sqrt(np.square(edges[k,:,:,0]) + np.square(edges[k,:,:,1]))

#train

rf = RandomForestClassifier(n_estimators = 100)

rf.fit(trainX, trainY)
#show data

index = np.random.randint(0,len(testX)-6)

testY = rf.predict(testX[index:index+5,:])



fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(8, 3),

                                    sharex=True, sharey=True)



ax1.imshow(testX[index].reshape(28,28), cmap=plt.cm.binary)

ax1.axis('off')

ax1.set_title('%d'% testY[0], fontsize=20)



ax2.imshow(testX[index+1].reshape(28,28), cmap=plt.cm.binary)

ax2.axis('off')

ax2.set_title('%d'% testY[1], fontsize=20)



ax3.imshow(testX[index+2].reshape(28,28), cmap=plt.cm.binary)

ax3.axis('off')

ax3.set_title('%d'% testY[2], fontsize=20)



ax4.imshow(testX[index+3].reshape(28,28), cmap=plt.cm.binary)

ax4.axis('off')

ax4.set_title('%d' % testY[3], fontsize=20)



ax5.imshow(testX[index+4].reshape(28,28), cmap=plt.cm.binary)

ax5.axis('off')

ax5.set_title('%d' % testY[4], fontsize=20)



fig.tight_layout()



plt.show()

plt.close()
#make a submission

testY = rf.predict(testX)

out_file = open("submission_dr_fishface.csv", "w")

out_file.write("ImageId,Label\n")

for i in range(len(testY)):

    out_file.write(str(i+1) + "," + str(int(testY[i])) + "\n")

   

out_file.close()