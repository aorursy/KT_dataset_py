import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib notebook
import PIL.Image as Image
from PIL import ImageFilter
from mpl_toolkits.mplot3d import Axes3D

import os
import glob
rootFolder = '../input/hrrfaced/HRRFaceD/'
facesClasses = sorted(glob.glob(rootFolder + '*'))
for f in facesClasses:
    print(f)

# 1 sample, 1 channel, width 180, height 180
inputData = np.array([1, 180, 180], np.int16)
count = 0
for i in facesClasses:
    count += 1
    if count > 3:
        break
    trainFiles = glob.glob(i + '/train/*')
    for img in trainFiles:
        inImg = Image.open(img)
        inImg = np.array(inImg)
        inputData = np.append(inputData, inImg)
# display the distribution of values
plt.hist(inputData.reshape(-1), bins=100)
plt.title('distribution of values 3d depth image')
plt.annotate('background', xy=(50, 2000000), xytext=(200, 3000000),
            arrowprops=dict(facecolor='blue'),
            )
        
count = 0
for i in facesClasses:
    trainFiles = sorted(glob.glob(i + '/train/*'))
    testFiles = sorted(glob.glob(i + '/test/*'))
    for img in trainFiles:
        inImg = Image.open(img)
        inImg = np.array(inImg)
        # shift zero level for correct view
        minDistance = np.median(inImg[(inImg > 500) & (inImg < 900)].reshape(-1))
        faceStd = np.std(inImg[(inImg > 500) & (inImg < 900)].reshape(-1))
        inImg[(inImg < 500) | (inImg > minDistance + faceStd * 2)] = minDistance
        plt.matshow(inImg, cmap='plasma')
        plt.colorbar()
        plt.show()
        count += 1
        if count >= 3:
            break
    break
def getData(subFolder):
    data = np.empty([0, 180, 180], np.int16)
    target = np.empty([0, 1], np.int16)
    for fc in facesClasses:
        trainFiles = sorted(glob.glob(fc + '/' + subFolder + '*'))
        trainDataBuff = np.empty([len(trainFiles), 180, 180], np.int16)
        trainTargetBuff = np.empty([len(trainFiles), 1], np.int16)
        for i, e in enumerate(testFiles):
            inImg = Image.open(img)
            inImg = np.array(inImg)
            # shift zero level for correct view
            minDistance = np.median(inImg[(inImg > 500) & (inImg < 900)].reshape(-1))
            faceStd = np.std(inImg[(inImg > 500) & (inImg < 900)].reshape(-1))
            inImg[(inImg < 500) | (inImg > minDistance + faceStd * 2)] = minDistance
            trainDataBuff[i] = inImg
            trainTargetBuff[i] = int(fc[-2:])
        target = np.append(target, trainTargetBuff)
        data = np.append(data, trainDataBuff, axis=0)

    return [data, target]

print('load data...')
trainData, trainTarget = getData('train/')
testData, testTarget = getData('test/')
print('train data: {}'.format(trainData.shape))
print('train target: {}'.format(trainTarget.shape))
print('test data: {}'.format(testData.shape))
print('test target: {}'.format(testTarget.shape))
print('complete')
