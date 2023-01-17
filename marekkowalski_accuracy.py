import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

%matplotlib inline

trainData = pd.read_csv("../input/train.csv", usecols=[1, 2, 3, 4])
trainLabels = pd.read_csv("../input/train.csv", usecols=[5])

trainData = np.array(trainData, dtype=np.float32)
trainLabels = np.array(trainLabels, dtype=np.int64).squeeze()
unique = np.unique(trainLabels)
for idx in range(0, 1000):
    idxs = np.where(trainLabels == unique[idx])[0]
    curCheckins = trainData[idxs]
    centroid = np.mean(curCheckins, axis=0)
    curDistsX = np.abs(curCheckins[:, 0] - centroid[0])
    curDistsY = np.abs(curCheckins[:, 1] - centroid[1])

    if idx == 0:
        distsX = curDistsX
        distsY = curDistsY
        checkins = curCheckins
    else:
        distsX = np.concatenate((distsX, curDistsX))
        distsY = np.concatenate((distsY, curDistsY))
        checkins = np.vstack((checkins, curCheckins))



hist, bins = np.histogram(checkins[:, 2], 10, (0, 1000))
idxs = np.digitize(checkins[:, 2], bins)
idxs = idxs - 1

mediansX = []
mediansY = []
for i in range(len(bins)):
    curDistsX = distsX[np.nonzero(idxs == i)]
    curDistsY = distsY[np.nonzero(idxs == i)]

    mediansX.append(np.median(curDistsX))
    mediansY.append(np.median(curDistsY))

x = np.linspace(0, 1000, 11)
plt.ylim((0, 0.3))

plt.plot(x, mediansX)
plt.plot(x, mediansY)
plt.show()