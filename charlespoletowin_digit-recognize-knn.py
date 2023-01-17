# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

print(train.shape)

print(train.head())
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

import csv
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train_data=train.values[:,1:]

train_label=train.values[:,0]

# PCA

trainData = np.array(train_data)

testData = np.array(test)

'''

n_components>=1

  n_components=NUM   设置占特征数量比

0 < n_components < 1

  n_components=0.99  设置阈值总方差占比

2）whiten ：判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.对于PCA降维本身来说，一般不需要白化。

'''

pca = PCA(n_components=0.7,whiten=False)

pca.fit(trainData)

pcaTrainData = pca.transform(trainData)

pcaTestData = pca.transform(testData)

print(pca.n_components_)
clf = KNeighborsClassifier(n_neighbors=15)

clf.fit(pcaTrainData, np.ravel(train_label))

testLabel = clf.predict(pcaTestData)

with open('result.csv', 'w') as myFile:

    myWriter = csv.writer(myFile)

    myWriter.writerow(["ImageId", "Label"])

    index = 0

    for r in testLabel:

        index += 1

        myWriter.writerow([index, int(r)])
!pwd
!ls -a