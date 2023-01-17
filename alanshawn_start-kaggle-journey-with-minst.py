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
trainTable = pd.read_csv('../input/train.csv')

sampleSubmission = pd.read_csv('../input/sample_submission.csv')

testTable = pd.read_csv('../input/test.csv')
trainTable.describe()
sampleSubmission.describe()
testTable.describe()
X = np.asarray(trainTable.iloc[:, 1:]).astype(np.float32)/255.0

Y = np.asarray(trainTable.iloc[:, 0])

print(X.shape)

print(Y.shape)
# using linear, multi-class SVM so that everything can be done within minutes

from sklearn.svm import LinearSVC



# leave some samples out for performance evaluation

trX, trY = X[:-2000, :], Y[:-2000]

tsX, tsY = X[-2000:, :], Y[-2000:]



svm = LinearSVC(verbose=True)

svm.fit(trX, trY)
from sklearn.metrics import accuracy_score, confusion_matrix



prY = svm.predict(tsX)

acc = accuracy_score(tsY, prY)

confMat = confusion_matrix(tsY, prY)



print(acc)

print(confMat)
# generate output



outX = np.asarray(testTable).astype(np.float32)/255.0

print(outX.shape)



outY = svm.predict(outX)

print(outY.shape)



result = np.zeros([outY.size, 2], np.int)

result[:, 0] = np.arange(1, outY.size + 1)

result[:, 1] = outY



pdOut = pd.DataFrame(result, columns=['ImageId', 'Label'])

pdOut.to_csv('output.csv', index=False)