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
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.columns
df_train.isnull().any().sum()
df_test.isnull().any().sum()
df_train.head()

b = df_train.iloc[1,1:].values
c = np.reshape(b,(1,784))
c
df_train.iloc[1,1:].values

#c[0][1:65]
#np.reshape(c[0][1:65],(1,64))
import pylab as pl
a = df_train.iloc[1,1:].values
pl.figure(figsize = (6,6))
pl.gray()
plt.subplot(1,1,1)
plt.imshow(np.reshape(df_train.iloc[1,1:].values,(28,28)), cmap = plt.cm.gray_r, interpolation = 'nearest')
plt.subplot(1,1,1)
plt.imshow(np.reshape(df_train.iloc[2,1:].values,(28,28)), cmap = plt.cm.gray_r, interpolation = 'nearest')
#pl.show()
df_train['label'].shape
df_train.tail()
from sklearn import ensemble
classifier  = ensemble.RandomForestClassifier()
y = df_train['label']
x = [df_train.iloc[i,1:].values for i in range(42000)]
classifier.fit(x,y)
df_test.tail()
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()

submission.iloc[70,1]
#yt = df_test['label']
#xt = [df_test.iloc[i,1:].values for i in range(2800)]

for i in range(28000):    
    submission.iloc[i,1] = classifier.predict(df_test.iloc[i,0:].values.reshape(1,784))[0]
#classifier.predict(df_test.iloc[1,0:].values.reshape(1,784))
#df_test.iloc[1,0:].values
import cv2
dictio = {0:'predicted :0 ',1:'predicted : 1',2:'predicted : 2',3:'predicted : 3',4:'predicted : 4',5:'predicted : 5',6:'predicted : 6',7:'predicted : 7',8:'predicted : 8',9:'predicted : 9'}
fig, axes = plt.subplots(4, 4, figsize = (15,15))
for row in axes:
    for axe in row:
        index = np.random.randint(len(submission))
        axe.imshow(np.reshape(df_test.iloc[index,0:].values,(28,28)))
        key = classifier.predict(df_test.iloc[index,0:].values.reshape(1,784))[0]
        axe.set_title(dictio[key])
        axe.set_axis_off()
submission.to_csv("subm.csv", index = False)