# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv").as_matrix()
clf = DecisionTreeClassifier()

#train dataset
xtrain = data[0:21000, 1:]
train_label = data [0:21000, 0]

clf.fit(xtrain, train_label)

#test dataset
xtest = data[21000:,1:]
actual_label = data[21000:,0]

#example show 9th img
##d=xtest[8]
##d.shape=(28,28)
##pt.imshow(255-d, cmap='gray')
##print(clf.predict([xtest[8]]))
##pt.show()

#give entire test dataset
p = clf.predict(xtest)
count = 0
for i in range(0,21000): 
    count+=1 if p[i]==actual_label[i] else 0
print("Accuracy=", (count/21000)*100)
