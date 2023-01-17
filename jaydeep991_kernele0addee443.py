import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("../input/train.csv")

data
datamat=data.as_matrix()

datamat
clf=DecisionTreeClassifier()



# training dataset

xtrain=datamat[:21000,1:] #pixel train data

train_label=datamat[:21000,0] #import train data



clf.fit(xtrain, train_label)

#testing data

xtest=datamat[21000:,1:]

actual_label=datamat[21000:,0]
d=xtest[3]

d.shape=(28,28)

plt.imshow(d,cmap='gray')

print(clf.predict([xtest[3]]))

plt.show()