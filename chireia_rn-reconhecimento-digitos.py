import numpy as np

import pandas as pd

import matplotlib.pyplot as pt

from sklearn.tree import DecisionTreeClassifier



data = pd.read_csv("../input/digit-recognizer/train.csv").as_matrix()

clf=DecisionTreeClassifier()



xtrain=data[0:21000,1:]

train_label=data[0:21000,0]



clf.fit(xtrain,train_label)



xtest=data[0:21000,1:]

actual_label=data[0:21000,0]



d=xtest[12]

d.shape=(28,28)

pt.imshow(255-d,cmap='gray')

print(clf.predict([xtest[12]]))

pt.show()



d=xtest[9]

d.shape=(28,28)

pt.imshow(255-d,cmap='gray')

print(clf.predict([xtest[9]]))

pt.show()



d=xtest[8]

d.shape=(28,28)

pt.imshow(255-d,cmap='gray')

print(clf.predict([xtest[8]]))

pt.show()



d=xtest[6]

d.shape=(28,28)

pt.imshow(255-d,cmap='gray')

print(clf.predict([xtest[6]]))

pt.show()
