import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#load data
data=pd.read_csv("../input/train.csv")
clf=DecisionTreeClassifier()

#dataset training
xtrain=data.iloc[0:21000,1:].values
train_label=data.iloc[0:21000,0].values

clf.fit(xtrain,train_label)

#testing dataset training
xtest=data.iloc[21000:,1:].values
actual_label=data.iloc[21000:,0].values

t=xtest[9]
t.shape=(28,28)
pt.imshow(255-t,cmap='gray')
print(clf.predict( [xtest[9]] ))
pt.show()
