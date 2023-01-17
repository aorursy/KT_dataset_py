import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
train = pd.read_csv("../input/database/train.csv")
test= pd.read_csv("../input/database/teste.csv")
train.head()
train.shape
#The meaning of the features
"""
hapaco =  overcrowding by rooms
v14a, =  has bathroom in the household
escolari = years of schooling
hhsize, household size
paredblolad, = if predominant material on the outside wall is block or brick
pisonatur, = if predominant material on the floor is  natural material
cielorazo, = if the house has ceiling
rooms,  number of all rooms in the house
instlevel1, =1 no level of education 
instlevel8, =1 undergraduate and higher education
"""
""" Plot 1: owns a tablet (no = 0, yes = 1)"""
train["v18q"].value_counts().plot(kind = "bar")
""" Plot 2: owns a television (no = 0, yes = 1)"""
train["television"].value_counts().plot(kind = "bar")
""" Plot 3: overcrowding by room (no = 0, yes = 1)"""
train["pisonatur"].value_counts().plot(kind="pie")
means=train.mean()
train=train.fillna(means)
from sklearn import preprocessing
train=train.fillna(means)
train_num=train.iloc[:,0:142].apply(preprocessing.LabelEncoder().fit_transform)
test_num=test.iloc[:,0:142].apply(preprocessing.LabelEncoder().fit_transform)
test_f=test_num[["v18q","television","pisonatur","escolari","cielorazo","rooms","instlevel1","instlevel8"]]
train_f=train_num[["v18q","television","pisonatur","escolari","cielorazo","rooms","instlevel1","instlevel8"]]
train_y=train.Target
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_f,train_y)
"""Doing the cross validation"""
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn,train_f,train_y,cv=300)
print(np.mean(scores))
test_y= knn.predict(test_f)
test_y
submission_id=test.Id
submission_pred=test_y
sub = pd.DataFrame({'Id':submission_id[:],'Target':submission_pred[:]})
sub.Target.value_counts()
