import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
x_train = pd.read_csv("../input/train.csv")

x_test = pd.read_csv("../input/test.csv")
x_train = x_train.reindex(np.random.permutation(x_train.index))

x_test = x_test.reindex(np.random.permutation(x_test.index))

y_train = x_train["Activity"]

y_test = x_test["Activity"]

subject_train = x_train["subject"]

subject_test = x_test["subject"]

x_train = x_train.drop(["subject","Activity"],axis=1)

x_test = x_test.drop(["subject","Activity"],axis=1)
x_train_test = x_train.append(x_test)

y_train_test = y_train.append(y_test)

clf = LinearDiscriminantAnalysis()

clf.fit(x_train_test, y_train_test)

x_train_test = clf.transform(x_train_test)
x_train = x_train_test[:x_train.shape[0]]

x_test = x_train_test[x_train.shape[0]:]

y_train = y_train_test[:x_train.shape[0]]

y_test = y_train_test[x_train.shape[0]:]
clf = svm.SVC(gamma="scale",kernel="rbf")

clf.fit(x_train, y_train) 
clf.score(x_train,y_train)
clf.score(x_test,y_test)
y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred,labels=["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"])