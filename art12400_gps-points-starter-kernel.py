import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import svm,metrics
trainSet = pd.read_csv('../input/gps-points/train.csv')

testSet = pd.read_csv('../input/gps-points/test.csv')
trainSet.head()
print("Points from 河北省 (Hebei):")

print(trainSet["河北省"].sum())

print("Points from 北京市 (Beijing):")

print(trainSet["北京市"].sum())

print("Points from 河北省 (Tianjin):")

print(trainSet["天津市"].sum())
print("Points from 河北省 (Hebei):")

print(testSet["河北省"].sum())

print("Points from 北京市 (Beijing):")

print(testSet["北京市"].sum())

print("Points from 河北省 (Tianjin):")

print(testSet["天津市"].sum())
plt.subplots(figsize=(15,10))

plt.scatter(x=trainSet.loc[trainSet["河北省"]==1]["lng"],y=trainSet.loc[trainSet["河北省"]==1]["lat"],c="y",label="Hebei")

plt.scatter(x=trainSet.loc[trainSet["天津市"]==1]["lng"],y=trainSet.loc[trainSet["天津市"]==1]["lat"],c="b",label="Tianjin")

plt.scatter(x=trainSet.loc[trainSet["北京市"]==1]["lng"],y=trainSet.loc[trainSet["北京市"]==1]["lat"],c="r",label="Beijing")

plt.xlabel("Latitude",fontsize=16)

plt.ylabel("Longitude",fontsize=16)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.legend(fontsize=16)

plt.show()
# Build training and test sets

X_train = trainSet[["lng","lat"]].to_numpy()

Y_train = trainSet[["河北省","北京市","天津市"]].to_numpy()

Y_train = np.argmax(Y_train,axis=1)

X_test = testSet[["lng","lat"]].to_numpy()

Y_test = testSet[["河北省","北京市","天津市"]].to_numpy()

Y_test = np.argmax(Y_test,axis=1)



# Train SVM

clf = svm.SVC(kernel="rbf", gamma=2)

clf.fit(X_train, Y_train)



# Compute accuracies

Y_pred = np.argmax(clf.decision_function(X_train),axis=1)

print("Train set accuracy: %.2f" % metrics.accuracy_score(Y_train,Y_pred))

Y_pred = np.argmax(clf.decision_function(X_test),axis=1)

print("Test set accuracy: %.2f" % metrics.accuracy_score(Y_test,Y_pred))



# Plot training set, and support vectors

plt.figure(figsize=(15, 10))

plt.clf()

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=85, facecolors='none', zorder=10, edgecolors='k', label="Support Vector")

plt.scatter(x=trainSet.loc[trainSet["河北省"]==1]["lng"],y=trainSet.loc[trainSet["河北省"]==1]["lat"],c="y",zorder=10,label="Hebei")

plt.scatter(x=trainSet.loc[trainSet["天津市"]==1]["lng"],y=trainSet.loc[trainSet["天津市"]==1]["lat"],c="b",zorder=10,label="Tianjin")

plt.scatter(x=trainSet.loc[trainSet["北京市"]==1]["lng"],y=trainSet.loc[trainSet["北京市"]==1]["lat"],c="r",zorder=10,label="Beijing")



x_min = 114

x_max = 119

y_min = 36

y_max = 42



# Plot contours between classes (learned district boundaries)

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

Z = np.argmax(Z,axis=1)

Z = Z.reshape(XX.shape)

plt.contour(XX, YY, Z, colors=['b', 'y', 'r'], linestyles=['--', '-', '--'], linewidths=[4,10,4])



plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)



plt.xlabel("Latitude",fontsize=16)

plt.ylabel("Longitude",fontsize=16)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.legend(fontsize=16)

plt.show()