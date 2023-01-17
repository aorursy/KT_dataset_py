import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from sklearn import svm

import statistics

import matplotlib.pyplot as plt


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

t_data = pd.read_csv("../input/train.csv")
ts_data = pd.read_csv("../input/test.csv")
ddata = t_data.drop(["PlayerID","Name"], axis=1)
sdata = ts_data.drop(["PlayerID","Name"], axis=1)
ddata = ddata.interpolate()
ddata = ddata.replace([np.inf], np.float64.max)
ddata = ddata.replace([-np.inf], np.float64.min)

features = ddata.loc[:, ddata.columns.values[:len(ddata.columns.values)-1]].values
labels = ddata.loc[:, ['TARGET_5Yrs']].values

st_features = preprocessing.StandardScaler().fit_transform(features)

sdata = sdata.interpolate()
sdata = sdata.replace([np.inf], np.float64.max)
sdata = sdata.replace([-np.inf], np.float64.min)

sfeatures = sdata.loc[:, sdata.columns.values].values

st_sfeatures = preprocessing.StandardScaler().fit_transform(sfeatures)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(features)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

test_pcs = pca.fit_transform(sfeatures)
test_pdf = pd.DataFrame(data=test_pcs
                       , columns = ['principal component 1', 'principal component 2'])

df = pd.concat([principalDf, ddata['TARGET_5Yrs']], axis=1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = df['TARGET_5Yrs'] == target
    ax.scatter(df.loc[indicesToKeep, 'principal component 1']
               , df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
pca = PCA(n_components=10)

pca.fit(features)
principalComponents = pca.transform(features)
test_principalComponenta = pca.transform(sfeatures)
print(principalComponents.shape, "\n", test_principalComponenta.shape)
# svmodel = svm.SVC(C=1, kernel="poly")
# svmodel.fit(main, labels)
# sklearn.metrics.accuracy_score(labels, svmodel.predict(features))
# sklearn.metrics.accuracy_score(tlabels, svmodel.predict(sfeatures))
# cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': svmodel.predict(test_principalComponenta) }
# submission = pd.DataFrame(cols)
# submission.to_csv("submission.csv", index=False)
# submission
inputf = features
# frst1 = IsolationForest(n_estimators=5)
# frst1.fit(inputf, labels)
# frst2 = IsolationForest(n_estimators=5)
# frst2.fit(inputf, labels)
# frst3 = IsolationForest(n_estimators=5)
# frst3.fit(inputf, labels)
# frst4 = IsolationForest(n_estimators=5)
# frst4.fit(inputf, labels)
# frst5 = IsolationForest(n_estimators=5)
# frst5.fit(inputf, labels)
# frst6 = IsolationForest(n_estimators=5)
# frst6.fit(inputf, labels)
# frst7 = IsolationForest(n_estimators=5)
# frst7.fit(inputf, labels)
# frst8 = IsolationForest(n_estimators=5)
# frst8.fit(inputf, labels)
# frst9 = IsolationForest(n_estimators=5)
# frst9.fit(inputf, labels)
# frst10 = IsolationForest(n_estimators=5)
# frst10.fit(inputf, labels)
testf = sfeatures
# pred1 = frst1.predict(testf)
# pred2 = frst2.predict(testf)
# pred3 = frst3.predict(testf)
# pred4 = frst4.predict(testf)
# pred5 = frst5.predict(testf)
# pred6 = frst6.predict(testf)
# pred7 = frst7.predict(testf)
# pred8 = frst8.predict(testf)
# pred9 = frst9.predict(testf)
# pred10 = frst10.predict(st_sfeatures)
# res1 = []
# for i1,i2,i3,i4,i5,i6,i7,i8,i9,i10 in zip(pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10):
#     j = np.sum([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10])
#     if j >= 5:
#         res1.append(1)
#     else:
#         res1.append(0)
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

res = []
res2 = []
for name, clf in zip(names, classifiers):
    clf.fit(inputf, labels)
    res.append(clf.predict(testf))
    temp = clf.predict(inputf)
    res2.append(temp)
    print(name, " : ", sklearn.metrics.accuracy_score(labels, temp))

# res.append(res1)
res2 = np.array(res2)
res3 = []
for i in range(res2.shape[1]):
    s = 0
    for j in res2:
        s = s + j[i]
    if s >= 5:
        res3.append(1)
    else:
        res3.append(0)
sklearn.metrics.accuracy_score(labels, res3)
res = np.array(res)
res3 = []
for i in range(res.shape[1]):
    s = 0
    for j in res:
        s = s + j[i]
    if s >= 5:
        res3.append(1)
    else:
        res3.append(0)
cols = { 'PlayerID': [i+901 for i in range(440)] , 'TARGET_5Yrs': res3 }
submission = pd.DataFrame(cols)
submission.to_csv("submission.csv", index=False)
print(submission)