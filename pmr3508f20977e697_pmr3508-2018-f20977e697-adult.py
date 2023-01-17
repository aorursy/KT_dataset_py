import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from math import sqrt, factorial, log, ceil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import sklearn as skl
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
adult = pd.read_csv('../input/adultdataset/train_data (1).csv',
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
adult.head()
nadult = adult.dropna()
nadult.shape
nadult.describe()
fig = plt.figure(figsize=(20,15))
cols = 5
rows = ceil(float(nadult.shape[1]) / cols)
for i, column in enumerate(nadult.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if nadult.dtypes[column] == np.object:
        nadult[column].value_counts().plot(kind="bar", axes=ax)
    else:
        nadult[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
TargetRace = pd.crosstab(nadult["Target"],nadult["Race"],margins=True)
TargetRace
def percent(colum):
    return colum*100//float(colum[-1])

TargetSex = pd.crosstab(nadult["Target"], nadult["Sex"],margins = True)
TargetSex
TargetSex.apply(percent,axis=0)
nadult["Country"].value_counts()
nadult["Occupation"].value_counts()
testAdult = pd.read_csv('../input/adultdataset/test_data (2).csv',
        names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
nTestAdult = testAdult.dropna()
testAdult.head()
testAdult['Target'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')
nTestAdult = testAdult.dropna()
testAdult.head()
neighbors = [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35]
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult = nadult.Target
XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
YtestAdult = nTestAdult.Target
print('With CV = 3:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xadult, Yadult, cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
print('With CV = 5:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xadult, Yadult, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    
        
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
print('With CV = 10:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xadult, Yadult, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    
        
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
print('With CV = 15:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xadult, Yadult, cv=15, scoring='accuracy')
    cv_scores.append(scores.mean())
    values = np.array([5,8,9,10,20,30,50,100])
        
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
print('With CV = 20:')
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xadult, Yadult, cv=20, scoring='accuracy')
    cv_scores.append(scores.mean())
    values = np.array([5,8,9,10,20,30,50,100])
        
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
knn = KNeighborsClassifier(n_neighbors=19)
scores = cross_val_score(knn, Xadult, Yadult, cv=3)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
accuracy_score(YtestAdult,YtestPred)
knn = KNeighborsClassifier(n_neighbors=23)
scores = cross_val_score(knn, Xadult, Yadult, cv=5)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
accuracy_score(YtestAdult,YtestPred)
knn = KNeighborsClassifier(n_neighbors=31)
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
accuracy_score(YtestAdult,YtestPred)
knn = KNeighborsClassifier(n_neighbors=19)
scores = cross_val_score(knn, Xadult, Yadult, cv=15)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
accuracy_score(YtestAdult,YtestPred)
knn = KNeighborsClassifier(n_neighbors=21)
scores = cross_val_score(knn, Xadult, Yadult, cv=20)
scores
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
YtestPred
accuracy_score(YtestAdult,YtestPred)
from sklearn import preprocessing
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult.iloc[:,0:14]
Yadult = numAdult.Target
XtestAdult = numTestAdult.iloc[:,0:14]
YtestAdult = numTestAdult.Target
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
Xadult = numAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country"]]
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
Xadult = numAdult[["Age", "Workclass", "Education-Num", 
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]
XtestAdult = numTestAdult[["Age", "Workclass", "Education-Num", 
        "Occupation", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week"]]
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(Xadult,Yadult)
YtestPred = knn.predict(XtestAdult)
accuracy_score(YtestAdult,YtestPred)
predict = pd.DataFrame(nTestAdult)
predict["Target"] = YtestPred
predict
predict.to_csv("prediction.csv", index=False)
