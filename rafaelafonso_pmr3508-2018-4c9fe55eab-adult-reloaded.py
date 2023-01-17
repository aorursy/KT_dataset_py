import pandas as pd
import sklearn
adult = pd.read_csv("../input/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
adult.head()
import matplotlib.pyplot as plt
adult["sex"].value_counts()
adult["sex"].value_counts().plot(kind="bar")
adult["education"].value_counts().plot(kind="pie")
adult["age"].mean()
adult["native.country"].value_counts().plot(kind="bar")
plt.ylim(top=750)
plt.xlim(left=0.5)
nadult = adult.dropna()
nadult.shape
from sklearn import preprocessing
numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
Xadult = numAdult[["age","education.num","capital.gain","relationship","native.country"]]
Yadult = numAdult.income
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=52)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xadult, Yadult, cv=10)
scores
sum(scores) / len(scores)
impAdult = adult.apply(lambda x:x.fillna(x.value_counts().index[0]))
impAdult.head()
testAdult = pd.read_csv("../input/test_data.csv",
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")
impTestAdult = testAdult.apply(lambda x:x.fillna(x.value_counts().index[0]))
impTestAdult.head()
from sklearn import preprocessing
numImpAdult = impAdult.apply(preprocessing.LabelEncoder().fit_transform)
numImpTestAdult = impTestAdult.apply(preprocessing.LabelEncoder().fit_transform)
numImpAdult.corr(method='pearson').income.sort_values(ascending=True)
XAdult = numImpAdult[["capital.gain", "education.num", "relationship", "age", "hours.per.week", "sex"]]
YAdult = numImpAdult.income
XAdultTest = numImpTestAdult[["capital.gain", "education.num", "relationship", "age", "hours.per.week", "sex"]]
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=7)
scores = cross_val_score(clf, XAdult, YAdult, cv=10)
scores
sum(scores) / len(scores)
clf.fit(XAdult, YAdult)
YtestPred = clf.predict(XAdultTest)
Id = testAdult["Id"]
submission = pd.DataFrame({"Id": Id, "income": YtestPred})
submission["income"] = submission["income"].replace(0, "<=50K")
submission["income"] = submission["income"].replace(1, ">50K")
submission.head()
submission.to_csv("submissionTree.csv", index = False)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=11, random_state=90)
scores = cross_val_score(clf, XAdult, YAdult, cv=10)
scores
sum(scores) / len(scores)
clf.fit(XAdult, YAdult)
YtestPred = clf.predict(XAdultTest)
Id = testAdult["Id"]
submission = pd.DataFrame({"Id": Id, "income": YtestPred})
submission["income"] = submission["income"].replace(0, "<=50K")
submission["income"] = submission["income"].replace(1, ">50K")
submission.head()
submission.to_csv("submissionForest.csv", index = False)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(40,), random_state=1)
scores = cross_val_score(clf, XAdult, YAdult, cv=10)
scores
sum(scores) / len(scores)
clf.fit(XAdult, YAdult)
YtestPred = clf.predict(XAdultTest)
Id = testAdult["Id"]
submission = pd.DataFrame({"Id": Id, "income": YtestPred})
submission["income"] = submission["income"].replace(0, "<=50K")
submission["income"] = submission["income"].replace(1, ">50K")
submission.head()
submission.to_csv("submissionNeural.csv", index = False)