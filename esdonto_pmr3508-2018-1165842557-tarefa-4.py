import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import os 
os.listdir('../input/')
dataTrain = pd.read_csv('../input/train_data.csv', na_values='?').dropna()
dataTrain.head()
XdataTrain = dataTrain.iloc[:, 1:-1]
YdataTrain = dataTrain['income']
dataTest = pd.read_csv('../input/test_data.csv', na_values='?')
dataTest.head()
XdataTest = dataTest.iloc[:, 1:]
IdTest = dataTest['Id']
print(XdataTrain.shape)
print(XdataTest.shape)
dataTrain.isnull().sum()[dataTrain.isnull().sum() > 0]
YdataTrain.value_counts().plot("bar")
print(YdataTrain.value_counts())
plt.rcParams['figure.figsize'] = [15,4]

labels = ['age', 'hours.per.week']
bins = [73, 15, 10]

for i, j in zip(labels, bins):
    XdataTrain[i].plot(kind='hist', legend=True, bins=j)
    plt.title(i)
    plt.show()
labels = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

for i in labels:
    XdataTrain[i].value_counts(sort=True)[:8].plot('bar')
    plt.title(i)
    plt.show()
XbalTrain = XdataTrain.copy()
YbalTrain = YdataTrain.copy()

maxValor = (XdataTrain[(YdataTrain=='<=50K')])
lenMax = len(XdataTrain[(YdataTrain=='<=50K')])

np.random.seed(3508)

remove_n = int(4 * lenMax / 5)
drop_indices = np.random.choice(maxValor.index, remove_n, replace=False)

XbalTrain = XbalTrain.drop(drop_indices)
YbalTrain = YbalTrain.drop(drop_indices)

XbalTrain = XbalTrain.dropna()
YbalTrain = YbalTrain.dropna()

YbalTrain.value_counts().plot("bar")
print(YbalTrain.value_counts())
XdataTrain.shape
del XbalTrain['fnlwgt']
del XbalTrain['education']
del XbalTrain['relationship']
priv = XbalTrain['workclass']=="Private"
self = (XbalTrain['workclass']=="Self-emp-not-inc") | (XbalTrain['workclass']=="Self-emp-inc")
gov = (XbalTrain['workclass']=="Local-gov")|(XbalTrain['workclass']=="State-gov")|(XbalTrain['workclass']=="Federal-gov")
rest = np.invert(priv | self | gov)
del XbalTrain['workclass']
for i, j in zip(['priv', 'self', 'gov', 'rest'], [priv, self, gov, rest]):
    XbalTrain[i+'Work'] = j
XbalTrain['marital.status'] = (XbalTrain['marital.status']=="Married-civ-spous")|(XbalTrain['marital.status']=="Married-spouse-absent")|(XbalTrain['marital.status']=="Married-AF-spouse")
indexs = list(XbalTrain['occupation'].value_counts().keys())
rest = np.zeros(XbalTrain['occupation'].shape, dtype=bool)
for i in indexs[:-3]:
    XbalTrain[i] = XbalTrain['occupation'] == i
    rest = rest | (XbalTrain['occupation']==i)
XbalTrain['restOccupation'] = np.invert(rest)
del XbalTrain['occupation']
white = XbalTrain['race'] == 'White'
black = XbalTrain['race'] == 'Black'
rest = np.invert(white | black)
del XbalTrain['race']
for i, j in zip(['white', 'black', 'rest'], [white, black, rest]):
    XbalTrain[i+'Race'] = j
XbalTrain['sex'] = XbalTrain['sex'] == 'Male'
XbalTrain['native.country'] = XbalTrain['native.country'] == 'United-States'
del XdataTest['fnlwgt']
del XdataTest['education']
del XdataTest['relationship']

priv = XdataTest['workclass']=="Private"
self = (XdataTest['workclass']=="Self-emp-not-inc") | (XdataTest['workclass']=="Self-emp-inc")
gov = (XdataTest['workclass']=="Local-gov")|(XdataTest['workclass']=="State-gov")|(XdataTest['workclass']=="Federal-gov")
rest = np.invert(priv | self | gov)

del XdataTest['workclass']

for i, j in zip(['priv', 'self', 'gov', 'rest'], [priv, self, gov, rest]):
    XdataTest[i+'Work'] = j

XdataTest['marital.status'] = (XdataTest['marital.status']=="Married-civ-spous")|(XdataTest['marital.status']=="Married-spouse-absent")|(XdataTest['marital.status']=="Married-AF-spouse")

indexs = list(XdataTest['occupation'].value_counts().keys())

rest = np.zeros(XdataTest['occupation'].shape, dtype=bool)
for i in indexs[:-3]:
    XdataTest[i] = XdataTest['occupation'] == i
    rest = rest | (XdataTest['occupation']==i)

XdataTest['restOccupation'] = np.invert(rest)

del XdataTest['occupation']

white = XdataTest['race'] == 'White'
black = XdataTest['race'] == 'Black'
rest = np.invert(white | black)

del XdataTest['race']

for i, j in zip(['white', 'black', 'rest'], [white, black, rest]):
    XdataTest[i+'Race'] = j

XdataTest['sex'] = XdataTest['sex'] == 'Male'

XdataTest['native.country'] = XdataTest['native.country'] == 'United-States'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

accuracies = {}

for k in tqdm(range(1,71)):
    knn = KNeighborsClassifier(n_neighbors=k, p=1)
    scores = cross_val_score(knn, XbalTrain,YbalTrain, cv=10)
    accuracies[k] = scores.mean()

accuraciesSorted = list(accuracies.items())
accuraciesSorted.sort(key=lambda x: x[1])
print(accuraciesSorted[-10:])
accX = sorted(list(accuracies.keys()))
accY = [accuracies[i] for i in accX]

plt.plot(accX, accY)
plt.grid(True)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=3508)
cross_val_score(clf, XbalTrain, YbalTrain, cv=10).mean()
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
cross_val_score(clf, XbalTrain, YbalTrain, cv=10).mean()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=3508)
cross_val_score(clf, XbalTrain, YbalTrain, cv=10).mean()
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
cross_val_score(clf, XbalTrain, YbalTrain, cv=10).mean()
from sklearn.svm import SVC
clf = SVC()
cross_val_score(clf, XbalTrain, YbalTrain, cv=10).mean()
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=400, random_state=3508)
cross_val_score(clf, XbalTrain, YbalTrain, cv=10).mean()
clf = GradientBoostingClassifier(n_estimators=400, random_state=3508)
clf.fit(XbalTrain,YbalTrain)
YdataPred = clf.predict(XdataTest)
IdTest = list(IdTest)
YdataPred = list(YdataPred)

submission = np.array([IdTest, YdataPred])
submission = pd.DataFrame(submission.T, columns=['Id', 'income'])
submission.head()
submission.to_csv('out.csv', index=False)