# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load data

data = pd.read_csv("../input/train.csv")
# Take a look at the data

data.head()
data.info()
traindata = data.sample(frac = 0.8)

valind = list(set(data.index) - set(traindata.index))

valdata = data.loc[valind, :]

traintarget = data.loc[traindata.index, 'Survived']

valtarget = data.loc[valind, 'Survived']
traintarget.value_counts()
traindata['Sex'].value_counts()
traindata['Pclass'].value_counts()
sex = traindata['Sex']

pd.crosstab(traintarget, sex)
from sklearn.metrics import mutual_info_score

mutual_info_score(traintarget, sex)
pclass = traindata['Pclass']

pd.crosstab(traintarget, pclass)
target12 = traindata[traindata['Pclass'].isin([1,2])]['Survived']

target13 = traindata[traindata['Pclass'].isin([1,3])]['Survived']

target23 = traindata[traindata['Pclass'].isin([2,3])]['Survived']

pclass12 = traindata[traindata['Pclass'].isin([1,2])]['Pclass']

pclass13 = traindata[traindata['Pclass'].isin([1,3])]['Pclass']

pclass23 = traindata[traindata['Pclass'].isin([2,3])]['Pclass']
mutual_info_score(target12, pclass12)
mutual_info_score(target13, pclass13)
mutual_info_score(target23, pclass23)
pd.crosstab(sex, pclass)
sex12 = traindata[traindata['Pclass'].isin([1,2])]['Sex']

sex13 = traindata[traindata['Pclass'].isin([1,3])]['Sex']

sex23 = traindata[traindata['Pclass'].isin([2,3])]['Sex']
mutual_info_score(sex12, pclass12)
mutual_info_score(sex13, pclass13)
mutual_info_score(sex23, pclass23)
pred1 = 1 * valdata['Sex'].isin(['female'])
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy: {0:.2f}".format(accuracy_score(valtarget, pred1)))
C = confusion_matrix(valtarget, pred1)

C
TN = C[0,0]

FN = C[1,0]

TP = C[1,1]

FP = C[0,1]
print("Precision: {0:.2f}".format(TP / (TP + FP)))
print("Recall: {0:.2f}".format(TP / (TP + FN)))
ind = valdata[valtarget != pred1].index

df = valdata.loc[ind, ['Survived', 'Sex' ,'Pclass']]

df
pd.crosstab(df['Survived'], df['Pclass'])
traindata = data.sample(frac = 0.8)

valind = list(set(data.index) - set(traindata.index))

valdata = data.loc[valind, :]

traintarget = data.loc[traindata.index, 'Survived']

valtarget = data.loc[valind, 'Survived']
pred1 = 1 * valdata['Sex'].isin(['female'])

pred2 = 1 * valdata['Pclass'].isin([3])

pred2 = 1 - pred2

pred2 = pred1 * pred2
print("Accuracy: {0:.2f}".format(accuracy_score(valtarget, pred2)))
C = confusion_matrix(valtarget, pred2)

C
TN = C[0,0]

FN = C[1,0]

TP = C[1,1]

FP = C[0,1]
print("Precision: {0:.2f}".format(TP / (TP + FP))); print("Recall: {0:.2f}".format(TP / (TP + FN)))
import matplotlib.pyplot as plt

data['Fare'].plot.hist(bins=40)

plt.show()
data['FareD'] = np.nan

data['FareD'][data['Fare'] <= 5] = 0

data['FareD'][(data['Fare'] > 5) & (data['Fare'] <= 10)] = 1

data['FareD'][(data['Fare'] > 10) & (data['Fare'] <= 15)] = 2

data['FareD'][(data['Fare'] > 15) & (data['Fare'] <= 20)] = 3

data['FareD'][(data['Fare'] > 20) & (data['Fare'] <= 30)] = 4

data['FareD'][(data['Fare'] > 30) & (data['Fare'] <= 50)] = 5

data['FareD'][(data['Fare'] > 50) & (data['Fare'] <= 80)] = 6

data['FareD'][data['Fare'] > 80] = 7
data['FareD'].plot.hist(bins=7)

plt.show()
data['AgeD'] = -1

data['AgeD'][data['Age'] <= 5] = 0

data['AgeD'][(data['Age'] > 5) & (data['Age'] <= 10)] = 1

data['AgeD'][(data['Age'] > 10) & (data['Age'] <= 20)] = 2

data['AgeD'][(data['Age'] > 20) & (data['Age'] <= 30)] = 3

data['AgeD'][(data['Age'] > 30) & (data['Age'] <= 40)] = 4

data['AgeD'][(data['Age'] > 40) & (data['Age'] <= 50)] = 5

data['AgeD'][(data['Age'] > 50) & (data['Age'] <= 60)] = 6

data['AgeD'][(data['Age'] > 60) & (data['Age'] <= 70)] = 7

data['AgeD'][data['Age'] > 70] = 8
data['Embarked'][61] = 'C'

data['Embarked'][829] = 'C'
datacopy = data.copy()
columns = ['Sex', 'Pclass', 'Embarked', 'FareD', 'AgeD']

target = data['Survived']

data = data[columns]
from sklearn.preprocessing import LabelEncoder

data['Sex'] = LabelEncoder().fit_transform(data['Sex'])

data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
def customdist(x,y):

    w = {'Sex': 5, 'Pclass': 1, 'Embarked': 1, 'FareD': 1, 'AgeD': 3}

    return w['Sex'] * (x[0] != y[0]) + w['Pclass'] * abs(x[1] - y[1])**2 + w['Embarked'] * (x[2] != y[2]) + w['FareD'] * (x[3] != y[3]) + w['AgeD'] * (x[4] != y[4])
traindata = data.sample(frac = 0.8)

valind = list(set(data.index) - set(traindata.index))

valdata = data.loc[valind, :]

traintarget = target.loc[traindata.index]

valtarget = target.loc[valind]
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors = 4, algorithm = 'ball_tree', metric = customdist)

kn.fit(traindata, traintarget)

pred = kn.predict(valdata)
print("Accuracy: {0:.2f}".format(accuracy_score(valtarget, pred)))
C = confusion_matrix(valtarget, pred)

C
ind = valdata[valtarget != pred].index

print(datacopy.loc[ind, ['Survived', 'Sex' ,'Pclass', 'Embarked', 'FareD', 'AgeD']])
kn = KNeighborsClassifier(n_neighbors = 4, algorithm = 'ball_tree', metric = customdist)
from sklearn.model_selection import cross_val_score

cvscores = cross_val_score(kn, data, target, cv=5)

print("Accuracy: {0:.2f} (+/-) {1:.2f}".format(cvscores.mean(),cvscores.std() * 2))