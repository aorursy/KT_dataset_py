# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
test = pd.read_csv('../input/test.csv')
sns.heatmap(train.isnull())
sns.countplot(x="Survived", data=train)
sns.countplot(x="Survived", data=train, hue="Sex")
sns.countplot(x="Survived", data=train[train.Age < 16], hue="Sex")
#sns.distplot(train["Fare"], hue="Survived")
sns.distplot(train["Age"].dropna(), bins=40)
sns.boxplot(x="Pclass", y = "Age", data = train)
ages = {x:int(train[train.Pclass == x].Age.median()) for x in range(1,4)}
train["Age"] = [ages[x] for x in train.Pclass] 
test["Age"] = [ages[x] for x in test.Pclass] 
train.drop(["Cabin", "Name", "Ticket"], axis=1, inplace=True)

test.drop(["Cabin", "Name", "Ticket"], axis=1, inplace=True)
train = train.dropna()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.lda import LDA

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn.neural_network import MLPClassifier
label_cols = ["Sex", "Embarked"]
le = {x: preprocessing.LabelEncoder() for x in label_cols}
for x in le:

    train[x] = le[x].fit_transform(train[x])

    test[x] = le[x].transform(test[x])
labels = train.Survived

train.drop("Survived", axis = 1, inplace=True)
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
pl = Pipeline([('scale', StandardScaler()), ('lda', LDA()), ('clf', LogisticRegression())])

pl = Pipeline([('scale', StandardScaler()), ('lda', PCA(n_components=4)), ('clf', svm.SVC())])

#pl = RandomForestClassifier()

#pl = XGBClassifier(silent=False)

#pl = GaussianNB()

#pl = QuadraticDiscriminantAnalysis()ponents



pl = svm.SVC()



pl = MLPClassifier(hidden_layer_sizes=(30, ), verbose = True, activation= 'tanh', max_iter=100)
pl.fit(train, labels)
preds = pl.predict(test)
len(test)
len(preds)
res = pd.DataFrame(preds, columns = ["Survived"], index = test.PassengerId)
res.to_csv("submit.csv")
train.head()
lda = LDA(n_components=2)

pca = PCA(n_components=2)
scaler = StandardScaler()

trains = scaler.fit_transform(train)



trans = pca.fit_transform(trains, labels)

dfp = pd.DataFrame(trans, index = train.index)



trans = lda.fit_transform(trains, labels)

df = pd.DataFrame(trans, columns = ["lda"], index = train.index)
df["labels"] = labels

dfp["labels"] = labels
df.plot(kind='scatter', x='lda', y='labels')
dfp.plot(kind='scatter', x='pca1', y='pca2',  c="labels")
%matplotlib inline