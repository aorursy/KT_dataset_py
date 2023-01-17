import numpy as np 

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

features = list(train.columns.values)

features.remove("Activity")

for i in features:

    train[i] = le.fit_transform(train[i])

train["Activity"] = train["Activity"].astype("category").cat.codes

X = train[features]

Y = train["Activity"]
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier()

clf = clf.fit(X, Y)

model = SelectFromModel(clf, prefit=True)

New_features = model.transform(X)

print(New_features.shape)
from sklearn.svm import LinearSVC

lsvc = LinearSVC()

lsvc = lsvc.fit(X, Y)

model_2 = SelectFromModel(lsvc, prefit=True)

New_features_2 = model_2.transform(X)

print(New_features_2.shape)
from sklearn.decomposition import PCA

pca = PCA(n_components=50)

pca = pca.fit(X);

X_pca = pca.transform(X)

X_pca.shape
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf = rf.fit(X,Y)

model_3 = SelectFromModel(rf, prefit=True)

New_features_3 = model_3.transform(X)

print(New_features_3.shape)