import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# reading input

train = pd.read_csv('../input/train.csv')

target = train['label']

train = train.drop("label", axis=1)
from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline # useful to tie the two together



# setting up the components

pca = ('pca', PCA(n_components = 50)) # I did play with the parameter

svc = ('svc', SVC(kernel = 'poly'))



# gluing into a pipe

estimators = [

    pca,

    svc

]

clf = Pipeline(estimators)
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(train, target, test_size=0.2, random_state=42)



clf.fit(features_train, labels_train)

print(clf.score(features_test, labels_test))
clf.fit(train, target)

test = pd.read_csv('../input/test.csv')

results = pd.Series(clf.predict(test), name="Label")

submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"),results],axis = 1)

submission.to_csv("results.csv", index = False)

# 0.98085