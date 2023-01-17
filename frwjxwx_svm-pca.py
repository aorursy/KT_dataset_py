import pandas as pd

from sklearn import svm, decomposition, preprocessing



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_x = train.values[:, 1:]

train_y = train.ix[:, 0]

COMPONENTS_RATIO = 0.8

train_x = preprocessing.scale(train_x)

pca = decomposition.PCA(n_components=COMPONENTS_RATIO, whiten=False)

train_x = pca.fit_transform(train_x)
train_x.shape
classifier = svm.SVC(C=2.0)

classifier.fit(train_x, train_y)
test_x = preprocessing.scale(test.values)

test_x = pca.transform(test_x)

test_pred = classifier.predict(test)