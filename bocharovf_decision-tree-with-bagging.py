import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.ensemble import BaggingClassifier

from sklearn import cross_validation
%matplotlib inline
def showDigit (pixelArray):

    pixelMatrix = pixelArray.reshape( (28, 28) )

    plt.imshow(pixelMatrix, cmap='Greys')

    plt.show()



def decisionTreeWithBagging(X, Y):

    clf = BaggingClassifier(max_samples=0.5, max_features=0.5)

    clf = clf.fit(X, Y)

    return clf



def prepareSubmission(name, clf, test):

    prediction = clf.predict(test)

    df = pd.DataFrame(prediction, columns = ['Label'])

    df.index += 1 

    df.to_csv(name, index_label = "ImageId")



def crossValidation(clf, X, Y):

    scores = cross_validation.cross_val_score(clf, X, Y, cv=5)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# load data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# extract labels and features 

Y = train.ix[:,0].as_matrix()

X = train.ix[0:,1:].as_matrix()
# prepare classifier

tree = decisionTreeWithBagging(X, Y)
# check accuracy

crossValidation(tree, X, Y)
# prepareSubmission

prepareSubmission('decisionTree.bagging.csv', tree, test)