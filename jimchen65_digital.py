# -*- encoding: utf-8 -*-

import pandas as pd

import numpy as np

import csv

from six.moves import xrange

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



def evaluate_classifier(clf, data, target, split_ratio):

    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)

    clf.fit(trainX, trainY)

    return clf.score(testX,testY)



df = pd.read_csv('../input/train.csv')

train_label = df["label"]

train_data = df.drop("label",1)



n_estimators = 100

clf = RandomForestClassifier(n_estimators = n_estimators, n_jobs=1, criterion="gini")

score = evaluate_classifier(clf, train_data.loc[0:], train_label.loc[0:], 0.8)

print(score)
# output csv file

test_file = pd.read_csv('../input/test.csv')

result = clf.predict(test_file)



with open('test_result.csv', 'w') as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow(["ImageId", "Label"])

    data=[(i+1, result[i]) for i in xrange(len(result))]

    print(data)

    writer.writerows(data)