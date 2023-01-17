# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#from sklearn import svm, metrics, neural_network

data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
data.head()

training_data = data[:30000]

training_label = training_data['label']

training_data = training_data.drop(['label'], axis=1)

test_data = data[30001:42000]

test_label = test_data['label']

test_data = test_data.drop(['label'], axis=1)

test_data.head()

data.shape
from sklearn import svm, metrics

classifier = sklearn.svm.LinearSVC()



# Learning Phase

classifier.fit(training_data, training_label)



# Predict Test Set

predicted = classifier.predict(test_data)



# classification report

print("Classification report for classifier %s:n%sn" % (classifier, metrics.classification_report(test_label, predicted)))



# confusion matrix

print("Confusion matrix:n%s" % metrics.confusion_matrix(test_label, predicted))



classifier.score(test_data, test_label)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha= 0.0005, 

                    hidden_layer_sizes=(125), random_state=1)



clf.fit(training_data, training_label)



# Predict Test Set

predicted = clf.predict(test_data)



# classification report

print("Classification report for classifier %s:n%sn" % (clf, metrics.classification_report(test_label, predicted)))



# confusion matrix

print("Confusion matrix:n%s" % metrics.confusion_matrix(test_label, predicted))



clf.score(test_data, test_label)

from sklearn import svm, metrics

classifier = svm.SVC(kernel='poly', degree=2)



# Learning Phase

classifier.fit(training_data, training_label)



# Predict Test Set

predicted = classifier.predict(test_data)



# classification report

print("Classification report for classifier %s:n%sn" % (classifier, metrics.classification_report(test_label, predicted)))



# confusion matrix

print("Confusion matrix:n%s" % metrics.confusion_matrix(test_label, predicted))



classifier.score(test_data, test_label)
preds = classifier.predict(test_data)



export = pd.DataFrame({"ImageId": list(range(1,len(preds)+1)),

                         "Label": preds})

export.to_csv("results.csv", index=False, header=True)