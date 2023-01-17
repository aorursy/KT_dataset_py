import pandas as pd

import numpy as np

from sklearn import svm



# loading the data

testdata = pd.read_csv("../input/test_binary.csv")

traindata = pd.read_csv("../input/training_binary.csv")

X = traindata.drop(['cuisine', 'id'],axis=1)

y = traindata['cuisine']

Xtest = testdata.drop(['cuisine', 'id'],axis=1)

ytest = testdata['cuisine']



lin_clf_default = svm.LinearSVC()

lin_clf_default.fit(X,y)



predictions = lin_clf_default.predict(Xtest)



from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, support = precision_recall_fscore_support(ytest, predictions, average='weighted')

accuracy = lin_clf_default.score(Xtest,ytest)

print('Test: Precision is {}, recall is {}, accuracy = {}'.format(precision, recall, accuracy))



predictionstrain = lin_clf_default.predict(X)

precision, recall, fscore, support = precision_recall_fscore_support(y, predictionstrain, average='weighted')

accuracy = lin_clf_default.score(X,y)

print('Train: Precision is {}, recall is {}, accuracy = {}'.format(precision, recall, accuracy))
# determining the optimal value for C

valC = np.linspace(0.001,1,50)

accuracytest = []

accuracytrain = []

for c in valC:

    lin_clf = svm.LinearSVC(multi_class = 'ovr', C = c)

    lin_clf.fit(X, y)

    predictionstrain = lin_clf.predict(X)

    predictions = lin_clf.predict(Xtest)

    accuracytest.append(lin_clf.score(Xtest,ytest))

    accuracytrain.append(lin_clf.score(X,y))
%matplotlib inline

import matplotlib.pyplot as plt



# plot the accuracy of the training and test set against the values of C

plt.figure()

plt.plot(valC,accuracytest,label = 'test')

plt.plot(valC,accuracytrain, label = 'train')

plt.legend()

plt.xlabel('c')

plt.ylabel('accuracy')

plt.show()
import pandas as pd

import numpy as np

from sklearn import svm



# training the classifier with the optimal value

lin_clf = svm.LinearSVC(C= 0.1, multi_class = 'ovr')

lin_clf.fit(X, y)

predictions = lin_clf.predict(Xtest)
# calculating the precision, recall and accuracy for the test set

from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, support = precision_recall_fscore_support(ytest, predictions, average='weighted')

accuracy = lin_clf.score(Xtest,ytest)

print('Test: Precision is {}, recall is {}, accuracy = {}'.format(precision, recall, accuracy))
# calculating the precision, recall and accuracy for the training set

predictionstrain = lin_clf.predict(X)

precision, recall, fscore, support = precision_recall_fscore_support(y, predictionstrain, average='weighted')

accuracy = lin_clf.score(X,y)

print('Train: Precision is {}, recall is {}, accuracy = {}'.format(precision, recall, accuracy))
# creating the confusion matrix for the test set

from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(ytest,predictions,labels=list(set(ytest))),index=list(set(ytest)),columns=list(set(ytest)))
# creating the confusion matrix for the training set

from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(y,predictionstrain,labels=list(set(y))),index=list(set(y)),columns=list(set(y)))