import numpy as np

from sklearn import metrics

from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split 

import pandas as pd

from sklearn.metrics import roc_curve, auc

from sklearn import datasets

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

import sklearn.metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.colors as colors

import seaborn as sns

import itertools

from mlxtend.evaluate import lift_score

from scipy.stats import norm

import scipy.stats

from sklearn.metrics import classification_report

def splitDataset(dataset, splitRatio=0.75):

    trainSize = int(len(dataset) * splitRatio)

    trainSet = []

    copy = list(dataset)

    while len(trainSet) < trainSize:

        index = random.randrange(len(copy))

        trainSet.append(copy.pop(index))

    return [trainSet, copy]

iris = pd.read_csv('../input/iris4/iris4.csv')

X = iris.drop(['species', 'Id'], axis=1)

y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

model = abc.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
pd.crosstab(y_test,y_pred)
metrics.f1_score(y_test, y_pred,labels=None, pos_label=1, average='weighted',sample_weight=None)
cm = confusion_matrix(np.array(y_test), np.array(y_pred))

print(cm)
print (classification_report(y_test, y_pred))
sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])

print('Sensitivity1 : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])

print('Specificity : ', specificity1)

#new

sensitivity2 = cm[1,1]/(cm[1,1]+cm[1,2])

print('Sensitivity2 : ', sensitivity2)

specificity2 = cm[2,2]/(cm[2,1]+cm[2,2])

print('Specificity2 : ', specificity2)

iris = datasets.load_iris()

X, y = iris.data, iris.target



y = label_binarize(y, classes=[0,1,2])

n_classes = 3



X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.75, random_state=50)



clf = OneVsRestClassifier(LinearSVC(random_state=0))

y_score = clf.fit(X_train, y_train).decision_function(X_test)



fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(n_classes):

    plt.figure()

    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()