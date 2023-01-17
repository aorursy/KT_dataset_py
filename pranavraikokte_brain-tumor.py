# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn import metrics

from sklearn.externals import joblib

import cv2

import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/braintumorfeaturesextracted/Dataset.csv', delimiter=',')
positive = data.loc[data['Class']==1]

negative = data.loc[data['Class']==0]

positive = positive[10:105]
cleaned_data = pd.concat([positive, negative])

cleaned_data = cleaned_data.drop(columns=['Eccentricity'])
X = cleaned_data.iloc[:,1:8]

Y = cleaned_data.iloc[:,-1]
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.25)
from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB().fit(xtrain,ytrain)

pred_MNB = MNB.predict(xtest)

acc_MNB = metrics.accuracy_score(ytest, pred_MNB)*100

#joblib.dump(MNB, 'Brain - NB.pkl')



print('1. Using Naive Bayes Method')

print('Accuracy - {}'.format(acc_MNB))

print('Recall - {}'.format(metrics.recall_score(ytest, pred_MNB)))

print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_MNB)))

print('Confusion matrix')

print(metrics.confusion_matrix(ytest, pred_MNB))

print('\n')
from sklearn import svm

SVM = svm.LinearSVC(dual=False)

SVM.fit(xtrain, ytrain)

pred_svm = SVM.predict(xtest)

acc_svm = metrics.accuracy_score(ytest, pred_svm)*100

#joblib.dump(SVM, 'Brain - SVM.pkl')



print('2. Using SVM Method')

print('Accuracy - {}'.format(acc_svm))

print('Recall - {}'.format(metrics.recall_score(ytest, pred_svm)))

print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_svm)))

print('Confusion matrix')

print(metrics.confusion_matrix(ytest, pred_svm))

print('\n')
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()

RFC.fit(xtrain, ytrain)

pred_rfc = RFC.predict(xtest)

acc_rfc = metrics.accuracy_score(ytest, pred_rfc)*100

#joblib.dump(RFC, 'Brain - RFC.pkl')



print('3. Using RandomForestClassifier Method')

print('Accuracy - {}'.format(acc_rfc))

print('Recall - {}'.format(metrics.recall_score(ytest, pred_rfc)))

print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_rfc)))

print('Confusion matrix')

print(metrics.confusion_matrix(ytest, pred_rfc))

print('\n')