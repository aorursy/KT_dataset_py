# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

dftest = pd.read_csv('../input/test.csv')

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

X_train = df.iloc[:,0:561]

labels = df.iloc[:,-1]

df['act'] = pd.factorize(labels)[0] + 1

y_train = df['act']

from sklearn import datasets, linear_model

from sklearn.svm import SVC

regr = SVC()

regr.fit(X_train,y_train)

X_test = dftest.iloc[:,0:561]

labels = dftest.iloc[:,-1]

dftest['act'] = pd.factorize(labels)[0] + 1

y_test = dftest['act']

y_pred = regr.predict(X_test)

target_names = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS',

       'WALKING_UPSTAIRS']



print(classification_report(y_test,y_pred,target_names=target_names))

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import roc_auc_score

clf = OneVsRestClassifier(LinearSVC(random_state=0))

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test,y_pred,target_names=target_names))
