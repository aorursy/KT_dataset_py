# Toy example for getting started with machine learning



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/creditcard.csv')

col = data.columns

num_cols = len(col)

num_rows = len(data.index)



x = data[col[1:9]]

y = data[col[num_cols-1]]



clf = svm.SVC(kernel='rbf',C=1)



cutoff= int(num_rows*0.7)



x_train = x[500:15000]

y_train = y[500:15000]



x_test = pd.concat([x[1:500],x[15000:num_rows-1]])

y_test = pd.concat([y[1:500],y[15000:num_rows-1]])



scores = cross_val_score(clf, x_train, y_train, cv = 5)

print("5-fold crossvalidation scores: ", scores)

print("5-fold crossvalidation scores: ", scores.mean())



clf.fit(x_train, y_train)



y_predict = clf.predict(x_test)



print("AUC-ROC score: ", roc_auc_score(y_test, y_predict))

print("Accuracy: ", accuracy_score(y_test, y_predict))

print("Precision: ", precision_score(y_test, y_predict))

print("Recall: ", recall_score(y_test, y_predict))



# Any results you write to the current directory are saved as output.