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
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score,roc_curve, auc

import matplotlib.pyplot as plt



data = pd.read_csv("../input/creditcard.csv")

X = data.iloc[:,:-1].values

y = data['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

''' Applying the RandomForest algorithm'''



clf = RandomForestClassifier(n_estimators = 100,n_jobs = -1)



clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

predict_prob = clf.predict_proba(X_test)

'''Using cross-validation technique to check the performance of my model and using area under the roc curve as my performance metrics'''

scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'f1')

fpr, tpr, thresholds = roc_curve(y_test, predict_prob[:,1] )

roc_area = auc(fpr, tpr)



''' other metrics for this model'''

test_scores = f1_score(y_test, predictions)

recall = recall_score(y_test, predictions)

precision = precision_score(y_test, predictions)

accuracy = accuracy_score(y_test, predictions)

''' Plotting the roc curve '''



plt.title('Receiver operating characteristics')

plt.plot(fpr, tpr, 'b')

plt.legend(loc='lower right')

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0,1.0])



plt.ylabel('Recall')

plt.xlabel('Fall-out')

plt.show()



print("The average cross validation score with f1 metrics  is {}".format(scores.mean()))

print("The area under the roc curve is {}".format(roc_area))

print("The accuracy of model is {}".format(accuracy))

print("The average test score is {}".format(test_scores))

print("The recall score for test data is {}".format(recall))

print("The precision score on test data is {}".format(precision))


