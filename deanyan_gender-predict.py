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

import os

import sklearn

import time
%matplotlib inline



try:

    data = pd.read_csv('../input/voice.csv')

    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))

except:

    print('Dataset could not be loaded. Is the dataset missing?')
from IPython.display import display
display(data.describe())
display(data.head())
label_row = data['label']

data = data.drop('label', axis = 1)

label = label_row.apply(lambda x: 1 if x == 'male' else 0)
print(label_row.head())

print(data.head(1))

print(label.head())
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

numerical = list(data.columns)

features = pd.DataFrame(scaler.fit_transform(data))

features.columns = numerical

display(features.head(1))
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size = 0.2, random_state = 0, stratify = label)



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0, stratify = Y_train)



print("Training set has {} samples.".format(X_train.shape[0]))

print("Validation set has {} samples.".format(X_val.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
from sklearn.metrics import fbeta_score, accuracy_score



def train_predict(learner, x_train, y_train, x_val, y_val):

    

    

    results = {}

    print(len(x_train), len(y_train))

    

    start = time.time()

    learner = learner.fit(x_train, y_train)

    end = time.time()

    

    results['train_time'] = end - start

    

    start = time.time()

    predictions_train = learner.predict(x_train[:300])

    predictions_val = learner.predict(x_val)

    end = time.time()

    

    results['pred_time'] = end - start

    

    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    results['acc_val'] = accuracy_score(y_val, predictions_val)

    

    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = 0.5)

    results['f_val'] = fbeta_score(y_val, predictions_val, beta = 0.5)

    

    print('{} trained on {} samples.'.format(learner.__class__.__name__, len(x_train)))

    

    return results
from sklearn import svm

from sklearn import neighbors

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



clf_svm = svm.SVC(random_state = 10)

clf_neighbors = neighbors.KNeighborsClassifier(n_neighbors = 5)

clf_logistic = LogisticRegression(random_state = 20)

clf_decisionTree = DecisionTreeClassifier(random_state=30)

clf_RandomForest = RandomForestClassifier(n_estimators = 50, max_depth = 5, random_state = 40)





results = {}

for clf in [clf_svm, clf_neighbors, clf_logistic, clf_decisionTree, clf_RandomForest]:

    clf_name = clf.__class__.__name__

    results[clf_name] = {}

    results[clf_name] = train_predict(clf, X_train, Y_train, X_val, Y_val)



# 对选择的三个模型得到的评价结果进行可视化

for result in results:

    print(result, results[result])
from sklearn.base import clone



clf_test = clone(clf_decisionTree).fit(X_train, Y_train)

pred = clf_test.predict(X_test)

    

print("Accuracy on validation data: {:.4f}".format(accuracy_score(Y_test, pred)))

print("F-score on validation data: {:.4f}".format(fbeta_score(Y_test, pred, beta = 0.5)))
submission = pd.DataFrame()

submission['index'] = Y_test.index

submission['sex'] = pred

submission.to_csv('output.csv', index=False)