# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def read(path):
    return pd.read_csv(path)

def col_delete(a,indxList):
    return np.delete(a,indxList)

def getAccuracy(y_pred,y_true,n):
    c = 0
    for i, j in zip(y_pred, y_true):
        if i == j:
            c = c + 1
    return (c/n)*100


import os
print(os.listdir("../input"))


train =pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train = train.interpolate(method='values')
test = test.interpolate(method='values')

col_names = train.columns.values
col_names_test = test.columns.values
col_names_data_test = col_delete(col_names_test,[0,1,10])
col_names_data = col_delete(col_names,[0,1,10,21])
col_name_target = col_names[21]

train_data = train[col_names_data]
train_target = train[col_name_target]
test_data = test[col_names_data_test]

# train_data = train_data.apply(lambda x: x/x.max(), axis=1)

# print(train_data)
train_data = 100 * (train_data - train_data.min()) / (train_data.max() - train_data.min())


data_train = train_data.values
sc = StandardScaler()
data_train = sc.fit_transform(data_train)
target_train = train_target.values

test_data = 100 * (test_data - test_data.min()) / (test_data.max() - test_data.min())
test_data = test_data.values

## Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

classifiers = [
    KNeighborsClassifier(3),
    svm.SVC(kernel="linear", C=0.5), #C=0.025
    svm.SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(random_state=1),
    BaggingClassifier(n_estimators=10, max_features=1)]

name_MyClassifier = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "Logistic Regression", "Bagging"]#, "Gaussian Process"]
# for name, clf in zip(name_MyClassifier, classifiers):
#     clf.fit(data_train[0:700],target_train[0:700])
#     y_pred = clf.predict(data_train[700:901])
#     accuacy_svm = getAccuracy(y_pred, target_train[700:901], 200)
#     print("SVM Accuracy:",name, accuacy_svm)
#Voting by all classifiers
ZipList = list(zip(name_MyClassifier, classifiers))
clf_Vot = VotingClassifier(estimators = ZipList, voting='hard')
clf_Vot.fit(data_train,target_train)
pred_Votting = clf_Vot.predict(test_data)
print("labels of prediction by ensumble of Nearest Neighbors, Linear SVM, RBF SVM,Decision Tree, Random Forest, Neural Net, AdaBoost,Naive Bayes, QDA, Logistic Regression, Bagging\n :")
print(pred_Votting)
# accuacy_svm = getAccuracy(pred_Votting, target_train[700:901],200)
# print("SVM Accuracy:",accuacy_svm)

# clf = svm.SVC(kernel='linear',C=0.025)

# clf.fit(data_train[0:700],target_train[0:700])
# pred_SVM = clf.predict(data_train[700:901])
# accuacy_svm = getAccuracy(pred_SVM, target_train[700:901],200)
# print("SVM Accuracy:",accuacy_svm)

cols = {"PlayerID": [i+901 for i in range(440)], "TARGET_5Yrs": pred_Votting}
submission = pd.DataFrame(cols)
# print(submission)
submission.to_csv("submission.csv",index=False)

submission
