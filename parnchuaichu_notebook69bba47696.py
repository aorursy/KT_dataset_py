# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
X_train = pd.read_csv('/kaggle/input/titanic/train.csv')
y_train =  X_train['Survived']
X_test = pd.read_csv('/kaggle/input/titanic/test.csv')
y_test = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
y_test = y_test['Survived']

X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean()).round(0)
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean()).round(0)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

X_train['Sex'] = enc.fit_transform(X_train['Sex'])

X_test['Sex'] = enc.fit_transform(X_test['Sex'])
from sklearn.preprocessing import LabelBinarizer
lb_style = LabelBinarizer()

lb_results = lb_style.fit_transform(X_train['Pclass'])
encode = pd.DataFrame(lb_results, columns=['1st_class','2nd_class','3rd_class'])
X_train = pd.concat([X_train, encode], axis=1)

lb_results = lb_style.fit_transform(X_test['Pclass'])
encode = pd.DataFrame(lb_results, columns=['1st_class','2nd_class','3rd_class'])
X_test = pd.concat([X_test, encode], axis=1)
X_train =X_train[['Age','Sex','1st_class','2nd_class','3rd_class']]
X_test =X_test[['Age','Sex','1st_class','2nd_class','3rd_class']]
from sklearn import metrics
def measure_performance(X,y,clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred=clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y,y_pred),"\n")
    if show_confussion_matrix:
        print("Confussion matrix")
        print(metrics.confusion_matrix(y,y_pred),"\n")


from sklearn import tree
#clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_leaf=5)
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X_train,y_train)
measure_performance(X_train,y_train,clf_tree, show_classification_report=True, show_confussion_matrix=False)
from sklearn.naive_bayes import MultinomialNB
clf_bay = MultinomialNB()
clf_bay = clf_bay.fit(X_train,y_train)
measure_performance(X_train,y_train,clf_bay, show_classification_report=True, show_confussion_matrix=False)
from sklearn.neural_network import MLPClassifier
clf_nn = MLPClassifier(random_state=1, max_iter=300)
clf_nn = clf_nn.fit(X_train,y_train)
measure_performance(X_train,y_train,clf_nn, show_classification_report=True, show_confussion_matrix=False)
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator of k=5 folds
    cv = KFold(K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score >>> method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))
clfs = [clf_tree, clf_bay, clf_nn]
for clf in clfs:
    evaluate_cross_validation(clf, X_train, y_train, 5)