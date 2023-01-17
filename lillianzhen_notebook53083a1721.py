# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats, integrate

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



%matplotlib inline



from sklearn import tree

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score, confusion_matrix



from IPython.display import Image # displaying images files in jupyter

from IPython.display import IFrame # displaying pdf file in jupyter
data = pd.read_csv('../input/creditcard.csv').dropna()
data.describe()
data['Class'] = data['Class']==1
##work with a regression tree

X_labels = [c for c in data.columns if c not in ['Time','Class']]

X = data.loc[:,X_labels]

Y = data['Class']
# Decision Tree

results = []

for d in range(1,8):

    clf = tree.DecisionTreeClassifier(max_depth=d)

    clf = clf.fit(X,Y)    

    scores = cross_val_score(clf, X, Y, cv=5)

    tn, fp, fn, tp = confusion_matrix(Y,clf.predict(X)).ravel()

    tp = precision_score(Y,clf.predict(X))

    print('depth %d, True positive %.2f,False negative %.2f, mean %.8f, '%(d,tp,fn,scores.mean()))

    results.append((d,tp,fn,scores.mean()))

    

df_tree_accuracy = pd.DataFrame(data=results,columns=['depth','True positive','False negative','scores mean'])

df_tree_accuracy





## For a credit card fraud problem, we can accecpt less False Negative which predict non-fraud when fraud happens



## For the regression tree method, depth 4 seems to be the best since it has a 

## lower false negative rate while the accurancy(to avoid over-fitting) with cross vilidation is high.
## Random Forest

clf_rf = RandomForestClassifier(max_features="sqrt", n_estimators=50, max_depth=1)

clf_rf = clf_rf.fit(X,Y)
results = []

for d in range(1,8):

    clf_rf = RandomForestClassifier(max_features="sqrt", n_estimators=10*d, max_depth=1)

    clf_rf = clf_rf.fit(X,Y)

    scores = cross_val_score(clf, X, Y, cv=5)

    tn, fp, fn, tp = confusion_matrix(Y,clf_rf.predict(X)).ravel()

    tp = precision_score(Y,clf_rf.predict(X))

    print('depth %d, True positive %.2f,False negative %.2f, mean %.8f, '%(d,tp,fn,scores.mean()))

    results.append((d,tp,fn,scores.mean()))

    

df_tree_accuracy = pd.DataFrame(data=results,columns=['depth','True positive','False negative','scores mean'])

df_tree_accuracy



## Random Forest didn't fit the model better than Regression Tree
res_boosting_mean = []

res_boosting_std = []

for n in n_range:

    clf_boosting = AdaBoostClassifier(n_estimators=n, learning_rate=0.5)

    clf_boosting_scores = cross_val_score(clf_boosting, X, Y, cv=5)

    res_boosting_mean.append(clf_boosting_scores.mean())

    res_boosting_std.append(clf_boosting_scores.std())

df_boosting = pd.DataFrame({'Boosting accuracy':res_boosting_mean,'Boosting error':res_boosting_std},index=n_range)
df_boosting