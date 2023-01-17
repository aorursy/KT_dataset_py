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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

X = training.iloc[:,:-1]
y = training.TARGET

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

selectK = SelectKBest(f_classif, k=220)
selectK.fit(X, y)
X_sel = selectK.transform(X)

features = X.columns[selectK.get_support()]
print (features)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel, y, random_state=1301)
features[0]
training['var3'].value_counts()
training.loc[training['var3'] != -999999, 'var3'].hist(bins=1000)
X_train = training.loc[training['var3'] != -999999, features[1:]]
y_train = training.loc[training['var3'] != -999999, 'var3']
X_test = training.loc[training['var3'] == -999999, features[1:]]
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)
y_test
# Check in what columns value -999999 appears
print ([f for f in X.columns if X[f][X[f] == -999999].shape[0] > 0])
#Replace value -999999 in column var3
X.var3 = X.var3.replace(-999999,2)
