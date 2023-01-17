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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tpot.builtins import StackingEstimator
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score

np.random.seed(8)

# NOTE: Make sure that the class is labeled 'target' in the data file
X = pd.read_csv("../input/all_train.csv")
y = X["has_parkinson"]
X.drop(["has_parkinson"], axis=1,inplace=True)

selector = SelectKBest(k=300)
selector.fit(X, y)
# Get idxs of columns to keep
mask = selector.get_support()
X = X[X.columns[mask]]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

selector = SelectKBest(k=300)
selector.fit(X_train, y_train)
# Get idxs of columns to keep
mask = selector.get_support()
X_train = X_train[X_train.columns[mask]]
X_test = X_test[X_train.columns[mask]]

# Score on the training set was:0.8877777777777778
exported_pipeline = make_pipeline(
    StandardScaler(),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.8, min_samples_leaf=18, min_samples_split=15, n_estimators=100)),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    GaussianNB()
)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)
print(accuracy_score(results,y_test))

