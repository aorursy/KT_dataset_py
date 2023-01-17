import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed = 45

import sklearn

import warnings

warnings.filterwarnings("ignore")

!pip install iterative-stratification

train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
COLS = ['cp_type','cp_dose']

FE = []

for col in COLS:

    for mod in train_features[col].unique():

        FE.append(mod)

        train_features[mod] = (train_features[col] == mod).astype(int)

del train_features['sig_id']

del train_features['cp_type']

del train_features['cp_dose']

FE+=list(train_features.columns) 

del train_targets_scored['sig_id']
X = np.array(train_features.to_numpy(), dtype=np.float)

y = np.array(train_targets_scored.to_numpy(), dtype=np.float)
from skmultilearn.model_selection import iterative_train_test_split

X_train, y_train, X_val, y_val = iterative_train_test_split(X, y, test_size = 0.1)

X_train.shape,y_train.shape,X_val.shape,y_val.shape
print(X_val.shape,y_val.shape)

print(X_val[:2])

y_val[:1]
k=4
for i in range(k):

    np.random.shuffle(X_train)

    np.random.shuffle(y_train)

    X_, y_, X_test, y_test = iterative_train_test_split(X_train,y_train, test_size = 0.2)

    print(X_.shape,y_.shape,X_test.shape,y_test.shape)

    print("model.fit(X_,y_)")

    print("model log loss evaluation for {} fold is {}".format(i+1,np.random.rand(1)))
print("hold out data of shape x: {} y: {} model log loss evaluation is {}".format(X_val.shape,y_val.shape,np.random.rand(1)))
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import numpy as np





mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True)



for train_index, test_index in mskf.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold



rmskf = RepeatedMultilabelStratifiedKFold(n_splits=2,n_repeats=2, random_state=0)



for train_index, test_index in rmskf.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit



msss  = MultilabelStratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)



for train_index, test_index in msss.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)