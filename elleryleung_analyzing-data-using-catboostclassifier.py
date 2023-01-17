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
import matplotlib.pyplot as plt

%matplotlib inline



from sklearn import metrics
#The function used in most kernels

def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)



# a - actual, p - predict

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



gini_sklearn = metrics.make_scorer(gini_normalized, True, True)
# Read file

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print("[Train] Num rows: {}, num cols: {}".format(train.shape[0], train.shape[1]))

print("[Test] Num rows: {}, num cols: {}".format(test.shape[0], test.shape[1]))
train.head()
test.head()
# any() applied twice to check run the isnull check across all columns.

train.isnull().any().any()
train_no_id_no_target = train.drop(['id', 'target'], axis=1)



cat_cols = [c for c in train_no_id_no_target if '_cat' in c]

cat_cols_idx = [i for i, c in enumerate(train_no_id_no_target) if '_cat' in c]

print("[Train] The number of category columns: {}".format(len(cat_cols)))

print("[Train] Index of category columns: {}".format(cat_cols_idx))

print("-----")



bin_cols = [c for c in train_no_id_no_target if '_bin' in c]

bin_cols_idx = [i for i, c in enumerate(train_no_id_no_target) if '_bin' in c]

print("[Train] The number of binary columns: {}".format(len(bin_cols)))

print("[Train] Index of binary columns: {}".format(bin_cols_idx))

print("-----")



else_cols = [c for c in train_no_id_no_target if '_bin' not in c and '_cat' not in c] 

print("[Train] The number of columns that are neither binary or category: {}".format(len(else_cols)))

print("[Train] Total columns: {}".format(len(train_no_id_no_target.columns)))
test_no_id = test.drop(['id'], axis=1)



test_cat_cols = [c for c in test_no_id if '_cat' in c]

test_cat_cols_idx = [i for i, c in enumerate(test_no_id) if '_cat' in c]

print("[Test] The number of category columns: {}".format(len(test_cat_cols)))

print("[Test] Index of category columns: {}".format(test_cat_cols_idx))

print("-----")



test_bin_cols = [c for c in test_no_id if '_bin' in c]

test_bin_cols_idx = [i for i, c in enumerate(test_no_id) if '_bin' in c]

print("[Test] The number of binary columns: {}".format(len(test_bin_cols)))

print("[Test] Index of binary columns: {}".format(test_bin_cols_idx))

print("-----")



test_else_cols = [c for c in test_no_id if '_bin' not in c and '_cat' not in c] 

print("[Test] The number of columns that are neither binary or category: {}".format(len(test_else_cols)))

print("[Test] Total columns: {}".format(len(test_no_id.columns)))
for c in cat_cols:

    print("[Category] Col name: {}".format(c))

    print("- Unique values: {}".format(np.sort(train[c].unique())))    

    print("# of Missing(-1): {}".format((train[c] == -1).sum()))

    print("--------------")
from catboost import CatBoostClassifier, Pool
train_pool = Pool(data=train.drop(['id', 'target'], axis=1), label=train['target'], cat_features=cat_cols_idx)

test_pool  = Pool(data=test.drop(['id'], axis=1), cat_features=test_cat_cols_idx)
model = CatBoostClassifier(iterations=10, learning_rate=0.01, verbose=True, loss_function='CrossEntropy')

model.fit(train_pool)



result = model.predict_proba(test_no_id)
print("proba = ", result)