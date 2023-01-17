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
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
%matplotlib inline
import gc
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
# read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Clean duplicated features
remove = []
c = train.columns
for i in range(len(c)-1):
    v = train[c[i]].values
    for j in range(i+1, len(c)):
        if np.array_equal(v, train[c[j]].values):
            remove.append(c[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
# remove features with 0 variance
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)
# common feature engineering on Kaggle forum
features = train.columns[1:-1]
train.insert(1, 'SumZeros', (train[features] == 0).astype(int).sum(axis=1))
test.insert(1, 'SumZeros', (test[features] == 0).astype(int).sum(axis=1))

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
features = train.columns[1:-1]
pca = PCA(n_components=2)
x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
x_test_projected = pca.transform(normalize(test[features], axis=0))
train.insert(1, 'PCAOne', x_train_projected[:, 0])
train.insert(1, 'PCATwo', x_train_projected[:, 1])
test.insert(1, 'PCAOne', x_test_projected[:, 0])
test.insert(1, 'PCATwo', x_test_projected[:, 1])
# There are 309 features in total, we are definitely not going to visualize them all
# How to find the important features?
features = train.columns[1:-1]
features
# My way is to run a fast and dirty XGB, and see what are the most important contributors
# The assumption is XGB will show underlying if then rules
# Chi2 and other might works
# but as far as I know, it cannot reveal if then relationship between features
# Let me know if there is any other ways.

# The following code is from:
# https://www.kaggle.com/scirpus/santander-customer-satisfaction/python-xgb-lb-41047/discussion
# Change a bit to make it faster

split = 3 # 3 fold is faster than 10 fold
skf = StratifiedKFold(train.TARGET.values,
                      n_folds=split,
                      shuffle=False,
                      random_state=42)

train_preds = None
test_preds = None
visibletrain = blindtrain = train
index = 0
print('Change num_rounds to 350')
num_rounds = 30
params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.3 # larger learning rate
params["subsample"] = 0.2
params["colsample_bytree"] = 0.7
params["silent"] = 0
params["max_depth"] = 5
params["min_child_weight"] = 6
params["eval_metric"] = "auc"
params["gamma"] = 0
for train_index, test_index in skf:
    visibletrain = train.iloc[train_index]
    blindtrain = train.iloc[test_index]
    dvisibletrain = \
    xgb.DMatrix(csr_matrix(visibletrain[features]),
                visibletrain.TARGET.values,
                silent=True)
    dblindtrain = \
    xgb.DMatrix(csr_matrix(blindtrain[features]),
                blindtrain.TARGET.values,
                silent=True)
    watchlist = [(dblindtrain, 'eval'), (dvisibletrain, 'train')]
    clf = xgb.train(params, dvisibletrain, num_rounds,
                    evals=watchlist, early_stopping_rounds=50,
                    verbose_eval=False)

    blind_preds = clf.predict(dblindtrain)
    
    index = index+1
    del visibletrain
    del blindtrain
    del dvisibletrain
    del dblindtrain
    gc.collect()
    dfulltrain = \
    xgb.DMatrix(csr_matrix(train[features]),
                train.TARGET.values,
                silent=True)
    dfulltest = \
    xgb.DMatrix(csr_matrix(test[features]),
                silent=True)
    if(train_preds is None):
        train_preds = clf.predict(dfulltrain)
        test_preds = clf.predict(dfulltest)
    else:
        train_preds *= clf.predict(dfulltrain)
        test_preds *= clf.predict(dfulltest)
        del dfulltrain
        del dfulltest
        #  del clf # we need the clf to extract useful features
        gc.collect()

    train_preds = np.power(train_preds, 1./index)
    test_preds = np.power(test_preds, 1./index)
    print("")
    print("mean AUC: %s +/- %s;" % (np.mean(AUCs),np.std(AUCs)))
    print('Average ROC:', roc_auc_score(train.TARGET.values, train_preds))
# Extract the useful features
# Sort by the contribution to the XGB
a = pd.DataFrame([clf.get_fscore().keys(),clf.get_fscore().values()]).T
a['featureID'] = a[0].apply(lambda x: int(x[1:]))
a['Feature'] = a.featureID.apply(lambda x: features[x])
a.columns = ['x','Imp','y',"feature"]
a = a[['feature',"Imp"]].sort("Imp",ascending = False)
a.head()
# Let's plot the graph!
# select two features as x and y, plot it as scatter plot
# black is TARGET == 1; white is TARGET == 0
subset = train.copy().sort('TARGET',ascending = True) # rank TARGET == 1 to the bottom,
                                                      # so it will plot at the front,
                                                      # and won't be blocked by TARGET == 0
k = 0
for i in range(len(a.feature.values)-1):
    for j in range(i+1,len(a.feature.values)):
        k += 1 # you may comment this in your local ipynb
        if (k == 604): # you may comment this in your local ipynb
            subset.plot(kind = 'scatter', x = a.feature.values[i], y = a.feature.values[j], c = subset.TARGET)
            plt.title(str(k)+": " + a.feature.values[i] + " -- " +a.feature.values[j])
            plt.show()
        if (k > 604): # you may comment this in your local ipynb
            break # you may comment this in your local ipynb
# Interesting to fin that num_var35 is negatively correlated to SumZeros
# lots of other similar relationships among different features
