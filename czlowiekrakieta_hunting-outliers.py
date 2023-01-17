# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



y = train.pop('y')

ID = train.pop('ID')
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.linear_model import RandomizedLogisticRegression

from sklearn.naive_bayes import BernoulliNB

from sklearn.base import TransformerMixin

from sklearn.preprocessing import FunctionTransformer

from sklearn.pipeline import make_pipeline, make_union
import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
ints = train.select_dtypes(['int']).columns.tolist()

objs = train.select_dtypes(['object']).columns.tolist()



for col in ints:

    if np.var(train[col])==0:

        train.pop(col)

        ints.remove(col)

outs = (y>120).as_matrix().astype(int)



#let's define outliers as labels greater than 120
def evaluate(y_true, pred, thresh=.5):

    print('precision', precision_score(y_true, pred[:, 1]>thresh))

    print('recall', recall_score(y_true, pred[:, 1]>thresh))

    print('roc', roc_auc_score(y_true, pred[:, 1]))

    print('f1', f1_score(y_true, pred[:, 1]>thresh))
cv_preds = cross_val_predict(BernoulliNB(), train[ints], outs, cv=10, method='predict_proba')
evaluate(outs, cv_preds)
pip = make_pipeline(RandomizedLogisticRegression(C=5), BernoulliNB())



selection_preds = cross_val_predict(pip, train[ints], outs, cv=10, method='predict_proba')

evaluate(outs, cv_preds)
class OutlierThresholder(TransformerMixin):

    

    def __init__(self, thresh=1.5):

        self.th = thresh

    

    def fit(self, X, y):

        

        X = np.asarray(X)

        maps = []

        for col in range(X.shape[1]):

            

            val = X[:, col].copy()

            useful = []

            not_useful = []

            for u in np.unique(X[:, col]):

                

                o, no = y[val==u].mean(), y[val!=u].mean()

                q = o/no if no else 0

                

                if q > self.th:

                    useful.append(u)

                else:

                    not_useful.append(u)

                    

            col_map = dict(zip(useful+not_useful, [0]*len(useful)+[1]*len(not_useful)))

            maps.append(col_map)

            

        self.maps = maps

        return self

        

    def transform(self, X, y=None):

        

        X = X.copy()

        X = np.asarray(X)

        for col in range(X.shape[1]):

            

            X[:, col] = [self.maps[col][x] if x in self.maps[col] else 1 for x in X[:, col]]

            

        return X
def sel_obj(X):

    return X[:, :8]



def sel_ints(X):

    return X[:, 8:]
pip = make_pipeline(OutlierThresholder(), BernoulliNB())



outlier_obj_preds = cross_val_predict(pip, train[objs], outs, method='predict_proba', cv=10)

evaluate(outs, outlier_obj_preds)
un = make_union(make_pipeline(FunctionTransformer(sel_obj), OutlierThresholder()), FunctionTransformer(sel_ints))



for col in objs:

    train[col] = pd.factorize(train[col])[0]



binary_with_obj = make_pipeline(un, BernoulliNB())
full_preds = cross_val_predict(binary_with_obj, train, outs, method='predict_proba', cv=10)
evaluate(outs, full_preds)
upd_binary_with_obj = make_pipeline(un, RandomizedLogisticRegression(C=5), BernoulliNB())



full_upd_preds = cross_val_predict(upd_binary_with_obj, train, outs, method='predict_proba', cv=10)

evaluate(outs, full_upd_preds)
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
plt.plot(*calibration_curve(outs, full_upd_preds[:, 1], n_bins=5)[::-1])

plt.xlabel('mean predicted probability')

plt.ylabel('percent of correctly assigned labels')

plt.show()
plt.plot(*calibration_curve(outs, outlier_obj_preds[:, 1], n_bins=5)[::-1])

plt.xlabel('mean predicted probability')

plt.ylabel('percent of correctly assigned labels')

plt.show()
plt.plot(*calibration_curve(outs, cv_preds[:, 1], n_bins=5)[::-1])

plt.xlabel('mean predicted probability')

plt.ylabel('percent of correctly assigned labels')

plt.show()
plt.plot(*calibration_curve(outs, selection_preds[:, 1], n_bins=5)[::-1])

plt.xlabel('mean predicted probability')

plt.ylabel('percent of correctly assigned labels')

plt.show()
from xgboost import XGBRegressor

from functools import partial



xgb_params = dict(max_depth=3, learning_rate=0.05, n_estimators=100, subsample=.7, colsample_bytree=.7)

xgbr = XGBRegressor(**xgb_params)

my_cv = partial(cross_val_score, scoring='r2', cv=10)

cv_ordinary = my_cv(xgbr, train, y)

cv_add = my_cv(xgbr, np.hstack([train, cv_preds[:, 1].reshape(-1, 1)]), y)
cv_ordinary.mean(), cv_add.mean()