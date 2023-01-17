from sklearn.datasets import load_iris

from sklearn.datasets import load_boston

from sklearn.preprocessing import normalize



from sklearn.feature_selection import f_regression

from sklearn.feature_selection import f_classif



from sklearn.feature_selection import mutual_info_classif 

from sklearn.feature_selection import mutual_info_regression



from sklearn.feature_selection import VarianceThreshold



from sklearn.linear_model import Lasso



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
X, y = load_iris(return_X_y=True)
level_sign = 0.05

np.where(f_classif(X, y)[1] < level_sign)[0] 
def add_random_col(X, num_cols=2):

    col_rand = np.random.rand(X.shape[0], num_cols).reshape(-1, num_cols)

    return np.concatenate((X, col_rand), axis=1)
# add 2 more random columns with random values

X_new = add_random_col(X)

X_new.shape, X.shape
np.where(f_classif(X_new, y)[1] < level_sign)[0] 
X, y = load_boston(return_X_y=True)
np.where(f_regression(X, y)[1] < level_sign)[0]
# also add new random columns

X_new = add_random_col(X)

X_new.shape, X.shape
np.where(f_regression(X_new, y)[1] < level_sign)[0]
X, y = load_iris(return_X_y=True)

mutual_info_classif(X, y, random_state=42)
X_new = add_random_col(X)

mutual_info_classif(X_new, y, random_state=42)
X, y = load_boston(return_X_y=True)

mutual_info_regression(X, y, random_state=42)
X_new = add_random_col(X)

mutual_info_regression(X_new, y, random_state=42)
threshold = 0.3

np.where(mutual_info_regression(X_new, y) > threshold)[0]
X, _ = load_iris(return_X_y=True)

# need to normalize to make std equal to 1

X_norm = normalize(X, norm='l2', axis=0)

var_threshold = VarianceThreshold(threshold=0.001)
var_threshold.fit(X_norm)

var_threshold.get_support()
X, y = load_boston(return_X_y=True)

lasso = Lasso()

lasso.fit(X, y)
lasso.coef_
X_new = add_random_col(X)

lasso.fit(X_new, y)
lasso.coef_
threshold = 0.05

np.where(abs(lasso.coef_) > threshold)[0]
k = 10

selectKBest = SelectKBest(score_func=chi2, k=10)

selectKBest.fit(X_new, y.astype('int'))

np.array(pd.DataFrame(selectKBest.scores_).sort_values(by=0, ascending=False).iloc[:k, :].index)
plt.figure(figsize=(20,20))

sns.heatmap(pd.DataFrame(X_new).corr(),annot=True,cmap="RdYlGn");