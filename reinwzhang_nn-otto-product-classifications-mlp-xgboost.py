import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
data.columns[1: len(data.columns) -1]
X = data[data.columns[1:len(data.columns) - 1]]
y = np.ravel(data['target'])
class_cnt = len(data.groupby('target'))
X_corr=X.corr()
X_corr.idxmin(axis=0)
#check feature correlatins
import seaborn as sns
sns.heatmap(X.corr())
# X_corr.iloc[i] for i in range(X_corr.shape[0])
y_percentile = data.groupby('target').size()/len(y)*100
y_percentile.plot(kind='bar')
plt.ylabel('percentage')
plt.xlabel('target')
plt.show()
for i in range(class_cnt):
    plt.subplot(3, 3, i+1)
    data[data.target == 'Class_' + str(i+1)].feat_20.hist()
plt.show()
plt.scatter(data.feat_19, data.feat_20)
from matplotlib import *

fig = plt.figure()
ax = fig.add_subplot(221)
cax = ax.matshow(X.corr(), interpolation = 'nearest')
plt.show()
num_fea = X.shape[1]
print(num_fea)
from sklearn.model_selection import GridSearchCV, StratifiedKFold,KFold
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

import xgboost as xgb
#alpha is L-2 regularization coefficient
nn_ml = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (30, 10), random_state = 1, verbose = True)
nn_ml.fit(X, y)
print(nn_ml.intercepts_)
print(nn_ml.coefs_[0].shape)
print(nn_ml.coefs_[1].shape)
print(nn_ml.coefs_[2].shape)
pred = nn_ml.predict(X)
print(nn_ml.score(X, y))
print(sum(pred == y) / len(y))
testdf = pd.read_csv('../input/test.csv')
testdata = testdf[testdf.columns[1:]]
testdf.head()
prob_test = nn_ml.predict_proba(testdata)
print(prob_test)
cols = []
cols = cols + list('Class_{}'.format(str(i+1)) for i in range(9))
pred_df = pd.DataFrame(prob_test, columns=cols)
pred_df['id'] = testdf['id']
cols = pred_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
solution = pred_df[cols]
solution.to_csv('./solutions.csv', index = False)
kfold = StratifiedKFold(n_splits = 3, random_state = 42)
mlp = MLPClassifier(random_state = 42)
mlp_grid= {'hidden_layer_sizes': [30],
           'activation': ['relu'],
           'solver': ['adam'],
           'alpha' : [0.3, 0.1],
           'learning_rate': ['constant'],
           'max_iter': [1000],
           'batch_size' :[40]
           }
mlp_cv = GridSearchCV(mlp, param_grid = mlp_grid, verbose = 1, cv = kfold, n_jobs = 5, scoring = 'accuracy')
mlp_cv.fit(X, y)
pred_mlpcv = mlp_cv.predict_proba(testdata)
cols= []
cols = cols + list('Class_{}'.format(str(i+1)) for i in range(9))
mlp_df = pd.DataFrame(pred_mlpcv, columns=cols)
mlp_df['id'] = testdf['id']
cols = mlp_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
solution = mlp_df[cols]
solution.to_csv('./mlpcv.csv', index = False)
xgb_clf = xgb.XGBClassifier(max_features='sqrt', min_samples_leaf=15, min_samples_split=10, learning_rate = 0.05, loss = 'huber')
xgb_cv = GridSearchCV(xgb_clf,{'max_depth': [2,4,6], 'n_estimators': [500]}, verbose=1)
xgb_cv.fit(X,y)
pred_xgbcv = xgb_cv.predict_proba(testdata)
cols= []
cols = cols + list('Class_{}'.format(str(i+1)) for i in range(9))
xgb_df = pd.DataFrame(pred_xgbcv, columns=cols)
xgb_df['id'] = testdf['id']
cols = xgb_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
solution = xgb_df[cols]
solution.to_csv('./mlpcv.csv', index = False)