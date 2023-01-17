import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%pylab inline
# Import DATA

x_train = pd.read_csv('../input/train.csv')
x_test = pd.read_csv('../input/test.csv')

print('x_train: {0}, x_test: {1}'.format(x_train.shape, x_test.shape))
# Create target array y

test_ID = x_test['ID']
y_train = x_train['target']
y_train = np.log1p(y_train)
x_train.drop("ID", axis = 1, inplace = True)
x_train.drop("target", axis = 1, inplace = True)
x_test.drop("ID", axis = 1, inplace = True)
# Drop features with only one value

cols_with_onlyone_val = x_train.columns[x_train.nunique() == 1]
x_train.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
x_test.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
print('remove n columns: {0}'.format(len(cols_with_onlyone_val)))
print('x_train: {0}, x_test: {1}'.format(x_train.shape, x_test.shape))
# Check for duplicated columns

colsToRemove = []
columns = x_train.columns
for i in range(len(columns)-1):
   v = x_train[columns[i]].values
   dupCols = []
   for j in range(i + 1,len(columns)):
      if np.array_equal(v, x_train[columns[j]].values):
          colsToRemove.append(columns[j])
x_train.drop(colsToRemove, axis=1, inplace=True)
x_test.drop(colsToRemove, axis=1, inplace=True)
print('removed columns: {0}'.format(colsToRemove))
print('x_train: {0}, x_test: {1}'.format(x_train.shape, x_test.shape))
# Select top NUM_OF_FEATURES informative features

from sklearn import model_selection as ms
from sklearn.ensemble import RandomForestRegressor

def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(y - pred, 2)))

x1, x2, y1, y2 = ms.train_test_split(
    x_train, y_train.values, test_size=0.10)
model = RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state=0)
model.fit(x1, y1)
print('rmsle: {0}'.format(rmsle(y2, model.predict(x2))))
# Delete features (from low to tom importance) until sum importance will be less 0.95

imp_cols = pd.DataFrame({'importance': model.feature_importances_, 
                    'feature': x_train.columns}).sort_values(by=['importance'], 
                    ascending=[True]).reset_index()

for i in range(imp_cols.shape[0]):
    imp_cols.drop(i, inplace=True)
    if imp_cols.importance.sum() <= 0.95:
        break

x_train = x_train[imp_cols.feature.values]
x_test = x_test[imp_cols.feature.values]
print('x_train: {0}, x_test: {1}'.format(x_train.shape, x_test.shape))
# Plot distribution of feature importances

fig, ax = subplots(figsize=(12,6))
ax.set_yscale('log')
plt.xlabel('feature number')
plt.ylabel('log(importance)')
plt.grid()
plt.plot(imp_cols.index, imp_cols.importance*10000);
# Create new features

ntrain = len(x_train)
ntest = len(x_test)
tmp = pd.concat([x_train, x_test])
weight = ((x_train != 0).sum()/len(x_train)).values # Sum of non zero elements of every feature
tmp_train = x_train[x_train!=0]
tmp_test = x_test[x_test!=0]

x_train["weight_count"] = (tmp_train*weight).sum(axis=1)
x_test["weight_count"] = (tmp_test*weight).sum(axis=1)
x_train["count_not0"] = (x_train != 0).sum(axis=1)
x_test["count_not0"] = (x_test != 0).sum(axis=1)
x_train["sum"] = x_train.sum(axis=1)
x_test["sum"] = x_test.sum(axis=1)
x_train["var"] = tmp_train.var(axis=1)
x_test["var"] = tmp_test.var(axis=1)
x_train["median"] = tmp_train.median(axis=1)
x_test["median"] = tmp_test.median(axis=1)
x_train["mean"] = tmp_train.mean(axis=1)
x_test["mean"] = tmp_test.mean(axis=1)
x_train["std"] = tmp_train.std(axis=1)
x_test["std"] = tmp_test.std(axis=1)
x_train["max"] = tmp_train.max(axis=1)
x_test["max"] = tmp_test.max(axis=1)
x_train["min"] = tmp_train.min(axis=1)
x_test["min"] = tmp_test.min(axis=1)
x_train["skew"] = tmp_train.skew(axis=1)
x_test["skew"] = tmp_test.skew(axis=1)
x_train["kurtosis"] = tmp_train.kurtosis(axis=1)
x_test["kurtosis"] = tmp_test.kurtosis(axis=1)

del(tmp_train)
del(tmp_test)
print('x_train: {0}, x_test: {1}'.format(x_train.shape, x_test.shape))
# Train XGBoost model and check results of cross-validation

import xgboost as xg
from sklearn import metrics

clf = xg.XGBRegressor(n_estimators=242, max_depth=5, learning_rate=0.02, min_child_weight=40)#211
cvs_res = -ms.cross_val_score(clf, x_train, y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
print('rmsle = ', round(cvs_res.mean()**0.5,3), '+/-', round(cvs_res.std(),3))
# Function to find best model

from scipy import optimize as opt
def f(x):
    n = int(x[0])
    md = int(x[1])
    lr = 0.1
    mchw = 20
    clf = xg.XGBRegressor(n_estimators=n, max_depth=md, learning_rate=lr, min_child_weight=mchw, reg_alpha=0.1)
    res = -ms.cross_val_score(clf, x_train, ravel(y_train), scoring='neg_mean_squared_error', cv=3, n_jobs=-1).mean()
    print('-> {0} | {1} = {2}'.format(n, md, round(res**0.5,3)))
    return res

#opt_res = opt.differential_evolution(f,[(20,300), (3,6)], maxiter=200, disp=True)
#print opt_res