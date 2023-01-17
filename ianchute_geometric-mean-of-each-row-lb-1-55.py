import numpy as np
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
X = train[train.columns[2:]]
X_test = test[test.columns[1:]]
X_all = pd.concat([X, X_test], ignore_index=True)

# remove all zero columns
nonzero_cols = X.columns[~np.all(X == 0, axis=0)]
X = X[nonzero_cols]
X_test = X_test[nonzero_cols]
X_all = X_all[nonzero_cols]

# remove duplicate columns
nondup_cols = X.columns[np.where(~X.T.duplicated())[0]]
X = X[nondup_cols]
X_test = X_test[nondup_cols]
X_all = X_all[nondup_cols]

X = X.astype(int)
Y = train.target.values.astype(int).tolist()
def geometric_mean(x): # geometric mean of non-zero uniques
    x = np.unique(x[x>0])
    return np.exp(np.log(x).mean())
train_pred = (X.T.apply(geometric_mean).values.astype(int))
test_pred = (X_test.T.apply(geometric_mean).values.astype(int))
def rmsle(y_true, y_pred):
    log_error = np.log1p(y_true) - np.log1p(y_pred)
    return np.sqrt( np.square(log_error).mean() )

rmsle(Y, train_pred)
results = pd.DataFrame({
    'ID': test.ID,
    'target': test_pred
})
results['target'] = results['target'].fillna(0).astype('int64')
results.loc[results['target'] < 0, 'target'] = 0
results.to_csv('geometric_mean.csv', index=False)

assert (results.target<0).sum() == 0 and np.isinf(results.target).sum() == 0 and np.isnan(results.target).sum() == 0