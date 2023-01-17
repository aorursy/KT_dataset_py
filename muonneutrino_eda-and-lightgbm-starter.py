import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv', index_col=0)
train.head()
train.info()
train['log_target'] = np.log1p(train['target'])
plt.hist(train.target,range=(0,4e7),bins=100)
plt.xlabel('Target')
plt.ylabel('Number of Samples')
plt.show()
plt.hist(train.target,range=(0,1e7),bins=100)
plt.xlabel('Target')
plt.ylabel('Number of Samples')
plt.show()
plt.hist(train.target,range=(0,2e6),bins=100)
plt.xlabel('Target')
plt.ylabel('Number of Samples')
plt.show()
std = train.std().sort_values()
bad_fields = std[std==0].index
train = train.drop(bad_fields, axis=1)
train.head()
changed_type = []
for col, dtype in train.dtypes.iteritems():
    if dtype==np.int64:
        max_val = np.max(train[col])
        bits = np.log(max_val)/np.log(2)
        if bits < 8:
            new_dtype = np.uint8
        elif bits < 16:
            new_dtype = np.uint16
        elif bits < 32:
            new_dtype = np.uint32
        else:
            new_dtype = None
        if new_dtype:
            changed_type.append(col)
            train[col] = train[col].astype(new_dtype)
print('Changed types on {} columns'.format(len(changed_type)))
print(train.info())
sparsity = {
    col: (train[col] == 0).mean()
    for idx, col in enumerate(train)
}
sparsity = pd.Series(sparsity)
    
fig = plt.figure(figsize=[7,12])
ax = fig.add_subplot(211)
ax.hist(sparsity, range=(0,1), bins=100)
ax.set_xlabel('Sparsity of Features')
ax.set_ylabel('Number of Features')
ax = fig.add_subplot(212)
ax.hist(sparsity, range=(0.8,1), bins=100)
ax.set_xlabel('Sparsity of Features')
ax.set_ylabel('Number of Features')
plt.show()
min_non0 = 10
too_sparse = sparsity[(((1-sparsity) * train.shape[0]) < min_non0)].index
train = train.drop(too_sparse, axis=1)
train.info()
train.head()
import lightgbm as lgb
features = train.drop(['target','log_target'], axis=1).values
targets = train['log_target'].values.reshape([-1])
feature_names = list(train.drop(['target','log_target'], axis=1).columns.values)
train_dataset = lgb.Dataset(
        features,
        targets,
        feature_name=feature_names 
)

    
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'metric': {'rmse'},
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8    
}

cv_output = lgb.cv(
    params,
    train_dataset,
    num_boost_round=500,
    nfold=10,
    stratified=False
)
n_iterations = np.argmin(cv_output['rmse-mean'])
print('Optimal # of iterations: {}'.format(n_iterations))
print('Score: {:0.5}, Std. Dev.: {:0.5}'.format(
    cv_output['rmse-mean'][n_iterations],
    cv_output['rmse-stdv'][n_iterations]
))
model = lgb.train(
    params,
    train_dataset,
    num_boost_round=n_iterations
)
test = pd.read_csv('../input/test.csv', index_col=0)
test = test.drop(bad_fields, axis=1)
test = test.drop(too_sparse, axis=1)
test.info()

preds = model.predict(test.values)
preds = np.exp(preds) - 1

test['target'] = preds
test[['target']].to_csv('lightgbm_basic.csv')
