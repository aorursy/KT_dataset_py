import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# Set to false for tuning

random_result = True



if random_result:

    xgb_seed = 0

    sklearn_seed = None

else:

    xgb_seed = 19

    sklearn_seed = 19
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df_train.head()
cata_cols = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
# List all possible values of X0 ~ X8



def lenkey(x):

    r = len(x)

    for c in x:

        r *= 100

        r += (ord(c) - ord('a'))

    return r



xs = set()

for col in cata_cols:

    xs |= set(df_train[col].unique())

    xs |= set(df_test[col].unique())

xs = list(xs)

xs.sort(key=lenkey)

print(xs, '\n', len(xs))
# Encode in sequence: a->0, b->1, ... z->25, aa->26, ..., bc->54



def tonum(x):

    r = 0

    for c in x:

        r *= 26

        r += ord(c)-ord('a')+1

    return r-1



for col in cata_cols:

    df_train[col] = df_train[col].apply(tonum).astype(int)

    df_test[col] = df_test[col].apply(tonum).astype(int)
# Count 1/0 of X10~X385

cols = df_train.columns.shape[0]

df_train['Count1'] = df_train.iloc[:, 10:].sum(axis=1)

df_test['Count1'] = df_test.iloc[:, 10:].sum(axis=1)

df_train['Count0'] = cols - 10 - df_train['Count1']

df_test['Count0'] = cols - 10 - df_test['Count1']

df_train['Count_ratio'] = df_train['Count1'] / df_train['Count0']

df_test['Count_ratio'] = df_test['Count1'] / df_test['Count0']
X = df_train.drop(['y'], axis=1)

y = df_train['y']

X_test = df_test.copy()
import xgboost as xgb



xgb_params = {

    'eta': 0.01,

    'max_depth': 3,

    'subsample': 1,

    'colsample_bytree': 0.5,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1,

    'seed': xgb_seed,

}
# Five fold CV training and stack y



from sklearn.model_selection import KFold

from sklearn.metrics import r2_score



kf = KFold(n_splits=5, shuffle=True, random_state=sklearn_seed)



models = []

best_iter = 0

_y_predict = np.zeros(X.shape[0])

for train_index, valid_index in kf.split(X):

    X_train, X_valid = X.iloc[train_index, :], X.iloc[valid_index, :]

    y_train, y_valid = y.values[train_index], y.values[valid_index]

    dtrain = xgb.DMatrix(X_train, y_train)

    dvalid = xgb.DMatrix(X_valid, y_valid)

    evals = [(dtrain,'train'), (dvalid,'eval')]

    print('---------------------------------------------------------')

    model = xgb.train(xgb_params, dtrain, 10000, evals, verbose_eval=100, early_stopping_rounds=50)

    best_iter += model.best_iteration

    _y_predict[valid_index] = model.predict(xgb.DMatrix(X_valid))

    models.append(model)



best_iter //= len(models)



r2 = r2_score(y, _y_predict)

print('=========================================================')

print('R2-stack = %.4f, best_iter = %d' % (r2, best_iter))
# Train on full train set

evals = [(xgb.DMatrix(X, y), 'train')]

model = xgb.train(xgb_params, xgb.DMatrix(X, y), best_iter, evals, verbose_eval=100)



r2 = r2_score(y, model.predict(xgb.DMatrix(X)))

print('R2-train = %.4f' % r2)
# Predict test set

_y_test = model.predict(xgb.DMatrix(X_test))

submit = pd.DataFrame({'ID': df_test['ID'], 'y': _y_test})

submit.head()
# Plot feature importantce

_, ax = plt.subplots(1, 1, figsize=(8, 12))

xgb.plot_importance(model, height=0.5, ax=ax, max_num_features=50)
# Observation: y above 115 are under-estimated (points located at right of red line)

# - Green line: y = 115

# - Blue line: y_predict = 115

# - Red line: y = y_predict



_, ax = plt.subplots(1, 1, figsize=(16, 6))

ax.set_aspect('equal')

ax.set_xlim(70, 275)

ax.set_ylim(70, 160)

ax.set_xticks(range(70, 270, 10))

output1 = pd.DataFrame({'y': df_train.y, 'y_predict': model.predict(xgb.DMatrix(X))})

output1.plot.scatter('y', 'y_predict', ax=ax)



xlim = range(70, 275)

ax.plot(xlim, [117]*len(xlim))

ylim = range(70, 160)

ax.plot([117]*len(ylim), ylim)



ax.plot(range(70,160), range(70,160))
threshold = 115

level = 0.5



# stack

X['y_predict'] = _y_predict

X_test['y_predict'] = _y_test



y = df_train['y'].apply(lambda x: 1 if x >= threshold else 0)
import xgboost as xgb



xgb_params = {

    'eta': 0.01,

    'max_depth': 8,

    'subsample': 1,

    'colsample_bytree': 1,

    'objective': 'binary:logistic',

    'eval_metric': 'logloss',

    'silent': 1,

    'seed': xgb_seed,

}
# Five fold CV training



from sklearn.model_selection import StratifiedKFold



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sklearn_seed)



best_iter = 0

for train_index, valid_index in kf.split(X, y):

    X_train, X_valid = X.iloc[train_index, :], X.iloc[valid_index, :]

    y_train, y_valid = y.values[train_index], y.values[valid_index]

    dtrain = xgb.DMatrix(X_train, y_train)

    dvalid = xgb.DMatrix(X_valid, y_valid)

    evals = [(dtrain,'train'), (dvalid,'eval')]

    print('---------------------------------------------------------')

    model = xgb.train(xgb_params, dtrain, 10000, evals, verbose_eval=100, early_stopping_rounds=50)

    best_iter += model.best_iteration



best_iter //= len(models)

print(best_iter)
# Train on full test set

evals = [(xgb.DMatrix(X, y), 'train')]

model = xgb.train(xgb_params, xgb.DMatrix(X, y), best_iter, evals, verbose_eval=100)

predict = model.predict(xgb.DMatrix(X))
# How dose the classifier perform

output2 = pd.DataFrame({'y': df_train.y, 'y_predict': model.predict(xgb.DMatrix(X))})

total = (output2.y >= threshold).sum()

corrcnt = (output2.y_predict >= level).sum()

errcnt = ((output2.y_predict >= level) & (output2.y < threshold)).sum()

print('total=%d, correct=%d, error=%d' % (total, corrcnt, errcnt))
# Red points are classified as above 115.

output = pd.concat([output1, output2.y_predict], axis=1)

output.columns = ['truth', 'predict', 'prob']



output_hi = output[output.prob>=level]

output_lo = output[output.prob<level]



_, ax = plt.subplots(1, 1, figsize=(16, 6))

ax.set_aspect('equal')

ax.set_xlim(70, 275)

ax.set_ylim(70, 160)

ax.set_xticks(range(70, 270, 10))

output_lo.plot.scatter('truth', 'predict', ax=ax)

output_hi.plot.scatter('truth', 'predict', ax=ax, color='red')



xlim = range(70, 275)

ax.plot(xlim, [threshold]*len(xlim))

ylim = range(70, 160)

ax.plot([threshold]*len(ylim), ylim)



ax.plot(range(70,160), range(70,160))
# Plot recall-precision line

from sklearn.metrics import precision_recall_curve, average_precision_score



label = output.truth

predict = output.prob



p, r, _ = precision_recall_curve(label>=threshold, predict)

a = average_precision_score(label>=threshold, predict)

plt.plot(r, p)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('AUC={0:0.2f}'.format(a))
_y_test_high = model.predict(xgb.DMatrix(X_test))

_y_test_high = _y_test_high >= level

_y_test_high.sum()
# Ugly adjustment: *1.03 for y>=115

adj = 1.03



submit_adj = submit.copy()

submit_adj.loc[_y_test_high, 'y'] *= adj

submit_adj.to_csv('benz-xgb.csv', index=False)