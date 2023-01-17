import numpy as np 

import pandas as pd

import os

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from numba import jit

from sklearn.linear_model import LogisticRegression



os.listdir('../input')
indir = '../input/kaggleportosegurocnoof/'

infiles = [

  'forza_pascal_oof.csv',

  'gp_pseu_val.csv',

  'rgf_valid.csv',

  'stacker_oof_preds_1.csv',

  'xgb_valid.csv'

]

for i, f in enumerate(infiles):

    indf = pd.read_csv(indir + infiles[i])

    if not i:

        oof = indf

    else:

        oof = pd.merge(oof, indf, on='id', suffixes=['',str(i)])

oof.rename(columns={'target':'target0'}, inplace=True)

oof.head()
train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')
df = pd.merge(train[['id','target']], oof, on='id')

df.head(10)
X = df.drop(['id', 'target'], axis=1)

X.head()
y = df['target']

y.head(10)
ids = df['id']

ids.head()
# Compute gini



# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation

@jit

def eval_gini(y_true, y_prob):

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini
# Set up folds

K = 5

kf = KFold(n_splits = K, random_state = 1, shuffle = True)

np.random.seed(0)

y_valid_pred = 0*df['target']
stacker = LogisticRegression()
for i, (train_index, test_index) in enumerate(kf.split(df)):

    

    # Create data for this fold

    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index].copy()

    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()

    print( "\nFold ", i)

    

    stacker.fit(X_train, y_train)

    pred = stacker.predict_proba(X_valid)[:,1]

    print( "  Gini = ", eval_gini(y_valid, pred) )

    

    y_valid_pred.iloc[test_index] = pred
print( "\nGini for full training set:" )

eval_gini(df['target'], y_valid_pred)
val = pd.DataFrame()

val['id'] = df['id'].values

val['target'] = y_valid_pred.values

val.to_csv('next_level_oof_preds.csv', float_format='%.6f', index=False)
val.head()