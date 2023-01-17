MAX_ROUNDS = 650

OPTIMIZE_ROUNDS = False

LEARNING_RATE = 0.05
import numpy as np

import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from numba import jit
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
# Read data

train_df = pd.read_csv('../input/train.csv', na_values="-1") # .iloc[0:200,:]

test_df = pd.read_csv('../input/test.csv', na_values="-1")
# Process data

id_test = test_df['id'].values

id_train = train_df['id'].values



train_df = train_df.fillna(999)

test_df = test_df.fillna(999)



col_to_drop = train_df.columns[train_df.columns.str.startswith('ps_calc_')]

train_df = train_df.drop(col_to_drop, axis=1)  

test_df = test_df.drop(col_to_drop, axis=1)  



for c in train_df.select_dtypes(include=['float64']).columns:

    train_df[c]=train_df[c].astype(np.float32)

    test_df[c]=test_df[c].astype(np.float32)

for c in train_df.select_dtypes(include=['int64']).columns[2:]:

    train_df[c]=train_df[c].astype(np.int8)

    test_df[c]=test_df[c].astype(np.int8)

    

y = train_df['target']

X = train_df.drop(['target', 'id'], axis=1)

y_valid_pred = 0*y

X_test = test_df.drop(['id'], axis=1)

y_test_pred = 0
# Set up folds

K = 5

kf = KFold(n_splits = K, random_state = 1, shuffle = True)
# Set up classifier

model = CatBoostClassifier(

    learning_rate=LEARNING_RATE, 

    depth=6, 

    l2_leaf_reg = 14, 

    iterations = MAX_ROUNDS,

#    verbose = True,

    loss_function='Logloss'

)
# Run CV



for i, (train_index, test_index) in enumerate(kf.split(train_df)):

    

    # Create data for this fold

    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

    X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]

    print( "\nFold ", i)

    

    # Run model for this fold

    if OPTIMIZE_ROUNDS:

        fit_model = model.fit( X_train, y_train, 

                               eval_set=[X_valid, y_valid],

                               use_best_model=True

                             )

        print( "  N trees = ", model.tree_count_ )

    else:

        fit_model = model.fit( X_train, y_train )

        

    # Generate validation predictions for this fold

    pred = fit_model.predict_proba(X_valid)[:,1]

    print( "  Gini = ", eval_gini(y_valid, pred) )

    y_valid_pred.iloc[test_index] = pred

    

    # Accumulate test set predictions

    y_test_pred += fit_model.predict_proba(X_test)[:,1]

    

y_test_pred /= K  # Average test set predictions



print( "\nGini for full training set:" )

eval_gini(y, y_valid_pred)
# Save validation predictions for stacking/ensembling

val = pd.DataFrame()

val['id'] = id_train

val['target'] = y_valid_pred.values

val.to_csv('cat_valid.csv', float_format='%.6f', index=False)
# Create submission file

sub = pd.DataFrame()

sub['id'] = id_test

sub['target'] = y_test_pred

sub.to_csv('cat_submit.csv', float_format='%.6f', index=False)