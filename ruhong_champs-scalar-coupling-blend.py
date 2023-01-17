# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import random

import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
SEED = 31

TRIALS = 200

TARGET = 'scalar_coupling_constant'

PREDICTION = 'pred'
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    

seed_everything(SEED)
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    maes = np.log(maes.map(lambda x: max(x, floor)))

    return maes.mean()
lasso = pd.read_csv(f'../input/champs-scalar-coupling-lasso/submission.csv')

rf = pd.read_csv(f'../input/champs-scalar-coupling-rf/submission.csv')

xgb = pd.read_csv(f'../input/champs-scalar-coupling-xgb/submission.csv')

lgb = pd.read_csv(f'../input/champs-scalar-coupling-lgb/submission.csv')

keras = pd.read_csv(f'../input/champs-scalar-coupling-keras/submission.csv')

test_sets = [lgb, keras, xgb, rf, lasso]

print(f'lasso={lasso.shape}, rf={rf.shape}, xgb={xgb.shape}, lgb={lgb.shape}, keras={keras.shape}')
lasso_train = pd.read_csv(f'../input/champs-scalar-coupling-lasso/train.csv')

rf_train = pd.read_csv(f'../input/champs-scalar-coupling-rf/train.csv')

xgb_train = pd.read_csv(f'../input/champs-scalar-coupling-xgb/train.csv')

lgb_train = pd.read_csv(f'../input/champs-scalar-coupling-lgb/train.csv')

keras_train = pd.read_csv(f'../input/champs-scalar-coupling-keras/train.csv')

train_sets = [lgb_train, keras_train, xgb_train, rf_train, lasso_train]

print(f'Train sets\nlasso={lasso_train.shape}, rf={rf_train.shape}, xgb={xgb_train.shape}, lgb={lgb_train.shape}, keras={keras_train.shape}')
%%time



def weights(n, min_weight=0.01, max_allocation=0.5):

    if n < 1:

        raise ValueError('n must not be less than 1')

    remainder = 1 - (n * min_weight)

    if remainder <= 0:

        raise ValueError('min weight exceeds budget of 1')

    res = []

    for _ in range(n - 1):

        a = random.uniform(0.01, max_allocation) * remainder

        res.append(a + min_weight)

        remainder -= a

    res.append(remainder + min_weight)

    return res





def trial(train_sets, prediction_column, target_column):

    ws = weights(len(train_sets), min_weight=0.05, max_allocation=0.9)

    df = train_sets[0].copy()

    df[prediction_column] = 0

    for i, t in enumerate(train_sets):

        df[prediction_column] += t[prediction_column] * ws[i]

    score = group_mean_log_mae(df[target_column], df[prediction_column], df['type'])

    return score, ws





best = sys.maxsize

best_weights = []

for _ in range(TRIALS):

    score, ws = trial(train_sets=train_sets, prediction_column=PREDICTION, target_column=TARGET)

    if score < best:

        best = score

        best_weights = ws

        

print(f'best={best:.4f}')

print(f'''best weights (sum={sum(best_weights)})

  lgb={best_weights[0]:.4f}

  keras={best_weights[1]:.4f}

  xgb={best_weights[2]:.4f}

  rf={best_weights[3]:.4f}

  lasso={best_weights[4]:.4f}

''')
submission = test_sets[0].copy()

submission[TARGET] = 0

for i, t in enumerate(test_sets):

    submission[TARGET] += t[TARGET] * best_weights[i]

submission.head()
submission.to_csv('submission.csv', index=False)

print(os.listdir("."))