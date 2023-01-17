import pandas as pd

import numpy as np



#mdf = 'c:/Users/John/Documents/Kaggle/Porto Seguro/'

mdf = '../input/'

train = pd.read_csv(mdf + "train.csv", usecols = ['target', 'ps_car_13','ps_ind_16_bin'])

train['id'] = np.NaN



test = pd.read_csv(mdf + "test.csv", usecols = ['id', 'ps_car_13','ps_ind_16_bin'], dtype={'id': np.int32})

test['target'] = np.NaN



train = train.replace(-1, np.NaN)

bins = [0.25, 0.573, 0.674, 0.754, 0.838, 0.945, 1.115, 4.031]

train['ps_car_13_d'] = pd.cut(train['ps_car_13'], bins)

model = pd.DataFrame()

model = train.groupby(['ps_ind_16_bin', 'ps_car_13_d'])['target'].agg([('exp_target','mean')])

model.reset_index(inplace = True)

model.head(5)
test['ps_car_13_d'] = pd.cut(test.ps_car_13, bins)

df1 = pd.DataFrame()

df1 = pd.merge(test, model, on = ['ps_ind_16_bin','ps_car_13_d'], how = 'left')

df1 = df1.drop('target', 1)

df1 = df1.rename(columns={'exp_target': 'target'})



df1.to_csv('bn_output.csv', index = False, columns = ['id', 'target'])

df1.shape

# thanks to cpmpml for : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation

from numba import jit



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



df2 = pd.DataFrame()

df2 = pd.merge(train, model, on = ['ps_ind_16_bin','ps_car_13_d'], how = 'left')



eval_gini(df2['target'], df2['exp_target'])