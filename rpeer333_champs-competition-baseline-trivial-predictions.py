import os

from os.path import join



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import metrics

from sklearn.linear_model import LinearRegression
DATA_DIR = '../input/champs-scalar-coupling'

train_all = pd.read_csv(join(DATA_DIR, 'train.csv'))

test = pd.read_csv(join(DATA_DIR, 'test.csv'))

structures = pd.read_csv(join(DATA_DIR, 'structures.csv'))
def map_atom_info(df, atom_idx, structures_df=structures):

    """

    From this kernel:

    https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark

    """

    df = pd.merge(df, structures_df, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df





train_all = map_atom_info(train_all, 0)

train_all = map_atom_info(train_all, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)



train_all.head()
# code in this cell is from:

# https://www.kaggle.com/rpeer333/brute-force-feature-engineering/edit



train_p_0 = train_all[['x_0', 'y_0', 'z_0']].values

train_p_1 = train_all[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train_all['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
def add_sc_type_features(df, one_hot=False):

    assert 'type' in df.columns

    

    df['bonds']    = df.type.map(lambda x: int(x[0]))

    df['atom_pair'] = df.type.map(lambda x: x[2:4])

    

    if one_hot:

        df = pd.get_dummies(df, prefix='bonds')

        df = pd.get_dummies(df, prefix='atom_pair')

    

    return df





train_all = add_sc_type_features(train_all)

test  = add_sc_type_features(test)

train_all.head()
is_train = np.random.rand(len(train_all)) < 0.8



val   = train_all[~is_train].copy()

train = train_all[is_train].copy()

print(f'training-fraction: {len(train) / (len(train) + len(val)):.4f}')
def metric(df, preds):

    """

    function from: https://www.kaggle.com/abhishek/competition-metric

    """

    

    df["prediction"] = preds

    maes = []

    for t in df.type.unique():

        y_true = df[df.type==t].scalar_coupling_constant.values

        y_pred = df[df.type==t].prediction.values

        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))

        maes.append(mae)

        

    return np.mean(maes)
median_prediction = np.tile(train.scalar_coupling_constant.median(), len(val))

metric(val, median_prediction)
mean_prediction = np.tile(train.scalar_coupling_constant.mean(), len(val))

metric(val, mean_prediction)
sns.boxplot(data=train, x='type', y='scalar_coupling_constant', fliersize=0.5);
group_medians = train.groupby('type').apply(lambda df: df.scalar_coupling_constant.median())

group_medians
group_median_prediction = val.type.map(group_medians)

type_median_score = metric(val, group_median_prediction)

type_median_score
lin_reg = LinearRegression()

lin_reg.fit(X=np.reshape(train.dist.values, (-1, 1)),

            y=train.scalar_coupling_constant.values)

y_hat = lin_reg.predict(np.reshape(val.dist.values, (-1, 1)))



metric(val, y_hat)
X_train = pd.get_dummies(train[['dist', 'type']], prefix='type', drop_first=True).values

X_val   = pd.get_dummies(val[['dist', 'type']], prefix='type', drop_first=True).values



lin_reg = LinearRegression()

lin_reg.fit(X=X_train, y=train.scalar_coupling_constant.values)

y_hat = lin_reg.predict(X_val)



score = metric(val, y_hat)

print(f'score: {score}')

print(f'improvement over type-mean-prediction: {type_median_score - score}')