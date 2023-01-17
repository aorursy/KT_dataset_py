# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from lightgbm import LGBMRegressor

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score, GridSearchCV
SEED = 31

FOLDS = 3

N_ESTIMATORS = 16000

TARGET = 'scalar_coupling_constant'

PREDICTORS = [

    'molecule_atom_index_0_dist_mean_div',

    'molecule_atom_index_0_dist_max_div',

    'molecule_atom_index_1_dist_max_div',

    'molecule_atom_index_0_dist_std_div',

    'molecule_atom_index_0_dist_min_div',

    'molecule_atom_index_1_dist_mean_div',

    'molecule_atom_index_1_dist_std_div',

    'molecule_atom_1_dist_std_diff',

    'molecule_atom_index_0_dist_std_diff',

    'molecule_atom_index_0_dist_mean_diff',

    'molecule_atom_index_1_dist_max_diff',

    'molecule_atom_index_0_dist_max_diff',

    'molecule_type_0_dist_std_diff',

    'molecule_atom_index_1_dist_mean_diff',

    'molecule_atom_index_1_dist_std_diff',

    'molecule_atom_1_dist_min_div',

    'molecule_atom_1_dist_min_diff',

    'type_0',

    'type_1',

    'molecule_type_dist_min',

    'molecule_type_dist_mean',

    'molecule_type_0_dist_std',

    'dist_to_type_1_mean',

    'dist',

    'molecule_type_dist_max',

    'dist_x',

    'dist_y',

    'dist_z'

]

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    

seed_everything(SEED)
file_folder = '../input/champs-scalar-coupling-preprocess'

train = pd.read_csv(f'{file_folder}/train.csv')

test = pd.read_csv(f'{file_folder}/test.csv')

print('train={}, test={}'.format(repr(train.shape), repr(test.shape)))
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    maes = np.log(maes.map(lambda x: max(x, floor)))

    print(maes)

    return maes.mean()
%%time



y_train = train[TARGET]

x_train = train[PREDICTORS]

model = LGBMRegressor(colsample_bytree=0.75, learning_rate=0.1, n_estimators=N_ESTIMATORS, reg_alpha=1)

pipe = Pipeline([('model', model)])

param_grid = {

    'model__num_leaves': [80],

    'model__min_child_samples': [200]

}

cv = GridSearchCV(pipe, cv=FOLDS, param_grid=param_grid, scoring='neg_mean_absolute_error')

cv.fit(x_train, y_train)

print('best_params_={}\nbest_score_={}'.format(repr(cv.best_params_), repr(cv.best_score_)))
y_pred_train = cv.predict(x_train)

gmlm = group_mean_log_mae(y_train, y_pred_train, train['type'])

print('group_mean_log_mae={}'.format(gmlm))
x_test = test[PREDICTORS]

# Use the model to make predictions

preds = cv.predict(x_test)

print(preds)
submission = pd.DataFrame({'id': test['id'], 'scalar_coupling_constant': preds})

submission.head()
submission.to_csv('submission.csv', index=False)

train = pd.DataFrame({'id': train['id'], 'type': train['type'], TARGET: train[TARGET], 'pred': y_pred_train})

train.to_csv('train.csv', index=False)

print(os.listdir("."))