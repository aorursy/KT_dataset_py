# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from deeptables.models.deeptable import ModelConfig, DeepTable
from tensorflow.keras.utils import plot_model
from scipy.stats import zscore

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
X = pd.read_csv(f'../input/cat-in-the-dat-ii/train.csv')
y = X['target']
test = pd.read_csv(f'../input/cat-in-the-dat-ii/test.csv')

catID = test['id']

X = X.drop('id', axis=1)
X = X.drop('target', axis=1)
test = test.drop('id', axis=1)
ord_order = [
    [1.0, 2.0, 3.0],
    ['Novice', 'Contributer', 'Expert', 'Master', 'Grandmaster'],
    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']
]

for i in range(1, 3):
    ord_order_dict = {i : j for j, i in enumerate(ord_order[i])}
    X[f'ord_{i}_en'] = X[f'ord_{i}'].fillna('NULL').map(ord_order_dict)
    test[f'ord_{i}_en'] = test[f'ord_{i}'].fillna('NULL').map(ord_order_dict)

for i in range(3, 6):
    ord_order_dict = {i : j for j, i in enumerate(sorted(list(set(list(X[f'ord_{i}'].dropna().unique()) + list(test[f'ord_{i}'].dropna().unique())))))}
    X[f'ord_{i}_en'] = X[f'ord_{i}'].fillna('NULL').map(ord_order_dict)
    test[f'ord_{i}_en'] =  test[f'ord_{i}'].fillna('NULL').map(ord_order_dict)
cat_cols = [i for i in X.columns if '_en' not in i]
X[cat_cols] = X[cat_cols].astype('category')
test[cat_cols] = test[cat_cols].astype('category')
X.head()
config = ModelConfig(
    dnn_params={
        'hidden_units':((200, 0.3, True),(200, 0.3, True),), #hidden_units
        'dnn_activation':'relu',
    },
    embeddings_output_dim=20,
    nets =['linear','cin_nets','dnn_nets'],
    output_use_bias = False,
    cin_params={
       'cross_layer_size': (200, 200),
       'use_bias': True,
       'direct': True
    },
)
model = DeepTable(config=config)
oof_proba, eval_proba, test_prob = model.fit_cross_validation(
    X, y, X_eval=None, X_test=test, num_folds=3, stratified=False, iterators=None, batch_size=128, epochs=1, verbose=1, callbacks=[], n_jobs=1
)
plot_model(model.get_model().model,rankdir='TB')
submission = pd.read_csv(f'../input/cat-in-the-dat-ii/sample_submission.csv')
submission['target'] = test_prob
submission.to_csv('submission.csv', index=False)