!pip install --upgrade lightgbm
import pandas as pd

import lightgbm as lgb



lgb.__version__
train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

target = train['isFraud']

del train['isFraud']
categories = [

    'ProductCD', 'P_emaildomain', 'R_emaildomain',

    'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2',

    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',

]

for c in categories:

    train[c] = train[c].astype('category')
params = {

    'bagging_seed': 13,

    'boosting_type': 'gbdt',

    'feature_fraction': 0.3797454081646243,

    'learning_rate': 0.006883242363721497,

    'max_depth': -1,

    'metric': 'auc',

    'min_child_weight': 0.03454472573214212,

    'min_data_in_leaf': 106,

    'num_leaves': 256,

    'objective': 'binary',

    'reg_alpha': 0.3899927210061127,

    'reg_lambda': 0.6485237330340494,

    'seed': 71,

    'verbosity': -1,

}
import lightgbm as lgb

from sklearn.model_selection import train_test_split



train_index, valid_index = train_test_split(train.index, test_size=0.2, shuffle=False)

train_data = lgb.Dataset(train.loc[train_index], label=target.loc[train_index])

valid_data = lgb.Dataset(train.loc[valid_index], label=target.loc[valid_index])



model = lgb.train(params, train_data, num_boost_round=5000, valid_sets=[train_data, valid_data],

                  verbose_eval=100, early_stopping_rounds=100)
lgb.plot_importance(model, figsize=(20, 60), importance_type='gain', max_num_features=50)
from IPython.display import display, HTML

import matplotlib.pyplot as plt



for feature in ['V258', 'C13', 'C14', 'C1', 'D2', 'V257', 'C11', 'V243', 'D15', 'V294', 'C4', 'V201', 'C8', 'D1', 'V317']:

    display(HTML(f'<h2>{feature} Split Histogram</h2>'))



    lgb.plot_split_value_histogram(model, feature=feature)

    plt.show()