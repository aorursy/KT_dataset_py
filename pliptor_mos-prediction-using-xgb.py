import pandas as pd

import numpy as np

#import mpl_scatter_density

from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
df = pd.read_excel('../input/dataset.xlsx')
# quantize MOS

df.loc[df['MOS']<=1.0,'MOS'] = 1.0

for th in [1.0, 2.0, 3.0, 4.0, 5.0]:

     df.loc[(df['MOS']>th - 1)&(df['MOS']<=th),'MOS'] = th

# renaming columns to shorter ones

df.columns = ['Date','Signal','Speed','Distance','Duration','Result','Type','Time','MOS']

df.dtypes

df[['Result','Type','MOS']] = df[['Result','Type','MOS']].apply(LabelEncoder().fit_transform)

df.drop(['Date','Time'], axis=1, inplace = True)
plt.hist(df['MOS'],100)

plt.show()
df['Signal'].fillna(-1, inplace = True)

plt.hist2d(df.Signal, df.MOS, (50, 50), cmap=plt.cm.jet)

plt.colorbar()

plt.show()



df.shape
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb



K = 5

kf = StratifiedKFold(n_splits = K, random_state = 2018, shuffle = True)

train = df

target_train = train['MOS']

train.drop('MOS', axis = 1,inplace = True)

kf.get_n_splits(train, target_train)
for train_index, test_index in kf.split(train, target_train):

    train_X, valid_X = train.iloc[train_index], train.iloc[test_index]

    train_y, valid_y = target_train.iloc[train_index], target_train.iloc[test_index]

    # use these parameters for regression mode

    xgb_params = {'eta': 0.002,

                  'max_depth': 8,

                  'subsample': 0.5,

                  'colsample_bytree': 1,

                  'objective': 'reg:linear',

                  'eval_metric': 'rmse',

                  'seed': 2018,

                  'silent': False}

    # use these parameters for classification mode

    xgb_params_c = {'eta': 0.002,

                'num_class': 5,

                  'max_depth': 8,

                  'subsample': 0.5,

                  'colsample_bytree': 1,

                  'objective': 'multi:softmax',

                  'eval_metric': 'merror',

                  'seed': 2018,

                  'silent': False}



    d_train = xgb.DMatrix(train_X, train_y)

    d_valid = xgb.DMatrix(valid_X, valid_y)

    

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # chose between xgb_params and xgb_params_c

    model = xgb.train(xgb_params_c, d_train, 100,  watchlist, verbose_eval=5, early_stopping_rounds=10)
import matplotlib.pyplot as plt



xgb.plot_importance(model)

plt.show()