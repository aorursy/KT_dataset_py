# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold

from sklearn.metrics import roc_auc_score



import lightgbm as lgb



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')



from pathlib import Path

from sklearn.utils import shuffle

from datetime import datetime

path=Path("../input/")

train=pd.read_csv(path/"train.csv").drop("ID_code",axis=1)

test=pd.read_csv(path/"test.csv").drop("ID_code",axis=1)
class DataAugment(object):



    def __init__(self, num_n, num_p, seed=0):

        """

        :param num_n: Negative sample enhancement multiple

        :param num_p: Positive sample enhancement multiple

        :param seed:

        """

        self.num_n = num_n

        self.num_p = num_p

        self.seed = seed



    def transform(self, X: pd.DataFrame, y):

        if not hasattr(y, 'tolist'):

            y = pd.Series(y)



        cols = X.columns

        X.columns = range(len(cols))

        Xn = self._augment(X[y == 0], self.num_n, self.seed)

        Xp = self._augment(X[y == 1], self.num_p, self.seed + 666666)



        X_ = pd.concat([X] + Xn + Xp, ignore_index=True)

        X.columns = cols

        y_ = y.tolist() + [0] * (y == 0).sum() * self.num_n + [1] * (y == 1).sum() * self.num_p

        return X_, pd.Series(y_)



    def _augment(self, X, num, seed=0):

        return [X.apply(lambda x: shuffle(x, random_state=x.name + seed + i)) for i in range(num)]

%%time

da = DataAugment(1, 2, 2019)

X, y = da.transform(train.drop('target', 1), train.target)
result=np.zeros(test.shape[0])



rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=10)

for counter,(train_index, valid_index) in enumerate(rskf.split(X, y),1):

    print (counter)

    X_train, y_train = X.iloc[train_index], y.iloc[train_index]

    X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]

    

    #Train data

    trn_data = lgb.Dataset(X_train, y_train)

    

    #Validation data

    val_data = lgb.Dataset(X_valid, y_valid)

    

    #Training

    model = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 4000)

    result +=model.predict(test)
submission = pd.read_csv(path/'sample_submission.csv')

submission['target'] = result/counter

filename="{:%Y-%m-%d_%H_%M}_sub.csv".format(datetime.now())

submission.to_csv(filename, index=False)