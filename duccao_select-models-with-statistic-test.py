# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from zipfile import ZipFile



file_path = '/kaggle/input/sberbank-russian-housing-market/train.csv.zip'

zip_file = ZipFile(file_path)

train = pd.read_csv(zip_file.open('train.csv'))
train.shape
features = ['sport_count_5000', 'full_sq', 'trc_count_5000', 'zd_vokzaly_avto_km', 'sadovoe_km', 'kremlin_km']

X = train[features].values
y = np.log(train['price_doc'])
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RepeatedKFold
def cross_validate(X, y, model):

    rf = RepeatedKFold(n_splits=5, n_repeats=6, random_state=0)

    

    scores = list()

    for train_index, test_index in rf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

            

        model.fit(X_train, y_train)

        score = mean_squared_error(model.predict(X_test), y_test)

        scores.append(score)

        

    return scores
%%time

scores_a = cross_validate(X, y, RandomForestRegressor(random_state=0, n_estimators=10))
%%time

scores_b = cross_validate(X, y, RandomForestRegressor(random_state=0, n_estimators=11))
from numpy.random import seed

from numpy.random import randint

from scipy.stats import ks_2samp, normaltest, ttest_ind
def is_normally_distributed(values):

    _, pvalue = normaltest(values)

    return pvalue > 0.05
def is_different(a, b):

    normal_a, normal_b = is_normally_distributed(a), is_normally_distributed(b)

    if normal_a and normal_b:

        test = ttest_ind

    elif not normal_a and not normal_b:

        test = ks_2samp

    else:

        raise Exception('Not implemented yet')

        

    _, pvalue = test(a, b)

    return pvalue < 0.05
from numpy.random import randint

seed(1)

n = 30

a = randint(50, 60, n)

b = randint(55, 65, n)

is_different(a, b)
from numpy.random import normal

seed(1)

n = 30

a = normal(50, 1, n)

b = normal(51, 10, n)

is_different(a, b)
np.mean(scores_a), np.mean(scores_b)
is_different(scores_a, scores_b)
%%time

scores_c = cross_validate(X, y, RandomForestRegressor(random_state=0, n_estimators=20))
np.mean(scores_c)
np.mean(scores_a), np.mean(scores_c)
is_different(scores_a, scores_c)