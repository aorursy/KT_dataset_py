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
train.head()
correlations = train.dropna(axis=1).corr()
correlations.iloc[-1].sort_values()[-10:]
correlations.iloc[-1].sort_values()[:3]
features = ['sport_count_5000', 'full_sq', 'trc_count_5000', 'zd_vokzaly_avto_km', 'sadovoe_km', 'kremlin_km']

X = train[features].values
y = np.log(train['price_doc'])
import seaborn as sns



sns.distplot(y)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import RepeatedKFold

from scipy.stats.mstats import winsorize
def cross_validate(X, y, model, outlier_treatment=None):

    rf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)

    

    scores = list()

    for train_index, test_index in rf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        

        if outlier_treatment is not None:

            X_train, y_train = outlier_treatment(X_train, y_train)

            

        model.fit(X_train, y_train)

        score = mean_squared_error(model.predict(X_test), y_test)

        scores.append(score)

        

    return np.mean(scores)
cross_validate(X, y, LinearRegression())
def iqr(X_train, y_train):

    q1 = np.percentile(y_train, 25)

    q3 = np.percentile(y_train,75)

    iqr = q3 - q1

    upper = q3 + 1.5 * iqr

    lower = q1 - 1.5 * iqr

    idx = (y_train < upper) & (y_train > lower)

    return X_train[idx], y_train[idx]



cross_validate(X, y, LinearRegression(), iqr)
def z_score(X_train, y_train):

    z = (y_train - np.mean(y_train)) / np.std(y_train)

    idx = np.absolute(z) < 2

    return X_train[idx], y_train[idx]



cross_validate(X, y, LinearRegression(), z_score)
def winsorizing(X_train, y_train):

    return X_train, winsorize(y_train, limits=[0.05, 0.05])



cross_validate(X, y, LinearRegression(), winsorizing)
cross_validate(X, y, RandomForestRegressor(random_state=0, n_estimators=10))
for outlier_treatment in [iqr, z_score, winsorizing]:

    print(outlier_treatment.__name__)

    score = cross_validate(X, y, RandomForestRegressor(random_state=0, n_estimators=10),

                  outlier_treatment=outlier_treatment)

    print(score)