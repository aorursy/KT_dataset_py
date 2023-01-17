import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
from sklearn.model_selection import KFold, cross_val_score, train_test_split



#Validation function

def rmsle_cv(X, y, model, n_folds = 10):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
import statsmodels.api as sm

train = pd.read_csv('../input/preprocessed_training.csv')

print('Feature types: {}'.format(train.dtypes.unique())) # all features are numerical



price = train['SalePrice']

train.drop('SalePrice', axis=1, inplace=True);
kf = KFold(n_splits=10, random_state=42, shuffle=True)

print(kf)  



results = []



for train_index, test_index in kf.split(train):

    X_train, X_test = train.iloc[train_index], train.iloc[test_index]

    y_train, y_test = price.iloc[train_index], price.iloc[test_index]

    glm_gaussian = sm.GLM(y_train, X_train)

    mod = glm_gaussian.fit()

    pred_test = mod.predict(X_test)

    results.append(np.sqrt(np.mean((pred_test-y_test)**2)))

    

print("GLM score: {:.4f} ({:.4f})\n".format(np.mean(results), np.std(results)))
log_price = np.log1p(price)



kf = KFold(n_splits=10, random_state=42, shuffle=True)

print(kf)  



results = []



for train_index, test_index in kf.split(train):

    X_train, X_test = train.iloc[train_index], train.iloc[test_index]

    y_train, y_test = log_price.iloc[train_index], log_price.iloc[test_index]

    glm_gaussian = sm.GLM(y_train, X_train)

    mod = glm_gaussian.fit()

    pred_test = mod.predict(X_test)

    results.append(np.sqrt(np.mean((pred_test-y_test)**2)))

    

print("GLM score: {:.4f} ({:.4f})\n".format(np.mean(results), np.std(results)))

(np.exp(0.1168)-1)*100
from sklearn.metrics import mean_absolute_error



kf = KFold(n_splits=10, random_state=42, shuffle=True)

print(kf)  



results = []



for train_index, test_index in kf.split(train):

    X_train, X_test = train.iloc[train_index], train.iloc[test_index]

    y_train, y_test = log_price.iloc[train_index], log_price.iloc[test_index]

    glm_gaussian = sm.GLM(y_train, X_train)

    mod = glm_gaussian.fit()

    pred_test = mod.predict(X_test)

    results.append(mean_absolute_error(pred_test, y_test))

    

print("GLM score: {:.4f} ({:.4f})".format(np.mean(results), np.std(results)))

print("Mean percentage error: {:.4f}\n".format((np.exp(np.mean(results))-1)*100))

glm_gaussian = sm.GLM(log_price, train)

mod = glm_gaussian.fit()

pred_test = mod.predict(train)



fig, ax = plt.subplots()

ax.scatter(x=log_price, y=log_price - pred_test)

plt.ylabel('Residuals', fontsize=13)

plt.xlabel('log(y)', fontsize=13)

ax.axhline(y=0)

plt.title('Residual plot')

plt.show()



sns.distplot(log_price - pred_test)

plt.title('Residual distribution')
from sklearn.ensemble import GradientBoostingRegressor



GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
score = rmsle_cv(train, log_price, GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_df = scaler.fit_transform(train)

scaled_df = pd.DataFrame(scaled_df, columns=train.columns)



glm_gaussian = sm.GLM(price, scaled_df)

mod = glm_gaussian.fit()

mod.summary()