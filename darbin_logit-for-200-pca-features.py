import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import statsmodels.formula.api as sm
random_state = 635

np.random.seed(random_state)

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
train_X = df_train.drop(['ID_code', 'target'], axis = 1)

test_X = df_test.drop(['ID_code'], axis = 1)
# scaling

scaler = StandardScaler()

train_X_scaled = pd.DataFrame(scaler.fit_transform(train_X),columns = train_X.columns)

test_X_scaled = pd.DataFrame(scaler.fit_transform(test_X),columns = test_X.columns)
# extract all PCA factors

pca = PCA()  

factors_train = pca.fit_transform(train_X_scaled) 

factors_test = pca.transform(test_X_scaled)
# replace 200 vars with PCA features

pca_columns_name = ["pca_" + str(col) for col in range(0, 200)]

factors_train = pd.DataFrame(factors_train, columns = pca_columns_name)

factors_test = pd.DataFrame(factors_test, columns = pca_columns_name)



train_pca = df_train.merge(factors_train, left_index = True, right_index = True)

test_pca = df_test.merge(factors_test, left_index = True, right_index = True)



train_pca_only = train_pca.drop(train_X.columns, axis = 1)

test_pca_only = test_pca.drop(test_X.columns, axis = 1)
# logit formula for regression

formula = 'target ~ '

for col in train_pca_only.drop(['ID_code', 'target'], axis = 1).columns:

    if( col == 'pca_0'):

        formula += str(col)

    else:

        formula += "+" + str(col)

formula
# logit model (based on all 200 derived pca factors)

logit = sm.ols(formula = formula, data = train_pca_only)

logit_trained = logit.fit()

logit_trained.summary()