import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import pymc3 as pm

import theano.tensor as tt



from scipy.stats import skew



from sklearn.linear_model import Lasso, Ridge

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold



from priors_penalties_functions import BayesianGLM

from priors_penalties_functions import plot_errors_and_coef_magnitudes, cross_validate_hyperparam_choices



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))



#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])



# Create dummy variables

all_data = pd.get_dummies(all_data)



# Mean imputation

all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
selector = SelectKBest(f_regression, k=5)

selector.fit(X_train, y)



scaler = StandardScaler()

scaler.fit(X_train, y)



columns = X_train.columns[selector.get_support()]



X_train = pd.DataFrame(selector.transform(scaler.transform(X_train)), columns=columns)

X_test = pd.DataFrame(selector.transform(scaler.transform(X_test)), columns=columns)



X_train.head()
cv_splitter = KFold(5)
alphas = np.logspace(0, 6, num=20)

alphas
results_l2 = cross_validate_hyperparam_choices(alphas, X_train, y, cv_splitter, Ridge)

results_l2
plot_errors_and_coef_magnitudes(results_l2, "Effect of L2 Penalty on Validation Error & Paramter Magnitude");
sigmas = np.sqrt(1 / alphas)

sigmas
results_normal = cross_validate_hyperparam_choices(sigmas, X_train, y, cv_splitter, BayesianGLM, 

                                                   is_bayesian=True, bayesian_prior_fn=pm.Normal)

results_normal
plot_errors_and_coef_magnitudes(results_normal, 

                                "Effect of Prior Variance on Validation Error & Parameter Magnitude",

                                hyperparam_name="sigma",

                                reverse_x=True);
results_l1 = cross_validate_hyperparam_choices(alphas, X_train, y, cv_splitter, Lasso)

results_l1
plot_errors_and_coef_magnitudes(results_l1, "Effect of L1 Penalty on Validation Error & Parameter Magnitude");
results_laplace = cross_validate_hyperparam_choices(sigmas, X_train, y, cv_splitter, BayesianGLM, 

                                                   is_bayesian=True, bayesian_prior_fn=pm.Laplace)

results_laplace
plot_errors_and_coef_magnitudes(results_laplace, 

                                "Effect of Laplace Prior Variance on Validation Error & Parameter Magnitude",

                                hyperparam_name="sigma",

                                reverse_x=True);