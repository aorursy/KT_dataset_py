import numpy as np

import pandas as pd

train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')

train.sort_index(inplace=True)

train_y = train['target']; test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True); test.drop('id', axis=1, inplace=True)

from sklearn.metrics import roc_auc_score

cat_feat_to_encode = train.columns.tolist();  smoothing=0.20

import category_encoders as ce

oof = pd.DataFrame([])

from sklearn.model_selection import StratifiedKFold

for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, train_y):

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])

    oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)

ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

ce_target_encoder.fit(train, train_y)

train = oof.sort_index()

test = ce_target_encoder.transform(test)
from sklearn import linear_model

glm = linear_model.LogisticRegression( random_state=1, solver='lbfgs', max_iter=2020, fit_intercept=True, penalty='none', verbose=0); glm.fit(train, train_y)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
n_folds = 5

def auc_score(model):

    kf = KFold( n_folds, shuffle= True).get_n_splits(train.values)

    auc_score = cross_val_score(model, train.values, train_y, scoring = "roc_auc", cv = kf)

    return auc_score
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = auc_score(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = auc_score(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = auc_score(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

score = auc_score(glm)

print("Logistic Regression: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

# score = auc_score(KRR)

# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = auc_score(GBoost)

# print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = auc_score(model_xgb)

# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class average_stacking(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self,models):

        self.models = models

    def fit(self, x,y):

        self.model_clones = [clone(x) for x in self.models]

        

        for model in self.model_clones:

            model.fit(x,y)

        return self

    def predict(self, x):

        preds = np.column_stack([

            model.predict(x) for model in self.model_clones

        ])

        return np.mean(preds, axis = 1)
averaged_models = average_stacking(models = (ENet, glm,model_lgb, lasso))



score = auc_score(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
averaged_models.fit(train.values, train_y)

avg_pred = averaged_models.predict(test)
pd.DataFrame({'id': test_id, 'target': avg_pred}).to_csv('submission.csv', index=False)