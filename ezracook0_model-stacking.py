import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel, RFE

import lightgbm as lgb

from xgboost import XGBClassifier

from eli5 import show_weights

from eli5.sklearn import PermutationImportance
train = pd.read_csv("../input/census.csv")

test = pd.read_csv("../input/test_census.csv")
# numerical

num_cols = ['age', 'education-num', 'hours-per-week']



# categorical

cat_cols = ['workclass', 'education_level', 

            'marital-status', 'occupation', 

            'relationship', 'race', 

            'sex', 'native-country']



# need log transform

log_transform_cols = ['capital-loss', 'capital-gain']
minmax = MinMaxScaler()

simp = SimpleImputer()

X_train = pd.get_dummies(train[cat_cols])

X_test = pd.get_dummies(test[cat_cols])

print(X_train.shape, X_test.shape)
X_num = simp.fit_transform(train[num_cols].values)

X_log = simp.fit_transform(train[log_transform_cols].values)

X_log = np.log1p(X_log)

X_num = minmax.fit_transform(X_num)

X_log = minmax.fit_transform(X_log)



test_num = simp.fit_transform(test[num_cols].values)

test_log = simp.fit_transform(test[log_transform_cols].values)

test_log = np.log1p(test_log)

test_num = minmax.fit_transform(test_num)

test_log = minmax.fit_transform(test_log)
X = np.concatenate((X_num,X_log,X_train.values), axis=1)

test = np.concatenate((test_num, test_log, X_test.values), axis=1)

y = train['income'].map({'<=50K': 0, '>50K': 1})
#Settings found by previous grid search.

log_reg = LogisticRegression(class_weight="balanced", C=1, penalty='l1', solver='liblinear', n_jobs=-1, max_iter=1000)

rf = RandomForestClassifier(class_weight='balanced', n_estimators=300, criterion='entropy', n_jobs=-1, min_samples_split=4, min_weight_fraction_leaf=0.0, max_depth=21, max_leaf_nodes=512)

xgb = XGBClassifier(n_estimators=300, objective='binary:logistic', silent=True, nthread=-1, colsample_bytree=0.75, gamma=1.5, learning_rate=0.1, max_depth=6, min_child_weight=1, subsample=1.0)

lgbm = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=300, objective='binary', n_jobs=-1, silent=True, max_depth=-1, max_bin = 510, subsample_for_bin=200,min_child_weight=1,colsample_bytree=0.65,lambda_l1=3, lambda_l2=3, learning_rate=0.1, min_child_samples=2,min_split_gain=0.2,num_leaves=128, subsample=0.75)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



log_reg.fit(X_train,y_train)

rf.fit(X_train,y_train)

xgb.fit(X_train,y_train)

lgbm.fit(X_train,y_train)



log_reg_predictions = log_reg.predict_proba(test)[:,-1]

rf_predictions = rf.predict_proba(test)[:,-1]

xgb_predictions = xgb.predict_proba(test)[:,-1]

lgbm_predictions = lgbm.predict_proba(test)[:,-1]



log_reg_predictions_train = log_reg.predict_proba(X)[:,-1]

rf_predictions_train = rf.predict_proba(X)[:,-1]

xgb_predictions_train = xgb.predict_proba(X)[:,-1]

lgbm_predictions_train = lgbm.predict_proba(X)[:,-1]



log_reg_rfe = RFE(log_reg,n_features_to_select=5,step=0.1)

log_reg_rfe.fit(X,y)

log_reg_rfe_predictions = log_reg_rfe.predict_proba(test)[:,-1]

log_reg_rfe_predictions_train = log_reg_rfe.predict_proba(X)[:,-1]



rf_rfe = RFE(rf,n_features_to_select=5,step=0.1)

rf_rfe.fit(X,y)

rf_rfe_predictions = rf_rfe.predict_proba(test)[:,-1]

rf_rfe_predictions_train = rf_rfe.predict_proba(X)[:,-1]



xgb_rfe = RFE(xgb,n_features_to_select=5,step=0.1)

xgb_rfe.fit(X,y)

xgb_rfe_predictions = xgb_rfe.predict_proba(test)[:,-1]

xgb_rfe_predictions_train = xgb_rfe.predict_proba(X)[:,-1]



lgbm_rfe = RFE(lgbm,n_features_to_select=5,step=0.1)

lgbm_rfe.fit(X,y)

lgbm_rfe_predictions = lgbm_rfe.predict_proba(test)[:,-1]

lgbm_rfe_predictions_train = lgbm_rfe.predict_proba(X)[:,-1]



perm_log_reg = PermutationImportance(log_reg).fit(X_test,y_test)

perm_rf = PermutationImportance(rf).fit(X_test,y_test)

perm_xgb = PermutationImportance(xgb).fit(X_test,y_test)

perm_lgbm = PermutationImportance(lgbm).fit(X_test,y_test)



sel_log_reg = SelectFromModel(perm_log_reg, threshold=0.0001, prefit=True)

X_trans = sel_log_reg.transform(X)

X_trans_test = sel_log_reg.transform(test)

log_reg.fit(X_trans,y)

log_reg_perm_predictions = log_reg.predict_proba(X_trans_test)[:,-1]

log_reg_perm_predictions_train = log_reg.predict_proba(X_trans)[:,-1]



sel_rf = SelectFromModel(perm_rf, threshold=0.0001, prefit=True)

X_trans = sel_rf.transform(X)

X_trans_test = sel_rf.transform(test)

rf.fit(X_trans,y)

rf_perm_predictions = rf.predict_proba(X_trans_test)[:,-1]

rf_perm_predictions_train = rf.predict_proba(X_trans)[:,-1]



sel_xgb = SelectFromModel(perm_xgb, threshold=0.0001, prefit=True)

X_trans = sel_xgb.transform(X)

X_trans_test = sel_xgb.transform(test)

xgb.fit(X_trans,y)

xgb_perm_predictions = xgb.predict_proba(X_trans_test)[:,-1]

xgb_perm_predictions_train = xgb.predict_proba(X_trans)[:,-1]



sel_lgbm = SelectFromModel(perm_lgbm, threshold=0.0001, prefit=True)

X_trans = sel_lgbm.transform(X)

X_trans_test = sel_lgbm.transform(test)

lgbm.fit(X_trans,y)

lgbm_perm_predictions = lgbm.predict_proba(X_trans_test)[:,-1]

lgbm_perm_predictions_train = lgbm.predict_proba(X_trans)[:,-1]
fo_train = {"log":log_reg_predictions_train,

            "rf":rf_predictions_train,

            "xgb":xgb_predictions_train,

            "lgbm":lgbm_predictions_train,

            "log_rfe":log_reg_rfe_predictions_train,

            "rf_rfe":rf_rfe_predictions_train,

            "xgb_rfe":xgb_rfe_predictions_train,

            "lgbm_rfe":lgbm_rfe_predictions_train,

            "log_reg_perm":log_reg_perm_predictions_train,

            "rf_perm":rf_perm_predictions_train,

            "xgb_perm":xgb_perm_predictions_train,

            "lgbm_perm":lgbm_perm_predictions_train} 

fo_test = {"log":log_reg_predictions,

            "rf":rf_predictions,

            "xgb":xgb_predictions,

            "lgbm":lgbm_predictions,

            "log_rfe":log_reg_rfe_predictions,

            "rf_rfe":rf_rfe_predictions,

            "xgb_rfe":xgb_rfe_predictions,

            "lgbm_rfe":lgbm_rfe_predictions,

            "log_reg_perm":log_reg_perm_predictions,

            "rf_perm":rf_perm_predictions,

            "xgb_perm":xgb_perm_predictions,

            "lgbm_perm":lgbm_perm_predictions}

X_train = pd.DataFrame(fo_train)

X_test = pd.DataFrame.from_dict(fo_test)

#print(fo_train)
param_grid = {

    "C": [0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3, 4, 5, 10],

    "penalty": ['l1', 'l2'],

    "solver": ["liblinear", "saga"]

}

linear = LogisticRegression(class_weight="balanced", n_jobs=-1, max_iter=1000)

search = GridSearchCV(estimator=linear, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

search.fit(X_train,y)
predictions = search.best_estimator_.predict(X_test)