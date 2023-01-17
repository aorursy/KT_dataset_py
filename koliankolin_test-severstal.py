!jupyter nbextension enable --py widgetsnbextension
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import catboost
from catboost.utils import create_cd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from catboost import cv
from catboost import CatBoostClassifier
from catboost.eval.catboost_evaluation import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from catboost.eval.evaluation_result import *
import gc
%matplotlib inline
train = pd.read_csv("../input/xtrain.csv")
test = pd.read_csv("../input/xtest.csv")
y = pd.read_csv("../input/ytrain.csv")['x']
train.head()
def check_nan(df):
    dct = {}
    for col in df:
        dct[col] = df[col].isnull().sum()
    return dct

def find_cats(df):
    lst = []
    for i, _ in enumerate(df):
        num_uni = df.iloc[:, i].nunique()
        if num_uni < 25:
            lst.append(i)
    return lst
cols_cat_ind = find_cats(train)
cols_num_ind = [i for i, _ in enumerate(train) if i not in cols_cat_ind]
print("Train:\n", check_nan(train), '\n')
print("Test:\n", check_nan(test))
print(cols_cat_ind)
print(cols_num_ind)
y.value_counts()
imputer_num = SimpleImputer()
imputer_cat = SimpleImputer(strategy="most_frequent")

train.iloc[:, cols_cat_ind] = imputer_cat.fit_transform(train.iloc[:, cols_cat_ind])
test.iloc[:, cols_cat_ind] = imputer_cat.transform(test.iloc[:, cols_cat_ind])
train = imputer_num.fit_transform(train)
test = imputer_num.transform(test)
del imputer_num
del imputer_cat
gc.collect();
cols = np.array(sorted(cols_cat_ind + cols_num_ind)) + 1
train_imp = pd.DataFrame(train, columns=cols)
test_imp = pd.DataFrame(test, columns=cols)
del train
del test
gc.collect();
train = train_imp
test = test_imp
cols_new_feat = list(range(58, 586))
print("Train:\n", check_nan(train), '\n')
print("Test:\n", check_nan(test))
params_cat = {
    "loss_function": "Logloss",
    "iterations": 2000,
    "custom_loss": "AUC",
    "random_seed": 42,
    "learning_rate": 0.03,
    "early_stopping_rounds": 20,
    "l2_leaf_reg": 3,
    "bagging_temperature": 1,
    "random_strength": 1,
    "leaf_estimation_method": "Newton",
}
def train_cat(train=train, test=test, y=y, cats=cols_cat_ind, params=params_cat, n_folds=5, gpu=None):
    kfols_str = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    if gpu is not None:
        params["task_type"] = "GPU"
    oof_train = []
    oof_test = []
    feat_import = []
    inds = []
    i = 1
    for train_ind, test_ind in kfols_str.split(train, y):
        print(f"Fold {i}:\n")
        X_train, X_val = train.iloc[train_ind, :], train.iloc[test_ind, :]
        y_train, y_val = y[train_ind], y[test_ind]
        model = CatBoostClassifier(**params).fit(X_train, y_train, eval_set=(X_val, y_val), verbose_eval=200, cat_features=cats)
        evals = model.get_evals_result()
        test_auc_max = max(evals["validation_0"]["AUC"])
        print('BestTestAUC: {:.4f}\n'.format(test_auc_max))
        oof_train.append(model.predict(X_val, prediction_type='RawFormulaVal'))
        oof_test.append(model.predict(test, prediction_type='RawFormulaVal'))
        inds.append(test_ind)
        feat_import.append(model.feature_importances_)
        i += 1
        del model
        del X_train
        del X_val
        del y_train
        del y_val
        gc.collect();
    return (oof_train, oof_test, feat_import, inds)
oof_train, oof_test, feat_import, inds = train_cat(train, test, y, cols_cat_ind, params_cat, 5, gpu=True)
oof_test_sum = np.zeros(100000)
for i in oof_test:
    oof_test_sum += i
oof_test_sum /= 5
oof_df_train = pd.DataFrame(np.array(oof_train).ravel(), index=np.array(inds).ravel(), columns=["oof_feat"]).sort_index()
oof_df_test = pd.DataFrame(np.array(oof_test_sum).ravel(), index=test.index, columns=["oof_feat"]).sort_index()
train = train.join(oof_df_train)
test = test.join(oof_df_test)
feat_imports = np.zeros(58)
for i in feat_import:
    feat_imports += i
feat_imports /= 5
feat_imports = dict(zip(train.columns[:-1], feat_imports))
[(k, feat_imports[k]) for k in sorted(feat_imports, key=feat_imports.get, reverse=True)]
params_cat["task_type"] = "GPU"
params_cat["iterations"] = 1.2 * params_cat["iterations"]
X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.2, stratify=y, random_state=42)
model = CatBoostClassifier(**params_cat)
model.fit(X_train, y_train, cat_features=cols_cat_ind, eval_set=(X_val, y_val), verbose=50, plot=True)
preds = model.predict_proba(test)[:, 1]
pd.DataFrame(preds, index=test.index, columns=["target"]).to_csv("predictions.csv", sep=',')
!ls
# poly = PolynomialFeatures()
# arr = poly.fit_transform(train.iloc[:, cols_num_ind])
# arr_df = pd.DataFrame(arr, columns=list(range(59, 587)))
# train_pol = pd.concat([train, arr_df], axis=1)
# del arr
# del arr_df
# del train
# gc.collect();
# train = train_pol
# arr = poly.transform(test.iloc[:, cols_num_ind])
# arr_df = pd.DataFrame(arr, columns=list(range(59, 587)))
# test_pol = pd.concat([test, arr_df], axis=1)
# del arr
# del arr_df
# del test
# gc.collect();
# test = test_pol






