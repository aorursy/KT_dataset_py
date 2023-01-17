# install latest version of sklearn(permutation importance needs ver.0.22 or later)

#  git+https://github.com/scikit-learn/scikit-learn.git

!pip install -U scikit-learn=="0.22"
import matplotlib

from matplotlib import font_manager

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from matplotlib import rc



from matplotlib import animation as ani

from IPython.display import Image



plt.rcParams["patch.force_edgecolor"] = True

from IPython.display import display # Allows the use of display() for DataFrames

import seaborn as sns

sns.set(style="whitegrid", palette="muted", color_codes=True)

sns.set_style("whitegrid", {'grid.linestyle': '--'})

red = sns.xkcd_rgb["light red"]

green = sns.xkcd_rgb["medium green"]

blue = sns.xkcd_rgb["denim blue"]



%matplotlib inline

%config InlineBackend.figure_format='retina'
import sklearn

sklearn.__version__
from sklearn.inspection import permutation_importance
import lightgbm as lgb

from sklearn import datasets

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import numpy as np

import pandas as pd



from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

import numpy.random as rd
data = load_boston(return_X_y=False)

X = pd.DataFrame(data["data"], columns=data["feature_names"])

y = data["target"]

X.head()
from sklearn.metrics import mean_squared_error, make_scorer

def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred)) 

mse_scorer = make_scorer(rmse)
rd.seed(123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

eval_result = {}

callbacks = [lgb.record_evaluation(eval_result)]



params_fit = {'X': X_train,

              'y': y_train,

              'eval_set': (X_test, y_test),

              'early_stopping_rounds': 5,

              'verbose': False,

              'eval_metric': 'l2',

             }

model = lgb.LGBMRegressor(objective="regression", n_estimators=100, importance_type="gain", random_state=123)

gbm = model.fit(**params_fit, callbacks=callbacks)

# gbm.evals_result_



importance_df = pd.DataFrame({"gain":model.feature_importances_}, index=X.columns).sort_values("gain", ascending=False)

print("[Feature importance]")

display(importance_df)



print("[Permutation importance]")

result = permutation_importance(model, X_train, y_train, scoring=mse_scorer, n_repeats=10, n_jobs=-1, random_state=71)

result_df = pd.DataFrame({"importances_mean":result["importances_mean"], "importances_std":result["importances_std"]}, index=X.columns)

display(result_df.sort_values("importances_mean", ascending=False))



result_df.sort_values("importances_mean", ascending=False).importances_mean.plot.barh()
rd.seed(123)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



FOLD_NUM = 5

fold_seed = 71

folds = KFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)

fold_iter = folds.split(X, y=y)



oof_preds = np.zeros(X.shape[0])

y_preds = np.zeros((FOLD_NUM, X_test.shape[0]))

models = []

importance_list = []

perm_imp_list = []

fold_label = np.zeros(X.shape[0])

for n_fold, (trn_idx, val_idx) in enumerate(fold_iter):

    print(f"========= fold:{n_fold} =========")



    X_train, X_valid = X.iloc[trn_idx], X.iloc[val_idx]

    y_train, y_valid = y[trn_idx], y[val_idx]



    params_fit = {'X': X_train,

                  'y': y_train,

                  'eval_set': (X_valid, y_valid),

                  'early_stopping_rounds': 5,

                  'verbose': False,

                  'eval_metric': 'l2',

                 }

    model = lgb.LGBMRegressor(objective="regression", n_estimators=100, importance_type="gain", random_state=123)

    gbm = model.fit(**params_fit, callbacks=callbacks)

    models += [model]

    

    fold_label[val_idx] = n_fold

    oof_preds[val_idx] = model.predict(X_valid, model.best_iteration_)

    

    rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[val_idx]))

    print(f"rmse score = {rmse: 0.5f}")

    

    # Feature importance

    importance_df = pd.DataFrame({"gain":model.feature_importances_}, index=X.columns).sort_values("gain", ascending=False)

    importance_list += [importance_df]

    print("[Importance]")

    display(importance_df)

    

    # run permutation importance

    result = permutation_importance(model, X_train, y_train, scoring=mse_scorer, n_repeats=10, n_jobs=-1, random_state=71)

    perm_imp_df = pd.DataFrame({"importances_mean":result["importances_mean"], "importances_std":result["importances_std"]}, index=X.columns)

    perm_imp_list += [perm_imp_df]

    print("[Permutation feature Importance]")

    display(perm_imp_df.sort_values("importances_mean", ascending=True))

    perm_imp_df.sort_values("importances_mean", ascending=False).importances_mean.plot.barh()

    plt.show()

    #break
perm_importances_mean = pd.concat(perm_imp_list, axis=1)["importances_mean"]

perm_importances_mean.columns = [f"fold_{i}" for i in range(FOLD_NUM)]

perm_importances_mean["ave"] = perm_importances_mean.mean(axis=1)

perm_importances_mean = perm_importances_mean.sort_values("ave", ascending=False)
perm_importances_std = pd.concat(perm_imp_list, axis=1)["importances_std"]

perm_importances_std.columns = [f"fold_{i}" for i in range(FOLD_NUM)]
plt.figure(figsize=(25, 6))

ax = plt.subplot(111)

perm_importances_mean.plot.bar(ax=ax, yerr=perm_importances_std)

plt.title("Permutation Feature Importance")
rmse = np.sqrt(mean_squared_error(y, oof_preds))

print(f"rmse score = {rmse: 0.5f}")
pred_result_df = pd.DataFrame({"ground_truth":y, "oof_preds":oof_preds, "label": [f"fold_{int(l)}" for l in  fold_label],})



plt.figure(figsize=(8,7))

ax = plt.subplot(111)

sns.scatterplot(x="ground_truth", y="oof_preds", hue="label", data=pred_result_df, hue_order=[f"fold_{int(l)}" for l in range(5)], ax=ax)