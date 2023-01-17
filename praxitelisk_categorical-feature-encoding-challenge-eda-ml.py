import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test_df = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

sub_df = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")
train_df.shape
train_df.head()
train_df.columns
train_df.dtypes
train_df.head()
train_df.isna().sum().sum()
sns.countplot(x="target", data=train_df)
for col in ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']:

    print(col)

    print(train_df[col].value_counts())
for col in ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']:

    

    f, ax = plt.subplots(1, 1, figsize=(10,5))

    

    sns.set(style="white", context="talk")

    

    sns.countplot(x=col, data=train_df, ax=ax, palette="Set1")

    f.tight_layout()
for col in ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']:

    

    f, ax = plt.subplots(1, 1, figsize=(20,8))

    

    sns.set(style="white", context="talk")

    

    sns.countplot(x=col, hue="target", data=train_df, ax=ax, palette="Set2")

    f.tight_layout()
for col in ['nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6','nom_7', 'nom_8']:

    print(col)

    print(train_df[col].value_counts())
for col in ['nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8']:

    

    f, ax = plt.subplots(1, 1, figsize=(20,8))

    

    

    sns.countplot(x=col, data=train_df, ax=ax, palette="Set1")

    f.tight_layout()
for col in ['nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8']:

    

    f, ax = plt.subplots(1, 1, figsize=(20,8))

    

    

    sns.countplot(x=col, hue="target", data=train_df, ax=ax, palette="Set2")

    f.tight_layout()
for col in ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]:

    print(col)

    print(train_df[col].value_counts())
for col in ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]:

    

    f, ax = plt.subplots(1, 1, figsize=(20,8))

    

    sns.countplot(x=col, data=train_df, ax=ax, palette="Set1")

    f.tight_layout()

    

import gc

gc.collect();
for col in ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]:

    

    f, ax = plt.subplots(1, 1, figsize=(20,8))

    

    sns.countplot(x=col, hue="target", data=train_df, ax=ax, palette="Set2")

    f.tight_layout()

    

import gc

gc.collect();
train_df.loc[train_df['bin_3'] == 'T', 'bin_3'] = 1

train_df.loc[train_df['bin_3'] == 'F', 'bin_3'] = 0



test_df.loc[test_df['bin_3'] == 'T', 'bin_3'] = 1

test_df.loc[test_df['bin_3'] == 'F', 'bin_3'] = 0



train_df.loc[train_df['bin_4'] == 'Y', 'bin_4'] = 1

train_df.loc[train_df['bin_4'] == 'N', 'bin_4'] = 0



test_df.loc[test_df['bin_4'] == 'Y', 'bin_4'] = 1

test_df.loc[test_df['bin_4'] == 'N', 'bin_4'] = 0
from sklearn.preprocessing import LabelEncoder







columns = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



for col in columns:

    

    le = LabelEncoder()

    le.fit(train_df[col].to_list()+test_df[col].to_list())

    train_df[col] = le.transform(train_df[col])

    test_df[col] = le.transform(test_df[col])

    

import gc

gc.collect();
import warnings

warnings.filterwarnings("ignore")



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import gc



for col in ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]:

    

    print(col, "one-hot-encoding")

    le = LabelEncoder()

    le.fit(train_df[col].to_list()+test_df[col].to_list())

    train_df[col] = le.transform(train_df[col])

    test_df[col] = le.transform(test_df[col])



    

    ohe = OneHotEncoder()

    ohe.fit(np.array(train_df[col].values.tolist() + test_df[col].values.tolist()).reshape(-1, 1))

    train_col = ohe.transform(train_df[col].values.reshape(-1,1)).toarray().astype(int)

    test_col = ohe.transform(test_df[col].values.reshape(-1,1)).toarray().astype(int)



    dfOneHot = pd.DataFrame(train_col, columns = [col+"."+str(int(i)) for i in range(train_col.shape[1])])

    train_df = pd.concat([train_df, dfOneHot], axis=1)



    dfOneHot = pd.DataFrame(test_col, columns = [col+"."+str(int(i)) for i in range(test_col.shape[1])])

    test_df = pd.concat([test_df, dfOneHot], axis=1)



    train_df.drop(col, axis=1, inplace=True);

    test_df.drop(col, axis=1, inplace=True);

    

    del train_col; del test_col; del dfOneHot;

    gc.collect();
train_df.columns
train_df.head()
train_df.describe()
train_columns = train_df.columns.to_list()
for elem in ["id","target"]:

    train_columns.remove(elem)
y = train_df['target']

X = train_df.drop(['target'], axis=1)

X = X[train_columns]



clf_stats_df = pd.DataFrame(columns=["clf_name", "F1-score", "auc-score"])
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import time

import seaborn as sns

import scikitplot as skplt



# create a 80/20 split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)



import xgboost as xgb



start_time = time.time()



predictions_probas_list = np.zeros([len(yvalid), 2])

predictions_test_xgb = np.zeros(len(test_df))

num_of_folds = 2

num_fold = 0

    #feature_importance_df = pd.DataFrame()



folds = StratifiedKFold(n_splits=num_of_folds, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Stratified Fold:", num_fold)

    num_fold = num_fold + 1

    print()



    clf_stra_xgb = xgb.XGBClassifier(n_estimators = 4000,

                                     objective= 'binary:logistic',

                                     nthread=-1,

                                     seed=42)



    clf_stra_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                early_stopping_rounds=100, eval_metric='auc', verbose=250)



    #fold_importance_df = pd.DataFrame()

    #fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].index

    #fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].values

    #fold_importance_df["fold"] = n_fold + 1

    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    predictions = clf_stra_xgb.predict(xvalid)

    predictions_probas = clf_stra_xgb.predict_proba(xvalid)

    predictions_probas_list += predictions_probas/num_of_folds



    predictions_test_xgb += clf_stra_xgb.predict_proba(test_df[xtrain.columns])[:,1]/num_of_folds





predictions = np.argmax(predictions_probas_list, axis=1)



print()

print(classification_report(yvalid, predictions))



print()

print("CV f1_score", f1_score(yvalid, predictions, average = "macro"))



print()

print("CV roc_auc_score", roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro"))



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_roc(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_ks_statistic(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_precision_recall(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_cumulative_gain(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_lift_curve(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(12, 38)})

xgb.plot_importance(clf_stra_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



clf_stats_df = clf_stats_df.append({"clf_name": "clf_stra_xgb",

                     "F1-score":f1_score(yvalid, predictions, average = "macro"),

                     "auc-score": roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro")}, ignore_index=True)



print()

print("elapsed time in seconds: ", time.time() - start_time)

print()

import gc

gc.collect();
from sklearn.feature_selection import SelectFromModel



selection = SelectFromModel(clf_stra_xgb, threshold=0.001, prefit=True)

select_X = selection.transform(X)



# https://stackoverflow.com/questions/41088576/is-there-away-to-output-selected-columns-names-from-selectfrommodel-method

feature_idx = selection.get_support()

feature_name = X.columns[feature_idx]

select_X = pd.DataFrame(select_X, columns = feature_name)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import time

import seaborn as sns

import scikitplot as skplt



# create a 80/20 split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(select_X, y, random_state=42, test_size=0.2, stratify = y)



import xgboost as xgb



start_time = time.time()



predictions_probas_list = np.zeros([len(yvalid), 2])

predictions_test_fs_xgb = np.zeros(len(test_df))

num_of_folds = 2

num_fold = 0

    #feature_importance_df = pd.DataFrame()



folds = StratifiedKFold(n_splits=num_of_folds, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Stratified Fold:", num_fold)

    num_fold = num_fold + 1

    print()



    clf_stra_fs_xgb = xgb.XGBClassifier(n_estimators = 4000,

                                     objective= 'binary:logistic',

                                     nthread=-1,

                                     seed=42)



    clf_stra_fs_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                early_stopping_rounds=100, eval_metric='auc', verbose=250)



    #fold_importance_df = pd.DataFrame()

    #fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].index

    #fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].values

    #fold_importance_df["fold"] = n_fold + 1

    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    predictions = clf_stra_fs_xgb.predict(xvalid)

    predictions_probas = clf_stra_fs_xgb.predict_proba(xvalid)

    predictions_probas_list += predictions_probas/num_of_folds



    predictions_test_fs_xgb += clf_stra_fs_xgb.predict_proba(test_df[xtrain.columns])[:,1]/num_of_folds





predictions = np.argmax(predictions_probas, axis=1)



print()

print(classification_report(yvalid, predictions))



print()

print("CV f1_score", f1_score(yvalid, predictions, average = "macro"))



print()

print("CV roc_auc_score", roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro"))



print()

print("elapsed time in seconds: ", time.time() - start_time)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_roc(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_ks_statistic(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_precision_recall(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_cumulative_gain(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_lift_curve(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(12, 38)})

xgb.plot_importance(clf_stra_fs_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



clf_stats_df = clf_stats_df.append({"clf_name": "clf_stra_fs_xgb",

                     "F1-score":f1_score(yvalid, predictions, average = "macro"),

                     "auc-score": roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro")}, ignore_index=True)



print()

import gc

gc.collect();
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import roc_auc_score



xtrain, xvalid, ytrain, yvalid = train_test_split(select_X, y, stratify = y, random_state=42, test_size=0.2)



# https://github.com/hyperopt/hyperopt/wiki/FMin



def objective(space):



    clf = xgb.XGBClassifier(n_estimators = 3000,

                            max_depth = space['max_depth'],

                            min_child_weight = space['min_child_weight'],

                            subsample = space['subsample'],

                            gamma=space['gamma'],

                            colsample_bytree=space['colsample_bytree'],

                            #reg_alpha = space['reg_alpha'],

                            #reg_lambda = space['reg_lambda'],

                            scale_pos_weight = space["scale_pos_weight"],

                            objective= 'binary:logistic',

                            nthread=-1,

                            seed=42)



    eval_set  = [( xtrain, ytrain), ( xvalid, yvalid)]



    clf.fit(xtrain, ytrain,

            eval_set=eval_set, eval_metric="auc",

            early_stopping_rounds=100, verbose=250)



    pred = clf.predict_proba(xvalid)[:,1]

    auc = roc_auc_score(yvalid, pred)

    print("SCORE:", auc)



    return{'loss':1-auc, 'status': STATUS_OK }



# https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster

space ={

        'eta': hp.uniform('eta', 0.025, 0.5),

        'max_depth': hp.choice('max_depth', range(3, 12)),

        'min_child_weight': hp.quniform ('min_child_weight', 1, 10, 1),

        'subsample': hp.uniform ('subsample', 0.5, 1),

        'gamma': hp.uniform('gamma', 0.1, 1),

        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),

        #'reg_alpha': hp.uniform ('reg_alpha', 0, 1),

        #'reg_lambda': hp.uniform ('reg_lambda', 0, 10),

        'scale_pos_weight': hp.uniform("scale_pos_weight", 1, np.round(train_df.target.value_counts()[0]/train_df.target.value_counts()[1],5))

    }





trials = Trials()

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=2,

            trials=trials,

            verbose = 0)



print(best)
import gc

gc.collect();
best
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import time

import seaborn as sns

import scikitplot as skplt



# create a 80/20 split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(select_X, y, random_state=42, test_size=0.2, stratify = y)



import xgboost as xgb



start_time = time.time()



predictions_probas_list = np.zeros([len(yvalid), 2])

predictions_test_fs_tuned_xgb = np.zeros(len(test_df))

num_of_folds = 2

num_fold = 0

    #feature_importance_df = pd.DataFrame()



folds = StratifiedKFold(n_splits=num_of_folds, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Stratified Fold:", num_fold)

    num_fold = num_fold + 1

    print()



    clf_stra_fs_tuned_xgb = xgb.XGBClassifier(n_estimators = 4000,

                                           objective= 'binary:logistic',

                                           nthread=-1,

                                           eta = best['eta'],

                                           max_depth=best['max_depth'],

                                           min_child_weight=best['min_child_weight'],

                                           subsample=best['subsample'],

                                           gamma=best['gamma'],

                                           colsample_bytree=best['colsample_bytree'],

                                           #reg_alpha = best['reg_alpha'],

                                           #reg_lambda = best['reg_lambda'],

                                           scale_pos_weight = best['scale_pos_weight'],

                                           seed=42)

    



    clf_stra_fs_tuned_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                early_stopping_rounds=100, eval_metric='auc', verbose=250)



    #fold_importance_df = pd.DataFrame()

    #fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].index

    #fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].values

    #fold_importance_df["fold"] = n_fold + 1

    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    predictions = clf_stra_fs_tuned_xgb.predict(xvalid)

    predictions_probas = clf_stra_fs_tuned_xgb.predict_proba(xvalid)

    predictions_probas_list += predictions_probas/num_of_folds



    predictions_test_fs_tuned_xgb += clf_stra_fs_tuned_xgb.predict_proba(test_df[xtrain.columns])[:,1]/num_of_folds





predictions = np.argmax(predictions_probas_list, axis=1)



print()

print(classification_report(yvalid, predictions))



print()

print("CV f1_score", f1_score(yvalid, predictions, average = "macro"))



print()

print("CV roc_auc_score", roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro"))



print()

print("elapsed time in seconds: ", time.time() - start_time)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_roc(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_ks_statistic(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_precision_recall(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_cumulative_gain(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_lift_curve(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(12, 38)})

xgb.plot_importance(clf_stra_fs_tuned_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



clf_stats_df = clf_stats_df.append({"clf_name": "clf_stra_fs_tuned_xgb",

                     "F1-score":f1_score(yvalid, predictions, average = "macro"),

                     "auc-score": roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro")}, ignore_index=True)



print()

import gc

gc.collect();
from imblearn.over_sampling import SMOTE



X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(select_X, y)

X_resampled = pd.DataFrame(X_resampled, columns= select_X.columns)

y_resampled = pd.Series(y_resampled)



import gc

gc.collect();
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import time

import seaborn as sns

import scikitplot as skplt



# create a 80/20 split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X_resampled, y_resampled, random_state=42, test_size=0.2, stratify=y_resampled)



import xgboost as xgb



start_time = time.time()



predictions_probas_list = np.zeros([len(yvalid), 2])

predictions_test_fs_smote_xgb = np.zeros(len(test_df))

num_of_folds = 2

num_fold = 0

    #feature_importance_df = pd.DataFrame()



folds = StratifiedKFold(n_splits=num_of_folds, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Stratified Fold:", num_fold)

    num_fold = num_fold + 1

    print()



    clf_stra_fs_smote_xgb = xgb.XGBClassifier(n_estimators = 4000,

                                     objective= 'binary:logistic',

                                     nthread=-1,

                                     seed=42)



    clf_stra_fs_smote_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                early_stopping_rounds=100, eval_metric='auc', verbose=250)



    #fold_importance_df = pd.DataFrame()

    #fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].index

    #fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].values

    #fold_importance_df["fold"] = n_fold + 1

    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    predictions = clf_stra_fs_smote_xgb.predict(xvalid)

    predictions_probas = clf_stra_fs_smote_xgb.predict_proba(xvalid)

    predictions_probas_list += predictions_probas/num_of_folds



    predictions_test_fs_smote_xgb += clf_stra_fs_smote_xgb.predict_proba(test_df[xtrain.columns])[:,1]/num_of_folds





predictions = np.argmax(predictions_probas, axis=1)



print()

print(classification_report(yvalid, predictions))



print()

print("CV f1_score", f1_score(yvalid, predictions, average = "macro"))



print()

print("CV roc_auc_score", roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro"))



print()

print("elapsed time in seconds: ", time.time() - start_time)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_roc(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_ks_statistic(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_precision_recall(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_cumulative_gain(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_lift_curve(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(12, 38)})

xgb.plot_importance(clf_stra_fs_smote_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



clf_stats_df = clf_stats_df.append({"clf_name": "clf_stra_fs_smote_xgb",

                     "F1-score":f1_score(yvalid, predictions, average = "macro"),

                     "auc-score": roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro")}, ignore_index=True)



print()

import gc

gc.collect();
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials



xtrain, xvalid, ytrain, yvalid = train_test_split(X_resampled, y_resampled, random_state=42, test_size=0.2, stratify=y_resampled)



# https://github.com/hyperopt/hyperopt/wiki/FMin



def objective(space):



    clf = xgb.XGBClassifier(n_estimators = 3000,

                            max_depth = int(space['max_depth']),

                            min_child_weight = space['min_child_weight'],

                            subsample = space['subsample'],

                            gamma=space['gamma'],

                            colsample_bytree=space['colsample_bytree'],

                            #reg_alpha = space['reg_alpha'],

                            #reg_lambda = space['reg_lambda'],

                            objective= 'binary:logistic',

                            nthread=-1,

                            seed=42)



    eval_set  = [( xtrain, ytrain), ( xvalid, yvalid)]



    clf.fit(xtrain, ytrain,

            eval_set=eval_set, eval_metric="auc",

            early_stopping_rounds=100, verbose=250)



    pred = clf.predict_proba(xvalid)[:,1]

    auc = roc_auc_score(yvalid, pred)

    print("SCORE:", auc)



    return{'loss':1-auc, 'status': STATUS_OK }



# https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster

space ={

        'eta': hp.uniform('eta', 0.025, 0.5),

        'max_depth': hp.choice('max_depth', range(3, 12)),

        'min_child_weight': hp.quniform ('min_child_weight', 1, 10, 1),

        'subsample': hp.uniform ('subsample', 0.5, 1),

        'gamma': hp.uniform('gamma', 0.1, 1),

        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),

        #'reg_alpha': hp.uniform ('reg_alpha', 0, 1),

        #'reg_lambda': hp.uniform ('reg_lambda', 0, 10)

    }





trials = Trials()

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=2,

            trials=trials,

            verbose = 0)



print(best)



import gc

gc.collect();
best
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import time

import seaborn as sns

import scikitplot as skplt



# create a 80/20 split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X_resampled, y_resampled, random_state=42, test_size=0.2, stratify=y_resampled)



import xgboost as xgb



start_time = time.time()



predictions_probas_list = np.zeros([len(yvalid), 2])

predictions_test_fs_smote_tuned_xgb = np.zeros(len(test_df))

num_of_folds = 2

num_fold = 0

    #feature_importance_df = pd.DataFrame()



folds = StratifiedKFold(n_splits=num_of_folds, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Stratified Fold:", num_fold)

    num_fold = num_fold + 1

    print()



    clf_stra_fs_smote_tuned_xgb = xgb.XGBClassifier(n_estimators = 4000,

                                           objective= 'binary:logistic',

                                           nthread=-1,

                                           eta = best['eta'],

                                           max_depth=int(best['max_depth']),

                                           min_child_weight=best['min_child_weight'],

                                           subsample=best['subsample'],

                                           gamma=best['gamma'],

                                           colsample_bytree=best['colsample_bytree'],

                                           #reg_alpha = best['reg_alpha'],

                                           #reg_lambda = best['reg_lambda'],

                                           seed=42)

    



    clf_stra_fs_smote_tuned_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                early_stopping_rounds=100, eval_metric='auc', verbose=250)



    #fold_importance_df = pd.DataFrame()

    #fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].index

    #fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].values

    #fold_importance_df["fold"] = n_fold + 1

    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    predictions = clf_stra_fs_smote_tuned_xgb.predict(xvalid)

    predictions_probas = clf_stra_fs_smote_tuned_xgb.predict_proba(xvalid)

    predictions_probas_list += predictions_probas/num_of_folds



    predictions_test_fs_smote_tuned_xgb += clf_stra_fs_smote_tuned_xgb.predict_proba(test_df[xtrain.columns])[:,1]/num_of_folds





predictions = np.argmax(predictions_probas, axis=1)



print()

print(classification_report(yvalid, predictions))



print()

print("CV f1_score", f1_score(yvalid, predictions, average = "macro"))



print()

print("CV roc_auc_score", roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro"))



print()

print("elapsed time in seconds: ", time.time() - start_time)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_roc(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_ks_statistic(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_precision_recall(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_cumulative_gain(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_lift_curve(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(12, 38)})

xgb.plot_importance(clf_stra_fs_smote_tuned_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



clf_stats_df = clf_stats_df.append({"clf_name": "clf_stra_fs_smote_tuned_xgb",

                     "F1-score":f1_score(yvalid, predictions, average = "macro"),

                     "auc-score": roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro")}, ignore_index=True)



print()

import gc

gc.collect();
del X_resampled;

del y_resampled;
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import time

import seaborn as sns

import scikitplot as skplt



# create a 80/20 split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)



from sklearn.ensemble import RandomForestClassifier



start_time = time.time()



predictions_probas_list = np.zeros([len(yvalid), 2])

predictions_test_random_forest = np.zeros(len(test_df))

num_of_folds = 2

num_fold = 0

    #feature_importance_df = pd.DataFrame()



folds = StratifiedKFold(n_splits=num_of_folds, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Stratified Fold:", num_fold)

    num_fold = num_fold + 1

    print()



    clf_stra_random_forest = RandomForestClassifier(n_estimators = 150,

                                     n_jobs=-1,

                                     random_state=42)



    clf_stra_random_forest.fit(xtrain_stra, ytrain_stra)



    #fold_importance_df = pd.DataFrame()

    #fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].index

    #fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].values

    #fold_importance_df["fold"] = n_fold + 1

    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    predictions = clf_stra_random_forest.predict(xvalid)

    predictions_probas = clf_stra_random_forest.predict_proba(xvalid)

    predictions_probas_list += predictions_probas/num_of_folds



    predictions_test_random_forest += clf_stra_random_forest.predict_proba(test_df[xtrain.columns])[:,1]/num_of_folds





predictions = np.argmax(predictions_probas, axis=1)



print()

print(classification_report(yvalid, predictions))



print()

print("CV f1_score", f1_score(yvalid, predictions, average = "macro"))



print()

print("CV roc_auc_score", roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro"))



print()

print("elapsed time in seconds: ", time.time() - start_time)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_roc(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_ks_statistic(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_precision_recall(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_cumulative_gain(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_lift_curve(yvalid, predictions_probas)



# sns.set(rc={'figure.figsize':(12, 38)})

# xgb.plot_importance(clf_stra_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



clf_stats_df = clf_stats_df.append({"clf_name": "clf_stra_random_forest",

                     "F1-score":f1_score(yvalid, predictions, average = "macro"),

                     "auc-score": roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro")}, ignore_index=True)



print()

import gc

gc.collect();
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials



xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, stratify = y, random_state=42, test_size=0.2)



# https://github.com/hyperopt/hyperopt/wiki/FMin



def objective(space):



    clf = RandomForestClassifier(n_estimators = space['n_estimators'],

                                 criterion = space['criterion'],

                                 max_features = space['max_features'],

                                 max_depth = space['max_depth'],

                                 n_jobs=-1,

                                 random_state=42)





    clf.fit(xtrain, ytrain)



    pred = clf.predict_proba(xvalid)[:,1]

    auc = roc_auc_score(yvalid, pred)

    print("SCORE:", auc)



    return{'loss':1-auc, 'status': STATUS_OK }



# https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster

space = {

    'max_depth': hp.choice('max_depth', range(1,20)),

    'max_features': hp.choice('max_features', range(1,150)),

    'n_estimators': hp.choice('n_estimators', range(100,500)),

    'criterion': hp.choice('criterion', ["gini", "entropy"])

}





trials = Trials()

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=2,

            trials=trials,

            verbose = 0)



print(best)
best
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, RepeatedStratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import time

import seaborn as sns

import scikitplot as skplt



# create a 80/20 split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)



from sklearn.ensemble import RandomForestClassifier



start_time = time.time()



predictions_probas_list = np.zeros([len(yvalid), 2])

predictions_test_tuned_random_forest = np.zeros(len(test_df))

num_of_folds = 2

num_fold = 0

    #feature_importance_df = pd.DataFrame()



folds = StratifiedKFold(n_splits=num_of_folds, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Stratified Fold:", num_fold)

    num_fold = num_fold + 1

    print()



    clf_stra_random_forest_tuned = RandomForestClassifier(n_estimators = best["n_estimators"],

                                                          max_features = best["max_features"],

                                                          max_depth = best["max_depth"],

                                                          criterion = ["gini", "entropy"][best["criterion"]],

                                                          n_jobs=-1,

                                                          random_state=42)



    clf_stra_random_forest_tuned.fit(xtrain_stra, ytrain_stra)



    #fold_importance_df = pd.DataFrame()

    #fold_importance_df["feature"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].index

    #fold_importance_df["fscore"] = pd.DataFrame.from_dict(data=clf_stra_xgb.get_fscore(), orient="index", columns=["FScore"])["FScore"].values

    #fold_importance_df["fold"] = n_fold + 1

    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    predictions = clf_stra_random_forest_tuned.predict(xvalid)

    predictions_probas = clf_stra_random_forest_tuned.predict_proba(xvalid)

    predictions_probas_list += predictions_probas/num_of_folds



    predictions_test_tuned_random_forest += clf_stra_random_forest_tuned.predict_proba(test_df[xtrain.columns])[:,1]/num_of_folds





predictions = np.argmax(predictions_probas, axis=1)



print()

print(classification_report(yvalid, predictions))



print()

print("CV f1_score", f1_score(yvalid, predictions, average = "macro"))



print()

print("CV roc_auc_score", roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro"))



print()

print("elapsed time in seconds: ", time.time() - start_time)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_roc(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_ks_statistic(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_precision_recall(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_cumulative_gain(yvalid, predictions_probas)



sns.set(rc={'figure.figsize':(8,8)})

skplt.metrics.plot_lift_curve(yvalid, predictions_probas)



# sns.set(rc={'figure.figsize':(12, 38)})

# xgb.plot_importance(clf_stra_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



clf_stats_df = clf_stats_df.append({"clf_name": "clf_stra_tuned_random_forest",

                     "F1-score":f1_score(yvalid, predictions, average = "macro"),

                     "auc-score": roc_auc_score(yvalid, predictions_probas_list[:,1], average = "macro")}, ignore_index=True)



print()

import gc

gc.collect();
clf_stats_df
sub_df['target'] = predictions_test_xgb

sub_df.to_csv('clf_xgboost.csv', index=False)



sub_df['target'] = predictions_test_fs_xgb

sub_df.to_csv('clf_xgboost_fs.csv', index=False)



sub_df['target'] = predictions_test_fs_tuned_xgb

sub_df.to_csv('clf_xgboost_fs_tuned.csv', index=False)



sub_df['target'] = predictions_test_fs_smote_xgb

sub_df.to_csv('clf_xgboost_fs_smote.csv', index=False)



sub_df['target'] = predictions_test_fs_smote_tuned_xgb

sub_df.to_csv('clf_xgboost_fs_smote_tuned.csv', index=False)



sub_df['target'] = predictions_test_random_forest

sub_df.to_csv('clf_random_forest.csv', index=False)



sub_df['target'] = predictions_test_tuned_random_forest

sub_df.to_csv('clf_random_forest_tuned.csv', index=False)