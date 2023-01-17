import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from time import clock, time

from datetime import datetime



from sklearn import metrics  

from sklearn.preprocessing import MinMaxScaler



from keras.models import Sequential, load_model, model_from_json

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import np_utils



%matplotlib inline



from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import scale, label_binarize

from sklearn.svm import SVC
plt.rcParams["font.size"] = 30

plt.rcParams['figure.figsize'] = [12, 8]

plt.rcParams['figure.titlesize'] = 20

plt.rcParams["axes.titlesize"] = 20

plt.rcParams["axes.labelsize"] = 20

plt.rcParams["xtick.labelsize"] = 20

plt.rcParams["ytick.labelsize"] = 20

plt.rcParams["legend.fontsize"] = 10
BASE_DIR = "."

DEBUG = False



np.random.seed(42)
df=pd.read_csv('../input/train.csv')

X_test=pd.read_csv('../input/test.csv')
df.head()
df.target.value_counts()
df.shape
df['var_139'].hist()
df.head(2)
X = df.loc[:,'var_0':]

y = df["target"]
#X, X_hideout, y, y_hideout = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
#df=X.join(y)
df.head(2)
# Class count

count_class_0, count_class_1 = df.target.value_counts()



# Divide by class

df_class_0 = df[df['target'] == 0]

df_class_1 = df[df['target'] == 1]
df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_sampled = pd.concat([df_class_0, df_class_1_over], axis=0)



print('Random over-sampling:')

print(df_sampled.target.value_counts())



df_sampled.target.value_counts().plot(kind='bar', title='Count (target)');
df['var_81'].hist()
df.describe()
df.info()
df_sampled.target.value_counts()
df_sampled.corr()
sns.heatmap(df_sampled.corr())
df_sampled1=df_sampled.drop('target', axis=1)
df_sampled.shape
df_sampled1.head(2)
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc



from sklearn import model_selection



from sklearn.model_selection import StratifiedShuffleSplit
X = df_sampled.loc[:,'var_0':]

y = df_sampled["target"]
X.shape, y.shape
df_sampled["target"].value_counts()
import skopt



from skopt import BayesSearchCV

from skopt.space import Real, Categorical, Integer

from skopt.callbacks import DeadlineStopper # Stop the optimization before running out of a fixed budget of time.

from skopt.callbacks import VerboseCallback # Callback to control the verbosity

from skopt.callbacks import DeltaXStopper # Stop the optimization If the last two positions at which the objective has been evaluated are less than delta



from sklearn.metrics import make_scorer

from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error



import pprint

import datetime as dt

import lightgbm as lgb

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedShuffleSplit
# Reporting util for different optimizers

def report_perf(optimizer, X, y, title, callbacks=None):

    """

    A wrapper for measuring time and performances of different optmizers

    

    optimizer = a sklearn or a skopt optimizer

    X = the training set 

    y = our target

    title = a string label for the experiment

    """

    start = time()

    if callbacks:

        optimizer.fit(X, y, callback=callbacks)

    else:

        optimizer.fit(X, y)

    best_score = optimizer.best_score_

    best_score_std = optimizer.cv_results_['std_test_score'][optimizer.best_index_]

    best_params = optimizer.best_params_

    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "

           +u"\u00B1"+" %.3f") % (time() - start, 

                                  len(optimizer.cv_results_['params']),

                                  best_score,

                                  best_score_std))    

    print('Best parameters:')

    pprint.pprint(best_params)

    print()

    return best_params
# Setting a 5-fold stratified cross-validation (note: shuffle=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# Converting average precision score into a scorer suitable for model selection

avg_prec = make_scorer(average_precision_score, greater_is_better=True, needs_proba=True)
df_sampled1.head(2)
clf = lgb.LGBMClassifier(boosting_type='gbdt',

                         class_weight='balanced',

                         objective='binary',

                         n_jobs=1, 

                         verbose=0)



search_spaces = {

        'learning_rate': Real(0.01, 1.0, 'log-uniform'),

        'num_leaves': Integer(2, 50),

        'max_depth': Integer(0, 20),

        'min_child_samples': Integer(0, 200),

        'max_bin': Integer(100, 100000),

        'subsample': Real(0.01, 1.0, 'uniform'),

        'subsample_freq': Integer(0, 10),

        'colsample_bytree': Real(0.01, 1.0, 'uniform'),

        'min_child_weight': Integer(0, 10),

        'subsample_for_bin': Integer(100000, 500000),

        'reg_lambda': Real(1e-9, 1000, 'log-uniform'),

        'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),

        'scale_pos_weight': Real(1e-6, 500, 'log-uniform'),

        'n_estimators': Integer(500, 2000)        

        }



opt = BayesSearchCV(clf,

                    search_spaces,

                    scoring=avg_prec,

                    cv=skf,

                    n_iter=40,

                    n_jobs=-1,

                    return_train_score=False,

                    refit=True,

                    optimizer_kwargs={'base_estimator': 'GP'},

                    random_state=22)

    

best_params = report_perf(opt, X, y,'LightGBM', 

                          callbacks=[DeltaXStopper(0.001), 

                                     DeadlineStopper(60*5)])

#DeadlineStopper and DeltaXStopper are skopt callbacks that control the total time spent and 

#the improvement of a BayesSearchCV (in our implementation to be called with report_perf, using the parameter callbacks=[]).
# this is the first method using the bayesian optimizer fit classifier, we will try also with seperate classifier

#which will give a LB %89.7 at 10K fold



Index=X_test['ID_code']

X_test=X_test.drop('ID_code', axis=1)
sub_preds=opt.predict_proba(X_test)[:,1]
submissions=pd.DataFrame({'ID_code':Index, 'target': sub_preds})
submissions.to_csv('submissions_lgb_samplingFEtuningbest1.csv', index=False)
val_preds = np.zeros(X.shape[0])

sub_preds = np.zeros(X_test.shape[0])

kf = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)



for i, (train_index, test_index) in enumerate(kf.split(X, y)):

    # Create stratified 5/10 CV data

    X_train, y_train = X.iloc[train_index].copy(), y.iloc[train_index]

    X_valid, y_valid = X.iloc[test_index].copy(), y.iloc[test_index].copy()

    print("\nFold ", i)

#Parameters defined by Bayesian Optimizer https://www.kaggle.com/lucamassaron/kaggle-days-paris-gbdt-workshop

    clf = LGBMClassifier(

            n_estimators=1704,

            learning_rate=0.16624226726409647, 

            num_leaves=4,

            colsample_bytree=.8,

            subsample=.7,

            subsample_freq= 7,

            subsample_for_bin= 375140,

            max_depth=5,

            reg_alpha=1.081049236893711e-05,

            reg_lambda=1.043686239159047,

            min_split_gain=.01,

            min_child_weight=4,

            min_child_samples=22,

            silent=-1,

            verbose=-1,

        )

        

    clf.fit(X_train, y_train, 

                eval_set= [(X_train, y_train), (X_valid, y_valid)], 

                eval_metric='auc', verbose=100, early_stopping_rounds=100

               )

            

    val_preds[test_index] = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:, 1]

    sub_preds += clf.predict_proba(X_test.loc[:,'var_0':], num_iteration=clf.best_iteration_)[:, 1] / kf.n_splits
val_preds[test_index] = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:,1]
predictions_lgbm_01 = np.where(val_preds[test_index] > 0.5, 1, 0)
from sklearn.metrics import f1_score



lgb_F1 = f1_score(y_valid, predictions_lgbm_01, average = 'weighted')

print("The Light GBM F1 for Validation Data is", lgb_F1)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_valid,predictions_lgbm_01)
predictions_lgbm_01.shape, y_valid.shape
clf.best_score_
#Print Confusion Matrix

plt.figure()

cm = confusion_matrix(y_valid, predictions_lgbm_01)

labels = ['0', '1']

plt.figure(figsize=(8,6))

sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')

plt.show()
#hide_out_preds= clf.predict_proba(X_hideout, num_iteration=clf.best_iteration_)[:, 1]
#predictions_lgbm_02 = np.where(hide_out_preds > 0.5, 1, 0)
#lgb_F1 = f1_score(y_hideout, predictions_lgbm_02, average = 'weighted')

#print("The Light GBM F1 on Hideout Data is", lgb_F1)
#from sklearn.metrics import roc_auc_score

#roc_auc_score(y_hideout,sub_preds)
#Plot Variable Importances

#lgb.plot_importance(clf, max_num_features=21, importance_type='gain')
#X_test=pd.read_csv('test.csv')
X_test.head(2)
Index=X_test['ID_code']

X_test=X_test.drop('ID_code', axis=1)
#test_data_preds= clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1]
#predictions_lgbm_03 = np.where(test_data_preds > 0.5, 1, 0)
submissions=pd.DataFrame({'ID_code':Index, 'target': sub_preds})

# Subpreds from the k-fold iteration above
submissions.to_csv('submissions_lgb_samplingFEtuningbest2.csv', index=False)
#train_df=df_sampled1.copy()

#test_df=X_test.copy()
df_sampled2=df_sampled1.drop('ID_code', axis=1)

df_sampled2.head(2)
# Display/plot feature importance

def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]



    plt.figure(figsize=(8, 10))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances.png')
boosting = ["goss","dart"]

def kfold_lightgbm(train_df, test_df, num_folds, stratified = False, boosting = boosting[0]):

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))



    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=2045)



    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    

    # k-fold

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, df_sampled["target"])):

        train_x, train_y = train_df.iloc[train_idx], df_sampled["target"].iloc[train_idx]

        valid_x, valid_y = train_df.iloc[valid_idx], df_sampled["target"].iloc[valid_idx]



        # set data structure

        lgb_train = lgb.Dataset(train_x,label=train_y,free_raw_data=False)

        lgb_test = lgb.Dataset(valid_x,label=valid_y,free_raw_data=False)



        # params optimized by optuna

        params ={

                        'task': 'train',

                        'boosting': 'goss',

                        'objective': 'binary',

                        'metric': 'auc',

                        'learning_rate': 0.01,

                        'subsample': 0.8,

                        'max_depth': -1,

                        'top_rate': 0.9064148448434349,

                        'num_leaves': 32,

                        'min_child_weight': 41.9612869171337,

                        'other_rate': 0.0721768246018207,

                        'reg_alpha': 9.677537745007898,

                        'colsample_bytree': 0.5665320670155495,

                        'min_split_gain': 9.820197773625843,

                        'reg_lambda': 8.2532317400459,

                        'min_data_in_leaf': 21,

                        'verbose': -1,

                        'seed':int(2**n_fold),

                        'bagging_seed':int(2**n_fold),

                        'drop_seed':int(2**n_fold)

                        }



        reg = lgb.train(

                        params,

                        lgb_train,

                        valid_sets=[lgb_train, lgb_test],

                        valid_names=['train', 'test'],

                        num_boost_round=7000,early_stopping_rounds= 200,

                        verbose_eval=100,

                        )



        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)

        sub_preds += reg.predict(test_df, num_iteration=reg.best_iteration) / folds.n_splits



        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = train_x.columns

        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d roc_auc_score : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del reg, train_x, train_y, valid_x, valid_y

        #gc.collect()



    # display importances

    display_importances(feature_importance_df)

    

        # save submission file

    submission = pd.read_csv("../input/sample_submission.csv")

    submission['target'] = sub_preds

    submission.to_csv(boosting+".csv", index=False)

    display(submission.head())

    return (submission)
kfold_lightgbm(df_sampled2, X_test, num_folds=5, stratified = True, boosting = boosting[0])
#folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=326)

#folds.n_splits
# Save the model

import joblib

#save model

joblib.dump(lgb, 'lgb_train model')
submissions_stacked