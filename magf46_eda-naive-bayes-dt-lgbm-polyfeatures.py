import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import os

import gc

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(5)
test.head(5)
TARGET='target'

ID='ID_code'
train[TARGET].hist()
def plot_dist(col, train=train, target=TARGET):

    plt.figure(figsize=(12,8))

    target_0 = train[train[target]==0].dropna()

    target_1 = train[train[target]==1].dropna()

    sns.distplot(target_0[col].values, label='target: 0')

    sns.distplot(target_1[col].values, color='red', label='target: 1')

    plt.xlabel(col)

    plt.legend()

    plt.show()



def plot_train_test_dist(col, train=train, test=test):

    plt.figure(figsize=(12,8))

    sns.distplot(train[col].values, label='target: 0')

    sns.distplot(test[col].values, color='red', label='target: 1')

    plt.xlabel(col)

    plt.legend()

    plt.show()
cols = test.drop(ID,axis=1).columns
for col in cols:

    print(col)

    plot_dist(col)
for col in cols:

    print(col)

    plot_train_test_dist(col)
correlations = train.drop(ID, axis=1).corr()
plt.figure(figsize=(20,20))

sns.heatmap(correlations)
correlations = correlations.sort_values(by=TARGET)

correlations.head(10)[TARGET]
correlations.tail(10)[TARGET]
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import roc_auc_score

from sklearn.naive_bayes import GaussianNB
def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('Features (avg over folds)')

    plt.tight_layout()
def kfold_train(train, test, num_folds, CLF, stratified = False, debug= False):

    # Divide in training/validation and test data

    train_df, test_df = train.copy(), test.copy()

    print("Start Training. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    gc.collect()

    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in [TARGET, ID]]

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[TARGET])):

        train_x, train_y = train_df[feats].iloc[train_idx], train_df[TARGET].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[TARGET].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization

        clf = CLF['class'](**CLF['params'])

        if CLF['eval_set']:

            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

                eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

            sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits            

        elif 'keras' in CLF:

            clf.fit(CLF['fit_params'])

            oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]

            sub_preds += clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits            

        else:

            clf.fit(train_x, train_y)

            oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]

            sub_preds += clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits

        if CLF['feature_importance']:

            fold_importance_df = pd.DataFrame()

            fold_importance_df["feature"] = feats

            fold_importance_df["importance"] = clf.feature_importances_

            fold_importance_df["fold"] = n_fold + 1

            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del clf, train_x, train_y, valid_x, valid_y

        gc.collect()



    print('Full AUC score %.6f' % roc_auc_score(train_df[TARGET], oof_preds))

    # Write submission file and plot feature importance

    if not debug:

        test_df[TARGET] = sub_preds

        test_df[[ID, TARGET]].to_csv(CLF['submission'], index= False)

    if CLF['feature_importance']:

        display_importances(feature_importance_df)

    return feature_importance_df
CLF = {

    'class':GaussianNB, 'params':{},'submission':'sub_naive.csv', 'eval_set':None, 'feature_importance':False

}
kfold_train(train, test, 5, CLF)
from sklearn.preprocessing import PolynomialFeatures
high_low = list(correlations.head(5).index) + list(correlations.tail(6).index)

high_low.remove('target')

poly = PolynomialFeatures(2, interaction_only=True)

poly_train = poly.fit_transform(train[high_low])

poly_test = poly.fit_transform(test[high_low])

new_train = pd.concat([train, pd.DataFrame(poly_train)], axis=1)

new_test = pd.concat([test, pd.DataFrame(poly_test)], axis=1)
CLF = {

    'class':GaussianNB, 'params':{},'submission':'sub_naive_new.csv', 'eval_set':None, 'feature_importance':False

}

kfold_train(new_train, new_test, 5, CLF)
from sklearn.tree import DecisionTreeClassifier

CLF = {

    'class':DecisionTreeClassifier, 'params':{},'submission':'sub_dt.csv', 'eval_set':None, 'feature_importance':True

}

feat_importance = kfold_train(train, test, 5, CLF)
from sklearn.tree import DecisionTreeClassifier

CLF = {

    'class':DecisionTreeClassifier, 'params':{},'submission':'sub_dt_new.csv', 'eval_set':None, 'feature_importance':True

}

feat_importance = kfold_train(new_train, new_test, 5, CLF)
from lightgbm import LGBMClassifier

CLF = {

    'class':LGBMClassifier, 'params':{'n_estimators':10000, 'learning_rate':0.01, 'max_depth':10},'submission':'sub_lgbm.csv', 'eval_set':True, 'feature_importance':True

}

feat_importance = kfold_train(train, test, 5, CLF)