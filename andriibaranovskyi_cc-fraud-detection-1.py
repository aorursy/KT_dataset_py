import os

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from scipy import stats

import itertools



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc



from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier

import lightgbm



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



RECALCULATE_OPTIMAL_PARAMS = False
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df['Class'].unique(), df['Class'].mean()
df
df.shape
df.info()
df.isna().mean()
plt.figure()

fig, ax = plt.subplots(8,4,figsize=(19,30))



for i in range(df.shape[1]):

    plt.subplot(8,4,i+1)

    plt.hist(df.iloc[:,i], bins=10)

    plt.xlabel(df.columns.values[i], fontsize=12)

    plt.tick_params(axis='both', which='major', labelsize=10)

fig.delaxes(ax[7][3])

plt.show();
corr, p = stats.ttest_rel(df['Time'], df['Class'])

if p > 0.05:

    print('Variables Time and Class are probably independent, p =', p, '; corr =', corr)

else:

    print('Variables Time and Class are probably dependent, p =', p,  '; corr =', corr)
df['Hour_of_day'] = (df['Time'] / 3600) % 24

corr, p = stats.ttest_rel(df['Hour_of_day'], df['Class'])



if p > 0.05:

    print('Variables Hour_of_day and Class are probably independent, p =', p, '; corr =', corr)

else:

    print('Variables Hour_of_day and Class are probably dependent, p =', p,  '; corr =', corr)
df.drop(columns=['Time'], inplace=True)
x_train, x_test, y_train, y_test = train_test_split(df.loc[:,df.columns != 'Class'],

                                                    df['Class'],

                                                    test_size=0.2, random_state=19)

x_train.reset_index(drop=True, inplace=True)

x_test.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)

y_test.reset_index(drop=True, inplace=True)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
# create masks list

already_used_indexes = pd.DataFrame(np.zeros(x_train.shape[0]))

positive_indexes = y_train[y_train == 1].index

positive_size = positive_indexes.shape[0]

already_used_indexes.loc[positive_indexes, 0] = 1

masks = []



while already_used_indexes.loc[already_used_indexes[0]==0].shape[0] > positive_size:

    mask = already_used_indexes.loc[already_used_indexes[0]==0].sample(positive_size, random_state=444).index

    already_used_indexes.loc[mask, 0] = 1

    masks.append(mask)

np.array(masks), np.array(masks).shape
# Random Forest learning



if RECALCULATE_OPTIMAL_PARAMS:

    max_depth_list = [5, 10, 20]

    max_features_list = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]

    min_samples_leaf_list = [0.05, 0.2, 0.35, 0.5]



    #params after tuning

    max_depth_list = [10]

    max_features_list = [0.65]

    min_samples_leaf_list = [0.05]



    combinations = list(itertools.product(max_depth_list, max_features_list, min_samples_leaf_list))



    best_score = 0

    for params in tqdm(combinations):

        print(params)

        max_depth, max_features, min_samples_leaf = params

        scores = []

        for mask in masks:

            curr_mask = np.concatenate([mask, positive_indexes])

            curr_x = x_train.loc[curr_mask]

            curr_y = y_train.loc[curr_mask]

            rf_clf = RandomForestClassifier(random_state=444, criterion="gini", max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf)

            rf_clf.fit(curr_x, curr_y)

            y_pred_rf = rf_clf.predict_proba(x_test)

            scores.append(roc_auc_score(y_test, y_pred_rf[:,1]))



        print('Random Forest AUC score ', np.mean(scores))

        if np.mean(scores) > best_score:

            best_score = np.mean(scores)

            best_params = params



    print('best_score', best_score)

    print('best_params', best_params)

# Catboost learning



if RECALCULATE_OPTIMAL_PARAMS:

    iterations_list = [1000]

    combinations = iterations_list



    best_score = 0

    for params in tqdm(combinations):

        print(params)

        iterations = params

        scores = []

        for mask in tqdm(masks):

            curr_mask = np.concatenate([mask, positive_indexes])

            curr_x = x_train.loc[curr_mask]

            curr_y = y_train.loc[curr_mask]

            cb_clf = CatBoostClassifier(

                eval_metric='AUC',

                silent = True,

                random_seed=444,

                iterations=iterations

            )

            cb_model = cb_clf.fit(curr_x, curr_y)

            y_pred_cb = cb_model.predict_proba(x_test)[:,1]

            scores.append(roc_auc_score(y_test, y_pred_cb))



        print('Catboost AUC score ', np.mean(scores))

        if np.mean(scores) > best_score:

            best_score = np.mean(scores)

            best_params = params

    print('best_score', best_score)

    print('best_params', best_params)

# # LightGBM learning



if RECALCULATE_OPTIMAL_PARAMS:

    #params after tuning

    num_leaves = [5, 10, 20, 40]

    learning_rate = [0.01, 0.02, 0.05, 0.1, 0.25]

    lambda_l1 = [0, 0.1, 0.5, 2, 5]

    lambda_l2 = [0, 0.1, 0.5, 2, 5]

    feature_fraction = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1]

    bagging_fraction = [0.25, 0.5, 0.75, 1]

    bagging_freq = [0, 5, 10, 25, 50]

    iterations = [100, 1000]



    num_leaves = [10]

    learning_rate = [0.05]

    lambda_l1 = [0]

    lambda_l2 = [2]

    feature_fraction = [1]

    bagging_fraction = [0.75]

    bagging_freq = [10]

    iterations = [100]



    combinations = list(itertools.product(num_leaves, learning_rate, lambda_l1, lambda_l2, feature_fraction, bagging_fraction, bagging_freq, iterations))





    best_score = 0

    for params in tqdm(combinations):

        print(params)

        num_leaves, learning_rate, lambda_l1, lambda_l2, feature_fraction, bagging_fraction, bagging_freq, iterations = params





        parameters = {

            'objective': 'binary',

            'metric': 'auc',

            'is_unbalance': 'true',

            'boosting': 'gbdt',

            'verbosity': 1,

            'num_leaves': num_leaves,

            'lambda_l1': lambda_l1,

            'lambda_l2': lambda_l2,

            'feature_fraction': feature_fraction,

            'learning_rate': learning_rate,

            'bagging_fraction': bagging_fraction,

            'bagging_freq': bagging_freq,

            'iterations': iterations

        }



        scores = []

        for mask in masks:

            curr_mask = np.concatenate([mask, positive_indexes])

            curr_x = x_train.loc[curr_mask]

            curr_y = y_train.loc[curr_mask]

            curr_train_data = lightgbm.Dataset(curr_x, label=curr_y)

            lgb_model = lightgbm.train(parameters,

                                   curr_train_data,

                                   verbose_eval=False,

                                )

            y_predict_lgb = lgb_model.predict(x_test)

            scores.append(roc_auc_score(y_test, y_predict_lgb))    

        print('LightGBM AUC score ', np.mean(scores))

        if np.mean(scores) > best_score:

            best_score = np.mean(scores)

            best_params = params



    print('best_score', best_score)

    print('best_params', best_params)

# Evaluate predicts



rf_predicts = []

cb_predicts = []

lgbm_predicts = []

for mask in tqdm(masks):

    curr_mask = np.concatenate([mask, positive_indexes])

    curr_x = x_train.loc[curr_mask]

    curr_y = y_train.loc[curr_mask]



    #Random Forest

    max_depth = 10

    max_features = 0.65

    min_samples_leaf = 0.05

    rf_clf = RandomForestClassifier(random_state=444, criterion="gini", max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf)

    rf_clf.fit(curr_x, curr_y)

    y_pred_rf = rf_clf.predict_proba(x_test)

    rf_predicts.append(y_pred_rf)



    #CatBoost

    cb_clf = CatBoostClassifier(

        eval_metric='AUC',

        silent = True,

        random_seed=444,

        iterations=1000

    )

    cb_model = cb_clf.fit(curr_x, curr_y)

    y_pred_cb = cb_model.predict_proba(x_test)[:,1]

    cb_predicts.append(y_pred_cb)



    #lightGBM



    num_leaves = 10

    lambda_l1 = 0

    lambda_l2 = 2

    feature_fraction = 1

    learning_rate = 0.05

    bagging_fraction = 0.75

    bagging_freq = 10

    iterations = 100



    parameters = {

        'objective': 'binary',

        'metric': 'auc',

        'is_unbalance': 'true',

        'boosting': 'gbdt',

        'verbosity': 1,

        'num_leaves': num_leaves,

        'lambda_l1': lambda_l1,

        'lambda_l2': lambda_l2,

        'feature_fraction': feature_fraction,

        'learning_rate': learning_rate,

        'bagging_fraction': bagging_fraction,

        'bagging_freq': bagging_freq,

        'iterations': iterations

    }

    curr_train_data = lightgbm.Dataset(curr_x, label=curr_y)

    lgb_model = lightgbm.train(parameters,

                           curr_train_data,

                           verbose_eval=False,

                        )

    y_predict_lgb = lgb_model.predict(x_test)

    lgbm_predicts.append(y_predict_lgb)

# Evaluate optimal coeffs for blending

if RECALCULATE_OPTIMAL_PARAMS:    

    best_score = 0

    for rf_cf in tqdm(np.arange(0, 1.01, 0.01)):

        for cb_cf in np.arange(0, 1.01 - rf_cf, 0.01):

            lgbm_cf = 1 - rf_cf - cb_cf

            scores = []

            for i in range(len(rf_predicts)):

                rf_pred = rf_predicts[i][:,1]

                cb_pred = cb_predicts[i]

                lgbm_pred = lgbm_predicts[i]    

                weighted_pred = rf_cf * rf_pred + cb_cf * cb_pred + lgbm_cf * lgbm_pred #vector of weighted predicts

                scores.append(roc_auc_score(y_test, weighted_pred))



            if np.mean(scores) > best_score:

                best_score = np.mean(scores)

                best_cfs = rf_cf, cb_cf, lgbm_cf

    print('rf_cf', "cb_cf", "lgbm_cf")

    print(best_cfs)

    

# best cfs

# rf_cf cb_cf lgbm_cf

# (0.0, 0.64, 0.36)
# Catboost optimal treshold

if RECALCULATE_OPTIMAL_PARAMS:

    best_score = 0

    for treshold in tqdm(np.arange(0.986, 0.995, 0.001)):

        scores = []

        for predict in cb_predicts:

            scores.append(f1_score(y_test, predict>treshold))  

        print('th ', treshold)

        print('score ', np.mean(scores))

        if np.mean(scores) > best_score:

            best_score = np.mean(scores)

            best_th = treshold

    print('best_score', best_score)

    print("best_th", best_th)

# best th 0.99
# LightGBM optimal treshold

if RECALCULATE_OPTIMAL_PARAMS:

    best_score = 0

    for treshold in tqdm(np.arange(0.9781, 0.99, 0.001)):

        scores = []

        for predict in lgbm_predicts:

            scores.append(f1_score(y_test, predict>treshold))  

        print('th ', treshold)

        print('score ', np.mean(scores))

        if np.mean(scores) > best_score:

            best_score = np.mean(scores)

            best_th = treshold

    print('best_score', best_score)

    print("best_th", best_th)

# best th 0.985
# Final predict

lgbm_cf = 0.36

cb_cf = 0.64

lgbm_th = 0.985

cb_th = 0.99

mean_lgb_predict = np.array(lgbm_predicts).mean(axis=0)

mean_cb_predict = np.array(cb_predicts).mean(axis=0)

weighted_predict_for_auc = mean_lgb_predict * lgbm_cf + mean_cb_predict * cb_cf

weighted_predict_for_f1 = (mean_lgb_predict * lgbm_cf / lgbm_th + mean_cb_predict * cb_cf / cb_th) > 1

print('F1 score: ', f1_score(y_test, weighted_predict_for_f1))

print('AUC score: ', roc_auc_score(y_test, weighted_predict_for_auc))
fpr, tpr, threshold = roc_curve(y_test, weighted_predict_for_auc)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()