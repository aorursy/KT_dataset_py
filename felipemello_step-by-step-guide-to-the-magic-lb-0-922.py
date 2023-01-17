import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt





train_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')

test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv') 



#1 set of fake data and 2 sets of true data, as mentioned above.

synthetic_samples_indexes = np.load('../input/fakedata/synthetic_samples_indexes.npy')

private_LB = np.load('../input/fakedata/private_LB.npy')

public_LB = np.load('../input/fakedata/public_LB.npy')



#Merge the train_df with the test_df true data

full = pd.concat([train_df, pd.concat([test_df.loc[private_LB], test_df.loc[public_LB]], sort = False)], sort = False)



#Created new columns with the unique values count

for feat in ['var_' + str(x) for x in range(200)]:

    count_values = full.groupby(feat)[feat].count()

    train_df['new_' + feat] = count_values.loc[train_df[feat]].values

    test_df['new_' + feat] = count_values.loc[test_df[feat]].values



print('full_df has {} rows and {} columns'.format(full.shape[0], full.shape[1]))



#trained the model with LGB and submited.

from sklearn.model_selection import train_test_split

seed = 2319

param = {

    'num_leaves': 8,

    'min_data_in_leaf': 17,

    'learning_rate': 0.01,

    'min_sum_hessian_in_leaf': 9.67,

    'bagging_fraction': 0.8329,

    'bagging_freq': 2,

    'feature_fraction': 1,

    'lambda_l1': 0.6426,

    'lambda_l2': 0.3067,

    'min_gain_to_split': 0.02832,

    'max_depth': -1,

    'seed': seed,

    'feature_fraction_seed': seed,

    'bagging_seed': seed,

    'drop_seed': seed,

    'data_random_seed': seed,

    'objective': 'binary',

    'boosting_type': 'gbdt',

    'verbosity': -1,

    'metric': 'auc',

    'is_unbalance': True,

    'save_binary': True, 

    'boost_from_average': 'false',   

    'num_threads': 8

}



features = [c for c in train_df.columns if (c not in ['ID_code', 'target'])]



test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df['target'], test_size = test_size, random_state=42)



iterations = 110

y_hat = np.zeros([int(200000*test_size), 200])

#test_hat = np.zeros([200000, 200])

i = 0

for feature in ['var_' + str(x) for x in range(200)]: # loop over all features

    #print(feature)

    feat_choices = [feature, 'new_' + feature]

    lgb_train = lgb.Dataset(X_train[feat_choices], y_train)

    gbm = lgb.train(param, lgb_train, iterations, verbose_eval=-1)

    y_hat[:, i] = gbm.predict(X_test[feat_choices], num_iteration=gbm.best_iteration)

    #test_hat[:, i] = gbm.predict(test_df[feat_choices], num_iteration=gbm.best_iteration)

    i += 1

    

    

sub_preds = (y_hat).sum(axis=1)

score = roc_auc_score(y_test, sub_preds)

print('Your CV score is', score)
iterations = 126

param = {'bagging_fraction': 0.7693,

   'bagging_freq': 2,

   'lambda_l1': 0.7199,

   'lambda_l2': 1.992,

   'learning_rate': 0.009455,

   'max_depth': 3,

   'min_data_in_leaf': 22,

   'min_gain_to_split': 0.06549,

   'min_sum_hessian_in_leaf': 18.55,

   'num_leaves': 20,

   'feature_fraction': 1,

   'save_binary': True,

   'seed': 2319,

   'feature_fraction_seed': 2319,

   'bagging_seed': 2319,

   'drop_seed': 2319,

   'data_random_seed': 2319,

   'objective': 'binary',

   'boosting_type': 'gbdt',

   'verbosity': -1,

   'metric': 'auc',

   'is_unbalance': True,

   'boost_from_average': 'false',

   'num_threads': 6}
iterations = 126

param = {'bagging_fraction': 0.7693,

   'bagging_freq': 2,

   'lambda_l1': 0.7199,

   'lambda_l2': 1.992,

   'learning_rate': 0.009455,

   'max_depth': 3,

   'min_data_in_leaf': 22,

   'min_gain_to_split': 0.06549,

   'min_sum_hessian_in_leaf': 18.55,

   'num_leaves': 20,

   'feature_fraction': 1,

   'save_binary': True,

   'seed': 2319,

   'feature_fraction_seed': 2319,

   'bagging_seed': 2319,

   'drop_seed': 2319,

   'data_random_seed': 2319,

   'objective': 'binary',

   'boosting_type': 'gbdt',

   'verbosity': -1,

   'metric': 'auc',

   'is_unbalance': True,

   'boost_from_average': 'false',

   'num_threads': 6}



#------------------------- Trains the model ---------------------

features = [c for c in train_df.columns if (c not in ['ID_code', 'target'])]



test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df['target'], test_size = test_size, random_state=42)



iterations = 110

y_hat = np.zeros([int(200000*test_size), 200])

#test_hat = np.zeros([200000, 200])

i = 0

for feature in ['var_' + str(x) for x in range(200)]: # loop over all features

    #print(feature)

    feat_choices = [feature, 'new_' + feature]

    lgb_train = lgb.Dataset(X_train[feat_choices], y_train)

    gbm = lgb.train(param, lgb_train, iterations, verbose_eval=-1)

    y_hat[:, i] = gbm.predict(X_test[feat_choices], num_iteration=gbm.best_iteration)

    #test_hat[:, i] = gbm.predict(test_df[feat_choices], num_iteration=gbm.best_iteration)

    i += 1

#The trainning code is hidden above

#--------Calculate weights to give to each feature prediction---------

weights = []

for col in range(200):

    if roc_auc_score(y_test, y_hat[:,col]) >= 0.5:

        weights.append(roc_auc_score(y_test, y_hat[:,col]))

    else:

        weights.append(0)

        

weights = np.array(weights)

weights = (weights - weights.mean()) / weights.mean()

weights += 1



#--------Calculates AUC score------------

sub_preds_regular = (y_hat).sum(axis=1)

sub_preds_weighted = (y_hat*weights).sum(axis=1)

score_regular = roc_auc_score(y_test, sub_preds_regular)

score_weighted = roc_auc_score(y_test, sub_preds_weighted)



print('Your unweigthed score is:', score_regular)

print('Your weigthed score is:', score_weighted)
#Here I just defined where I would divide my y_test. This is not the optimal way to do it though.

#You should get samples randomly, and a good way to do that is using the train_test_split of sklearn

test_data_length = len(y_test)

validation_length = int(test_data_length/4)



#Calculates teh weights

weights = []

for col in range(200):

    if roc_auc_score(y_test[:validation_length], y_hat[:validation_length,col]) >= 0.5:

        weights.append(roc_auc_score(y_test[:validation_length], y_hat[:validation_length,col]))

    else:

        weights.append(0)

        

weights = np.array(weights)

weights = (weights - weights.mean()) / weights.mean()

weights += 1



#tests with and without weights     

sub_preds_regular = (y_hat[validation_length:]).sum(axis=1)

sub_preds_weighted = (y_hat[validation_length:]*weights).sum(axis=1)



score_regular = roc_auc_score(y_test[validation_length:], sub_preds_regular)

score_weighted = roc_auc_score(y_test[validation_length:], sub_preds_weighted)



print('Your unweigthed score is:', score_regular)

print('Your weigthed score is:', score_weighted)

if score_weighted > score_regular:

    print('Your weights ARE NOT overfitting')

else:

    print('Your weights ARE overfitting')
n = 12

var_68_values_frequency = train_df['new_var_' + str(n)]

n = 81

var_81_values_frequency = train_df['new_var_' + str(n)]



fig = plt.figure()

x_68 = var_68_values_frequency.value_counts().index

Y_68 = var_68_values_frequency.value_counts().values / sum(var_68_values_frequency.value_counts().values)

plt.scatter(x_68, Y_68)

plt.title('PDF of frequency groups - Var 12')

plt.xlabel('Group')

plt.ylabel('Probability')

plt.show()



fig = plt.figure()

x_81 = var_81_values_frequency.value_counts().index

Y_81 = var_81_values_frequency.value_counts().values / sum(var_81_values_frequency.value_counts().values)

plt.scatter(x_81, Y_81)

plt.title('PDF of frequency groups - Var 81')

plt.xlabel('Group')

plt.ylabel('Probability')

plt.show()
pd.options.mode.chained_assignment = None #disables copying warning



#Example: If the group 1, with values that repeat only once, has less than 2000 samples,

#we will merge it with group 2, that has values that repeat twice, and extinguish group 1.

#If the new formed group 2 has less than 2000 samples, than we will merge it with group 3,

#and extinguish group 2, until they are in a group with 2000 samples.

    

#sets the minimum sample quantity per group

min_n_unique_full = 2000

min_n_unique_train = int(min_n_unique_full*2/3)

min_n_unique_test = int(min_n_unique_full*1/3)



#merge train and test df

full = pd.concat([train_df, pd.concat([test_df.loc[private_LB], test_df.loc[public_LB]], sort = False)], sort = False)

true_test_df = pd.concat([test_df.loc[private_LB], test_df.loc[public_LB]], sort = False)



count = 1



#Do the groupping process for every variable in the dataset

for feat in ['var_' + str(x) for x in range(200)]:



    if count%50 == 1:

        print('Processing reached', feat)

        

    n_unique_list_full = list(set(full['new_' + feat]))

 

    n_unique_list_full.sort()

    

    #Do it forward: Group 1 -> group 2 - > groups 3

    for i in range(len(n_unique_list_full)):

        n_unique_full = n_unique_list_full[i]

        

        len_n_unique_full = len(full[feat][full['new_' + feat] == n_unique_full])

        len_n_unique_train = len(train_df[feat][train_df['new_' + feat] == n_unique_full])

        len_n_unique_test = len(true_test_df[feat][true_test_df['new_' + feat] == n_unique_full])

        

#        print(len_n_unique)



        if len_n_unique_full < min_n_unique_full or len_n_unique_train < min_n_unique_train or len_n_unique_test < min_n_unique_test :

            try:

                full['new_' + feat][full['new_' + feat] == n_unique_full] = n_unique_list_full[i+1]

                train_df['new_' + feat][train_df['new_' + feat] == n_unique_full] = n_unique_list_full[i+1]

                true_test_df['new_' + feat][true_test_df['new_' + feat] == n_unique_full] = n_unique_list_full[i+1]

                test_df['new_' + feat][test_df['new_' + feat] == n_unique_full] = n_unique_list_full[i+1]

            except:

                continue



    #Do it backwards, so small groups at the end can be merged with bigger groups before them

    #Groups n -> group n-1 -> group n-2

    for i in reversed(range(len(n_unique_list_full))):

        n_unique_full = n_unique_list_full[i]

      

        len_n_unique_full = len(full[feat][full['new_' + feat] == n_unique_full])

        len_n_unique_train = len(train_df[feat][train_df['new_' + feat] == n_unique_full])

        len_n_unique_test = len(true_test_df[feat][true_test_df['new_' + feat] == n_unique_full])

        

#        print(len_n_unique)



        if len_n_unique_full < min_n_unique_full or len_n_unique_train < min_n_unique_train or len_n_unique_test < min_n_unique_test :

            try:

                full['new_' + feat][full['new_' + feat] == n_unique_full] = n_unique_list_full[i-1]

                train_df['new_' + feat][train_df['new_' + feat] == n_unique_full] = n_unique_list_full[i-1]

                true_test_df['new_' + feat][true_test_df['new_' + feat] == n_unique_full] = n_unique_list_full[i-1]

                test_df['new_' + feat][test_df['new_' + feat] == n_unique_full] = n_unique_list_full[i-1]

            except:

                continue

    

    count+=1 

print('Done!')
for n in [2, 53, 81, 111, 121, 126, 130, 146]:

    

    print('Variable', 'var_' + str(n))



    plt.figure(figsize=(15,8))



    count = 1



    for n_unique in list(set(train_df['new_var_' + str(n)]))[:6]:



        var_tar_0 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                              (train_df['target'] == 0)]

        var_tar_1 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                              (train_df['target'] == 1)]



        var_tar_0

        samples_0 = len(var_tar_0)

        samples_1 = len(var_tar_1)

        

        if samples_0 < 20 or samples_1 < 20:

            continue

            

        samples_percentage = np.round((samples_0 + samples_1)*100/ 200000)

        plt.subplot(2, 3, count)

        sns.kdeplot(var_tar_0, shade=False, color="red", label = 'target = 0')

        sns.kdeplot(var_tar_1, shade=False, color="blue", label = 'target = 1')



        plt.title('Count = {} represents {}% of the data'.format(n_unique, samples_percentage))

        plt.xlabel('Feature Values')

        plt.ylabel('Probability')

        count += 1



    plt.tight_layout()

    plt.show()

    
for n in [117, 120]:

    

    print('Variable', 'var_' + str(n))



    plt.figure(figsize=(15,8))



    count = 1



    for n_unique in list(set(train_df['new_var_' + str(n)]))[:6]:



        var_tar_0 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                              (train_df['target'] == 0)]

        var_tar_1 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                              (train_df['target'] == 1)]



        var_tar_0

        samples_0 = len(var_tar_0)

        samples_1 = len(var_tar_1)

        

        if samples_0 < 20 or samples_1 < 20:

            continue

            

        samples_percentage = np.round((samples_0 + samples_1)*100/ 200000)

        plt.subplot(2, 3, count)

        sns.kdeplot(var_tar_0, shade=False, color="red", label = 'target = 0')

        sns.kdeplot(var_tar_1, shade=False, color="blue", label = 'target = 1')



        plt.title('Count = {} represents {}% of the data'.format(n_unique, samples_percentage))

        plt.xlabel('Feature Values')

        plt.ylabel('Probability')

        count += 1



    plt.tight_layout()

    plt.show()
var_81_values = train_df['var_' + str(n)]



var_81_values_frequency = train_df['new_var_' + str(n)]



plt.figure()

plt.scatter(var_81_values_frequency, var_81_values)



plt.title('Var_81 - values Frequency vs Values')

plt.xlabel('Values Frequency')

plt.ylabel('Values')

plt.show()
#BEFORE TRANSFORMATION

n_unique = 1

n = 126

var_126_tar_0 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                              (train_df['target'] == 0)]

var_126_tar_1 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                      (train_df['target'] == 1)]



n = 81

var_81_tar_0 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                      (train_df['target'] == 0)]

var_81_tar_1 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                      (train_df['target'] == 1)]



fig = plt.subplots()

sns.kdeplot(var_126_tar_0, shade=False, color="r", label = 'var_126_tar_0')

sns.kdeplot(var_126_tar_1, shade=False, color="blue", label = 'var_126_tar_1')

sns.kdeplot(var_81_tar_0, shade=False, color="orange", label = 'var_81_tar_0')

sns.kdeplot(var_81_tar_1, shade=False, color="black", label = 'var_81_tar_1')

plt.title('PDFs of VAR 126 and 81 BEFORE transformation')

plt.xlabel('Feature Values')

plt.ylabel('Probability')

plt.show()



#AFTER TRANSFORMATION (just doing var_126 * 2.486 -18.5)

n_unique = 1

n = 126

var_126_tar_0 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                              (train_df['target'] == 0)] * 2.486 -18.5

var_126_tar_1 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                      (train_df['target'] == 1)] * 2.486 - 18.5



n = 81

var_81_tar_0 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                      (train_df['target'] == 0)]

var_81_tar_1 = train_df['var_' + str(n)][(train_df['new_var_' + str(n)] == n_unique) &

                                      (train_df['target'] == 1)]



fig = plt.subplots()

sns.kdeplot(var_126_tar_0, shade=False, color="r", label = 'var_126_tar_0')

sns.kdeplot(var_126_tar_1, shade=False, color="blue", label = 'var_126_tar_1')

sns.kdeplot(var_81_tar_0, shade=False, color="orange", label = 'var_81_tar_0')

sns.kdeplot(var_81_tar_1, shade=False, color="black", label = 'var_81_tar_1')

plt.title('PDFs of VAR 126 and 81 AFTER transformation')

plt.xlabel('Feature Values')

plt.ylabel('Probability')

plt.show()
param = {'bagging_fraction': 0.5166,

   'bagging_freq': 3,

   'lambda_l1': 3.968,

   'lambda_l2': 1.263,

   'learning_rate': 0.00141,

   'max_depth': 3,

   'min_data_in_leaf': 17,

   'min_gain_to_split': 0.2525,

   'min_sum_hessian_in_leaf': 19.55,

   'num_leaves': 20,

   'feature_fraction': 1,

   'save_binary': True,

   'seed': 2319,

   'feature_fraction_seed': 2319,

   'bagging_seed': 2319,

   'drop_seed': 2319,

   'data_random_seed': 2319,

   'objective': 'binary',

   'boosting_type': 'gbdt',

   'verbosity': -1,

   'metric': 'auc',

   'is_unbalance': True,

   'boost_from_average': 'false',

   'num_threads': 6}



#-------------Trains and predict-----------

folds = StratifiedKFold(n_splits=4, shuffle=False, random_state=2319)

target = train_df['target']

y_hat = np.zeros([200000, 200])

test_hat = np.zeros([200000, 200])

i = 0

for feature in ['var_' + str(x) for x in range(200)]: # loop over all features

    print(feature)

    feat_choices = [feature, 'new_' + feature]

    oof = np.zeros(len(train_df))

    predictions = np.zeros(len(test_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df[feat_choices].values, target.values)):

        #print("Fold :{}".format(fold_ + 1))

        trn_data = lgb.Dataset(train_df.iloc[trn_idx][feat_choices], label=target.iloc[trn_idx])

        val_data = lgb.Dataset(train_df.iloc[val_idx][feat_choices], label=target.iloc[val_idx])

        clf = lgb.train(param, trn_data, 126, valid_sets = [trn_data, val_data], verbose_eval=-1)

        oof[val_idx] = clf.predict(train_df.iloc[val_idx][feat_choices], num_iteration=clf.best_iteration)

        predictions += clf.predict(test_df[feat_choices], num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

    

    y_hat[:, i] = oof

    test_hat[:, i] = predictions

    i += 1



#--------------Calculate the weight importance of each feature----------

weights = []

for col in range(200):

    if roc_auc_score(target, y_hat[:,col]) >= 0.5:

        weights.append(roc_auc_score(target, y_hat[:,col]))

    else:

        weights.append(0)

        

weights = np.array(weights)

weights = (weights - weights.mean()) / weights.mean()

weights += 1





#-------------Get the test prediction-----------------------------

sub_preds = (y_hat*weights).sum(axis=1)/200

print('Your CV score is', roc_auc_score(target, sub_preds))



sub_preds_test = (test_hat*weights).sum(axis=1)/200

sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = sub_preds_test

sub.to_csv('submission.csv', index=False)