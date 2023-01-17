import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import sys

import csv

import os

print(os.listdir("../input"))
np.random.seed()
def mean_std(x, mean, std):

    """normalizing data for preprocessing."""

    return float((x-mean)/std)
def mean_std_transform(x_train, X_test):

    """Mean std Data Transformation."""

    train_t = np.transpose(x_train)

    test_t = np.transpose(X_test)



    train = []

    train_c = []

    test = []

    test_c = []

    means = []

    stds = []



    for t in train_t:

        mean, std = np.mean(t), np.std(t)

        means.append(mean)

        stds.append(std)

        for x in t:

            train_c.append(mean_std(x, mean, std))

        train.append(train_c)

        train_c = []



    for i in range(len(test_t)):



        for j in range(len(test_t[i])):

            test_c.append(mean_std(test_t[i][j], means[i], stds[i]))

        test.append(test_c)

        test_c = []



    return np.transpose(train), np.transpose(test)
path = '../input/'

    

print("model extract train init")

df_train = pd.read_csv(path + 'train.csv')

df_train['ID_code'] = df_train['ID_code'].str.replace('train_', '', regex=True)



y = (np.array(df_train['target'].values.tolist()).astype(np.int)).copy()



df_train = df_train.drop(columns=['target'])

X = (np.array(df_train.values.tolist()).astype(np.float)).copy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("model extract train end")



print("model mean_std_transform init")

X_train, X_test = mean_std_transform(X_train, X_test)

print("model mean_std_transform end")
print("model extract test init")

df_test_final = pd.read_csv(path + 'test.csv')

df_test_final['ID_code'] = df_test_final['ID_code'].str.replace('test_', '', regex=True)



X_test_final = np.array(df_test_final.values.tolist()).astype(np.float)

X_test_final, a = mean_std_transform(X_test_final, [X_test])

print("model extract train end")
# create dataset for lightgbm

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict

params = {

    'bagging_freq': 5,

    'bagging_fraction': 0.38,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.045,

    'learning_rate': 0.01,

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1

}
print('Starting training...')

# train

gbm = lgb.train(

    params,

    lgb_train,

    num_boost_round=2000,

    valid_sets=lgb_eval,

    verbose_eval=5000,

    early_stopping_rounds=1000

)
print('Saving model...')

# save model to file

gbm.save_model('model.txt')
print('Starting predicting...')

# predict

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# eval

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
print('Starting final predicting...')

# predict

final_predict = gbm.predict(X_test_final, num_iteration=gbm.best_iteration)

y = [float((x)) for x in final_predict]

df = pd.DataFrame({'ID_code': ['test_' + str(x) for x in range(200000)], 'target': y})
print('y mean: {}\ny std: {}'.format(np.mean(y), np.std(y)))
df.to_csv('submission.csv', index=False)