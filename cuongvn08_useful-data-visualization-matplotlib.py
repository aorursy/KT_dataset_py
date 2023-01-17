import numpy as np

import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

import gc

import xgboost as xgb
def visualize():

    # config

    train_path = '../input/train.csv'



    # load data

    print('\nloading data ... ')

    train_df = pd.read_csv(train_path)

    train_df.drop(['id'], axis = 1, inplace = True)

    print('train shape: ', train_df.shape)

    print('feature types: ', Counter(train_df.dtypes.values))

    print('features: ')

    for feature in train_df:

        print(' ', feature)



    # check NA ratio

    # replace -1 by NA because -1 in data indicate that the feature was missing 

    print('\nchecking NA ratio ... ')

    train_df_copied = train_df

    train_df_copied = train_df_copied.replace(-1, np.NaN) # Values of -1 indicate that the feature was missing from the observation"



    na_ratio = (train_df_copied.isnull().sum() / len(train_df_copied)).sort_values(ascending=False)

    print('NA ratio: ')

    print(na_ratio)



    del train_df_copied

    gc.collect()

    

    # show the target feature

    print('\nshowing the target feature ... ')

    zero_count = (train_df['target']==0).sum()

    one_count = (train_df['target']==1).sum()

    plt.bar(np.arange(2), [zero_count, one_count])

    plt.show()

    

    print('target 0: ', zero_count)

    print('target 1: ', one_count)

    

    # show feature's distribution

    print('\ndislaying distribution of features ... ')

    for feature in train_df:

        plt.figure(figsize=(8,6))

        plt.scatter(range(train_df.shape[0]), np.sort(train_df[feature].values))

        plt.xlabel('index', fontsize=12)

        plt.ylabel(feature, fontsize=12)

        plt.show() 

        

    # compute features's correlation

    print('\ncomputing correlation of the features and showing the most positive and negative correlated features ... ')

    f, ax = plt.subplots(figsize = (15, 15))

    plt.title('correlation of continuous features')

    sns.heatmap(train_df.corr(), ax = ax)

    plt.show()

    

    corr_values = train_df.corr().unstack().sort_values(ascending=False)

    print(type(corr_values))

    for pair, value in corr_values.iteritems():

        if abs(value) > 0.3 and abs(value) < 1.0:

            print(pair, value)

    

    # binary features inpection

    print('\n inspecting binary features ... ')

    bin_cols = [col for col in train_df.columns if '_bin' in col]

    zero_list = []

    one_list = []

    for col in bin_cols:

        zero_list.append((train_df[col]==0).sum())

        one_list.append((train_df[col]==1).sum())

        

    plt.figure(figsize = (10, 10))

    p1 = plt.bar(np.arange(len(bin_cols)), zero_list, width = 0.5)

    p2 = plt.bar(np.arange(len(bin_cols)), one_list, bottom = zero_list, width = 0.5)

    plt.xticks(np.arange(len(bin_cols)), bin_cols, rotation = 90)

    plt.legend((p1[0], p2[0]), ('zero count', 'one count'))

    plt.show()

    

    # compute feature importance

    print('\n computing feature importance ... ')

    xgb_params = {

        'eta': 0.05,

        'max_depth': 8,

        'subsample': 0.7,

        'colsample_bytree': 0.7,

        'objective': 'reg:linear',

        'silent': 1,

        'seed' : 0

    }

    

    train_y = train_df['target'].values

    train_x = train_df.drop(['target'], axis=1)



    d_train = xgb.DMatrix(train_x, train_y, feature_names=train_x.columns.values)

    model = xgb.train(dict(xgb_params, silent=0), d_train, num_boost_round = 100)

    

    importance = model.get_fscore()

    features_df = pd.DataFrame()

    features_df['feature'] = importance.keys()

    features_df['fscore'] = importance.values()

    features_df['fscore'] = features_df['fscore'] / features_df['fscore'].sum()

    features_df.sort_values(by = ['fscore'], ascending = True, inplace = True)

    

    plt.figure()

    features_df.plot(kind = 'barh', x = 'feature', y='fscore', legend = False, figsize = (10, 10))

    plt.title('XGBoost Feature Importance')

    plt.xlabel('fscore')

    plt.ylabel('features')

    plt.show()



    print(features_df)

    

    # release

    del train_df

    gc.collect()

    

if __name__ == "__main__":

    visualize()

    print('\n\n\nThe end.')