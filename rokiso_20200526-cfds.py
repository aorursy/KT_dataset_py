import datetime

from time import time



import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GroupKFold

from sklearn.model_selection import train_test_split as split

import category_encoders as ce
INPUT_DIR_PATH = '../input/'
#使用メモリ削減

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics: 

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



#データ読み込み

def read_data():

    train = pd.read_csv(INPUT_DIR_PATH + 'train_data.csv')

    train = reduce_mem_usage(train)

    print('train has {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    test = pd.read_csv(INPUT_DIR_PATH + 'test_data.csv')

    test = reduce_mem_usage(test)

    print('train has {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    return train, test

#データ読み込み

train, test = read_data()
train.head()
#trainとtestの分布確認

def hist_train_vs_test(feature,bins,clip = False):

    plt.figure(figsize=(16, 8))

    if clip:

        th_train = np.percentile(train[feature], 99)

        th_test = np.percentile(test[feature], 99)

        plt.hist(x=[train[train[feature]<th_train][feature], test[test[feature]<th_test][feature]])

    else:

        plt.hist(x=[train[feature], test[feature]])

    plt.legend(['train', 'test'])

    plt.show()
hist_train_vs_test('likes',50,False)
#train中のTRUE,FALSEの分布確認



#同じヒストグラムに重ねて表示

def hist_train_target_same(feature,target,bins,clip = False):

    plt.figure(figsize=(16, 8))

    if clip:

        th_True = np.percentile(train[train[target]==False][feature], 99)

        th_False = np.percentile(train[train[target]==True][feature], 99)

        plt.hist(x=[train[train[target]==False][train[feature]<th_False][feature], train[train[target]==True][train[feature]<th_True][feature]])

    else:

        plt.hist(x=[train[train[target]==False][feature], train[train[target]==True][feature]])

    plt.legend(['False', 'True'])

    plt.show()

    

#違うヒストグラムで表示(N数が異なる場合こちらを使う)    

def hist_train_target_separate(feature,target,bins,clip = False):

    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(2, 2, 1)

    ax2 = fig.add_subplot(2, 2, 2)

    if clip:

        th_True = np.percentile(train[train[target]==False][feature], 99)

        th_False = np.percentile(train[train[target]==True][feature], 99)

        ax1.hist(x=[train[train[target]==False][train[feature]<th_False][feature]])

        ax2.hist(x=[train[train[target]==True][train[feature]<th_True][feature]],color = 'red')

    else:

        ax1.hist(x=[train[train[target]==False][feature]])

        ax2.hist(x=[train[train[target]==True][feature]],color = 'red')

    ax1.legend(['False'])

    ax2.legend(['True'])

    plt.show()
hist_train_target_separate('likes','comments_disabled',20)

#train['likes']<20000
#散布図(数値vs数値の場合)

def scatter_train_target(feature,target):

    plt.figure(figsize=(10, 10))

    plt.scatter(x=[train[feature]],y=train[target], alpha=0.4)

    plt.show()



#violin plot(カテゴリvs数値の場合)

def violin_train_target(feature, target):

    plt.figure(figsize=(13,10), dpi=80)

    sns.violinplot(x=feature, y=target, data=train, scale='width', inner='quartile')

    plt.show()
violin_train_target('comments_disabled','y')