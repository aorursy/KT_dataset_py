import os

import pandas as pd

import numpy as np



from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.model_selection import train_test_split

from xgboost import plot_importance



import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

X = train.copy()

#https://www.kaggle.com/dlarionov/feature-engineering-xgboost

X = X[X.item_price<100000]

X = X[X.item_cnt_day<1001]



X.dropna(axis=0, subset=['item_cnt_day'], inplace=True)



#https://www.kaggle.com/c/competitive-data-science-predict-future-sales/discussion/50591

X['item_cnt_day'][X['item_cnt_day'] < 0] = 0



X['year'] = (pd.DatetimeIndex(X['date'], dayfirst=True).year) - 2012

X['month'] = pd.DatetimeIndex(X['date'], dayfirst=True).month



'''

X['target_month'] = X['date'].str.contains(pat = '.10').astype(int)

test['target_month'] = [1]* len(test['shop_id'])

'''

X.drop(['date'], axis=1, inplace=True)



X.head()
X.drop(['item_price'], axis=1, inplace=True)

X = X.groupby(['date_block_num','shop_id','item_id']).mean().reset_index()



a = (np.mean(np.array([h.get_height() for h in 

                      sns.distplot(X[X['month'] == 11]['item_id']).patches]))

                          *100000000).round(decimals = 0)

plt.close()

f, axes = plt.subplots(2, 2)

plt.subplots_adjust(right = 1.8,top = 1.8)

sns.distplot(X['item_id'],kde = False,ax=axes[0,0])

axes[0,0].axhline(a, ls='--',color='r')

sns.distplot(test['item_id'],kde = False,ax=axes[0,1])

axes[0,1].axhline(a, ls='--',color='r')

sns.distplot(X['shop_id'],kde = False,color='r',ax=axes[1,0])

axes[1,0].axhline(test[test['shop_id'] == 50]['shop_id'].count(), ls='--')

sns.distplot(test['shop_id'],kde = False,color='r',ax=axes[1,1])

axes[1,1].axhline(test[test['shop_id'] == 50]['shop_id'].count(), ls='--')

new_X = X[ (X['month'] == 11) | (X['month'] == 10) | (X['month'] == 12) ]

dr = new_X[ (new_X['item_id'] > 1000) & (new_X['item_id'] < 8000) ].sample(frac = 0.3)

new_X = new_X.drop(dr.index)



f, axes = plt.subplots(2, 2)

plt.subplots_adjust(right = 1.8,top = 1.8)



sns.distplot(new_X['item_id'],kde = False,ax=axes[0,0])

axes[0,0].axhline(a, ls='--',color='r')



sns.distplot(test['item_id'],kde = False,ax=axes[0,1])

axes[0,1].axhline(a, ls='--',color='r')



sns.distplot(new_X['item_id'],kde = False,color='r',ax=axes[1,0])

axes[1,0].axhline(test[test['shop_id'] == 50]['shop_id'].count(), ls='--')



sns.distplot(test['shop_id'],kde = False,color='r',ax=axes[1,1])

axes[1,1].axhline(test[test['shop_id'] == 50]['shop_id'].count(), ls='--')



print(len(X[X['month'] == 11]['item_id']),len(test['item_id']))



X.drop(['year','month'], axis=1, inplace=True)

new_X.drop(['year','month'], axis=1, inplace=True)

undo = X

X = new_X
test_list = pd.DataFrame()

print('Not fulfilled:')

for i in range(60):

    test_samples = test[test['shop_id'] == i+1]['shop_id'].count()

    if test_samples != 0:

        try:

            X_test_samples = X[X['shop_id'] == i+1].sample(n=test_samples)

            X = X.drop(X_test_samples.index)

            test_list = test_list.append(X_test_samples, ignore_index=True)

        except:

            # 100% of X

            print('shop:',i+1,'samples:',test_samples,'X samples:',len(X[X['shop_id'] == i+1]))

            #print( int(len(X[X['shop_id'] == i+1])/2),'instead' )

            print( len(X[X['shop_id'] == i+1]),'instead' )

            #X_test_samples = X[X['shop_id'] == i+1].sample( n=int(len(X[X['shop_id'] == i+1])/2) )

            X_test_samples = X[X['shop_id'] == i+1].sample( n=len(X[X['shop_id'] == i+1]) )

            X = X.drop(X_test_samples.index)

            test_list = test_list.append(X_test_samples, ignore_index=True)

# Plots ----------------------------------------------------------------------------------------------------

f, axes = plt.subplots(2, 2)

plt.subplots_adjust(right = 1.8,top = 1.8)

sns.distplot(test_list['item_id'],kde = False,ax=axes[0,0])

axes[0,0].axhline(a, ls='--',color='r')

sns.distplot(test['item_id'],kde = False,ax=axes[0,1])

sns.distplot(test_list['shop_id'],kde = False,color='r',ax=axes[1,0])

sns.distplot(test['shop_id'],kde = False,color='r',ax=axes[1,1])
# items ----------------------------------------------------------------------------------------------------

undo = undo.drop(X.index)

X = undo

# Train Data ----------------------------------------------------------------------------------------------------

y_train = X.item_cnt_day

X.drop(['item_cnt_day'], axis=1, inplace=True)

X_train = X

y_valid = test_list.item_cnt_day

test_list.drop(['item_cnt_day'], axis=1, inplace=True)

X_valid = test_list
# Possible new features (saved in csv)

# https://www.kaggle.com/grindsc/fs-features

'''

# pattern df

Xf = X[['date_block_num','shop_id','item_id']]

Xf = Xf.groupby(['shop_id','item_id']).agg(['unique'])

# train functions

def train_first(shop_val,item_val):

    return Xf['date_block_num']['unique'][shop_val][item_val][0]

def train_last(shop_val,item_val):

    return Xf['date_block_num']['unique'][shop_val][item_val][-1]

def train_appearance(shop_val,item_val):

    return len(Xf['date_block_num']['unique'][shop_val][item_val])

# test functions

def test_first(shop_val,item_val):

    try:

        return Xf['date_block_num']['unique'][shop_val][item_val][0]

    except:

        return 34

def test_last(shop_val,item_val):

    try:

        return Xf['date_block_num']['unique'][shop_val][item_val][-1]

    except:

        return 34

def test_appearance(shop_val,item_val):

    try:

        return len(Xf['date_block_num']['unique'][shop_val][item_val])

    except:

        return 1

# train new features

X['first'] = X.apply(lambda row: train_first(row['shop_id'], row['item_id']), axis=1)

X['last'] = X.apply(lambda row: train_last(row['shop_id'], row['item_id']), axis=1)

X['months_amount'] = X.apply(lambda row: train_appearance(row['shop_id'], row['item_id']), axis=1)

# test new features

test['first'] = test.apply(lambda row: test_first(row['shop_id'], row['item_id']), axis=1)

test['last'] = test.apply(lambda row: test_last(row['shop_id'], row['item_id']), axis=1)

test['months_amount'] = test.apply(lambda row: test_appearance(row['shop_id'], row['item_id']), axis=1)

# save train

output = pd.DataFrame({'first': X['first'],

                       'last': X['last'],

                       'months_amount': X['months_amount']})

output.to_csv('train_date.csv', index=False)

# save test

output = pd.DataFrame({'first': test['first'],

                       'last': test['last'],

                       'months_amount': test['months_amount']})

output.to_csv('test_date.csv', index=False)

'''
my_model2 = XGBRegressor(

            n_estimators=1000,

            min_child_weight=200, 

            eta=0.5)    



my_model2.fit(X_train, y_train, 

             early_stopping_rounds=20, 

             eval_set=[(X_train, y_train), (X_valid, y_valid)],

             eval_metric="rmse",

             verbose=False)

predictions = my_model2.predict(X_valid)

mae = mean_absolute_error(predictions, y_valid)

rmse = mean_squared_error(y_valid,predictions,squared=False)

print("Mean Absolute Error:" , mae)

print("RMSE:" , rmse)



print(list(X_train))

print(my_model2.feature_importances_)

plot_importance(my_model2)

plt.show()



test['date_block_num'] = [max(X['date_block_num'])+1] * len(test['shop_id'])

pred = my_model2.predict(test[['date_block_num','shop_id','item_id']])

output = pd.DataFrame({'ID': test.ID,

                       'item_cnt_month': pred})

output.to_csv('submission.csv', index=False)