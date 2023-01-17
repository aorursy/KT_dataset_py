import pandas as pd

import numpy as np

import xgboost as xgb



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm

from collections import Counter

from catboost import *

from sklearn.ensemble import *

from sklearn.model_selection import *

from sklearn.metrics import accuracy_score
sgd = pd.read_csv('../input/small_group_description.csv')

items_dict = dict(zip(sgd.small_group_code.values, sgd.small_group.values))

items_dict
transactions_train=pd.read_csv('../input/transactions_train.csv')
train_target=pd.read_csv('../input/train_target.csv')
transactions_train.head(5)
train_target.head(5)
transactions_train
agg_features=transactions_train.groupby('client_id')['amount_rur'].agg(['sum','mean','std','min','max','count','mad','var','sem','skew','quantile']).reset_index() 
agg_features.head(5)
counter_df_train=transactions_train.groupby(['client_id','small_group'])['amount_rur'].count()
cat_counts_train=counter_df_train.reset_index().pivot(index='client_id', columns='small_group',values='amount_rur')
cat_counts_train=cat_counts_train.fillna(0)
cat_counts_train.columns=['small_group_'+str(i) for i in cat_counts_train.columns]
cat_counts_train.head()
train=pd.merge(train_target,agg_features,on='client_id')
train=pd.merge(train,cat_counts_train.reset_index(),on='client_id') 
train.head()
transactions_test=pd.read_csv('../input/transactions_test.csv')



test_id=pd.read_csv('../input/test.csv')
agg_features_test=transactions_test.groupby('client_id')['amount_rur'].agg(['sum','mean','std','min','max','count','mad','var','sem','skew','quantile']).reset_index() 
counter_df_test=transactions_test.groupby(['client_id','small_group'])['amount_rur'].count()
cat_counts_test=counter_df_test.reset_index().pivot(index='client_id', columns='small_group',values='amount_rur')
cat_counts_test=cat_counts_test.fillna(0)
cat_counts_test.columns=['small_group_'+str(i) for i in cat_counts_test.columns]
test=pd.merge(test_id,agg_features_test,on='client_id')
test=pd.merge(test,cat_counts_test.reset_index(),on='client_id')
train.index = train.client_id

test.index = test.client_id
tr = train.copy()

tt = test.copy() 



trt = train.copy()

ttt = test.copy() 
tr['trans_date'] = transactions_train['trans_date'] 

tt['trans_date'] = transactions_test['trans_date'] 
tr['day_of_week'] = transactions_train['trans_date'] % 7

tt['day_of_week'] = transactions_test['trans_date'] % 7 
for i in tqdm(range(1, 26)):

    tr["mode_{}".format(i)] = tr.filter(like='small_group').apply(lambda x: Counter(dict(zip(x.index, x.values))).most_common(i)[i - 1][0], axis = 1)

    

    a = []

    for j in tr["mode_{}".format(i)].index:

        a.append(float(tr["mode_{}".format(i)][j][12:])) 

    

    tr["mode_{}".format(i)] = a



for i in tqdm(range(1, 26)):

    tt["mode_{}".format(i)] = tt.filter(like='small_group').apply(lambda x: Counter(dict(zip(x.index, x.values))).most_common(i)[i - 1][0], axis = 1)

    

    a = []

    for j in tt["mode_{}".format(i)].index:

        a.append(float(tt["mode_{}".format(i)][j][12:]))    

    

    tt["mode_{}".format(i)] = a
for i in tr.filter(like='small_group'):

    tr[i] = tr[i] / cat_counts_train.sum(axis = 1, skipna=True)



for i in tt.filter(like='small_group'):

    tt[i] = tt[i] / cat_counts_test.sum(axis = 1, skipna=True)
tr['min*max'] = tr['min'] * tr['max'] 

tr['min+max'] = tr['min'] + tr['max'] 

tr['max-min'] = tr['max'] - tr['min'] 

tr['min-max'] = tr['min'] - tr['max'] 

tr['sum-mami'] = tr['sum'] - (tr['min'] + tr['max']) 

tr['sum/mean'] = tr['sum'] / tr['mean'] 

tr['mean/sum'] = tr['mean'] / tr['sum'] 

tr['std*std'] = tr['std'] ** 2 

tr['mean-min'] = tr['mean'] - tr['min']

tr['HasGotMoney'] = tr['sum'] > 30000



tt['min*max'] = tt['min'] * tt['max'] 

tt['min+max'] = tt['min'] + tt['max'] 

tt['max-min'] = tt['max'] - tt['min'] 

tt['min-max'] = tt['min'] - tt['max'] 

tt['sum-mami'] = tt['sum'] - (tt['min'] + tt['max']) 

tt['sum/mean'] = tt['sum'] / tt['mean'] 

tt['mean/sum'] = tt['mean'] / tt['sum'] 

tt['std*std'] = tt['std'] ** 2 

tt['mean-min'] = tt['mean'] - tt['min'] 

tt['HasGotMoney'] = tt['sum'] > 30000
tr = tr.drop("client_id", axis = 1).reset_index()

tt = tt.drop("client_id", axis = 1).reset_index()
trr = tr.filter(like='small_group')

trr['client_id'] = tr['client_id']



trr = trr.groupby('client_id').agg(['sum','mean','min','max','count','quantile']).reset_index() 



ttt = tt.filter(like='small_group')

ttt['client_id'] = tt['client_id']



ttt = ttt.groupby('client_id').agg(['sum','mean','min','max','count','quantile']).reset_index() 
tr = pd.merge(tr, trr, on='client_id')

tt = pd.merge(tt, ttt, on='client_id')
tr["small_group_sum"] = cat_counts_train.sum(axis = 1, skipna=True)

tt["small_group_sum"] = cat_counts_test.sum(axis = 1, skipna=True) 
tr["small_group_mean"] = cat_counts_train.mean(axis = 1, skipna=True)

tt["small_group_mean"] = cat_counts_test.mean(axis = 1, skipna=True) 
tr.index = tr['client_id']

tt.index = tt['client_id']
try:

    tr = tr.drop('client_id', axis = 1) 

    tt = tt.drop('client_id', axis = 1) 

except:

    pass
def convert_columns(df):

    for col in df.columns:

        col_type = df[col].dtype

        col_items = df[col].count()

        col_unique_itmes = df[col].nunique()

        

        if (col_type == 'object') and (col_unique_itmes < col_items):

                df[col] = df[col].astype('category')

        

        if (col_type == 'int64'):

                df[col] = df[col].astype('int32')

        

        if (col_type == 'float64'):

                df[col] = df[col].astype('float32')

        

    return df



tr = convert_columns(tr) 

tt = convert_columns(tt) 
# common_features=list(set(train.columns).intersection(set(test.columns))) 

common_features=list(set(tr.columns).intersection(set(tt.columns))) 
# y_train=train['bins']

# X_train=train[common_features]

# test=test[common_features]



y_train=tr['bins']

X_train=tr[common_features]

test=tt[common_features] 
X, X_test, y, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state = 6741) 
%%time

# model=xgb.XGBClassifier(**param, n_estimators=850)

# model.fit(X_train,y_train)



# model = CatBoostClassifier(iterations = 2500, task_type = "GPU", devices = '0:1', verbose = 0, eval_metric = 'Accuracy')

# model.fit(X, y, plot = True, eval_set = (X_test, y_test), use_best_model = True) 



model = CatBoostClassifier(eval_metric = "Accuracy", loss_function = 'MultiClass', n_estimators = 20000, learning_rate = 0.02, task_type = 'GPU', verbose = 0, random_state = 6741)  

model.fit(X, y, plot = True, eval_set = (X_test, y_test), use_best_model = True)
feature_score = pd.DataFrame(list(zip(X.dtypes.index, model.get_feature_importance())), columns = ['Feature','Score']) 

feature_score = feature_score.sort_values(by = 'Score', ascending = False, inplace = False, kind = 'quicksort', na_position = 'last')

feature_score = feature_score.reset_index(drop = True)

feature_score.to_csv("importance.csv", index = False) 
feature_score.head(60) 
plt.rcParams["figure.figsize"] = (100, 15)

ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')

ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)

ax.set_xlabel('')



rects = ax.patches



labels = feature_score['Score'].round(2)



for rect, label in zip(rects, labels):

    height = rect.get_height() 

    ax.text(rect.get_x() + rect.get_width()/2, height + 0.1, label, ha='center', va='bottom')



plt.show()
# pred = model.predict(test).astype(int) 

# pred = pred.tolist() 

# pred = [x[0] for x in pred] 



pred = model.predict(tt).astype(int).flatten()
submission = pd.DataFrame({'bins': pred, 'client_id': test_id.client_id}) 
import time

import os



current_timestamp = int(time.time())

submission_path = 'submissions/{}.csv'.format(current_timestamp)



if not os.path.exists('submissions'):

    os.makedirs('submissions')



print(submission_path)

submission.to_csv(submission_path, index=False)