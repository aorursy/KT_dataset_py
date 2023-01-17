import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd

matrix = pd.read_csv("../input/mean-encodings-predict-future-sales/mean_encodings.csv")

def downcastColumns(df):

    for column in df.columns.values:

        if df[column].dtype == 'float64' or df[column].dtype == 'int64':

            df[column] = df[column].astype('float32')
downcastColumns(matrix)
def expandingMeanEncodings(train,train_new,column,target):

    cumsum = train.groupby(column)[target].cumsum() - train[target]

    cumcnt = train.groupby(column).cumcount()



    train_new[column+'_target_enc'] = cumsum / cumcnt

    train_new[column+'_target_enc'].fillna(train[target].mean(), inplace=True) 
def calculateMapMeanEncodings(train,val,train_new,val_new,column,target):

    target_mean = train.groupby(column)[target].mean()

    train_new[column+'_target_enc'] = train_new[column].map(target_mean)

    val_new[column+'_target_enc'] = val_new[column].map(target_mean)

    val_new[column+'_target_enc'].fillna(train[target].mean(), inplace=True) 

from xgboost import XGBRegressor

from xgboost import plot_importance

from tqdm import tqdm

from sklearn.metrics import mean_squared_error

import statistics, gc, copy

import numpy as np

scores = []



#initilize variables

target_col = 'item_cnt_month'

features_to_consider = ['item_id','item_category_id','type_code','city_code','subtype_code','shop_id']



for cur_block_num in tqdm([32,33]):

    

    print(cur_block_num,"current block number")

    #split data into train and val

    train = matrix[matrix['date_block_num'] <cur_block_num]

    val = matrix[matrix['date_block_num'] == cur_block_num]

    train_new = copy.deepcopy(train)

    val_new = copy.deepcopy(val)

    

    

    

    for feature in features_to_consider:

        #calculate mean encodings on train and map to val and train

        calculateMapMeanEncodings(train,val,train_new,val_new, feature,target_col)

    

        #expanding mean encodings on train

        expandingMeanEncodings(train,train_new,feature,target_col)

        

    

    #validate model

    _X_train = train_new[[column for column in train_new.columns.values if column not in ['date_block_num','type_code','item_category_id','item_id','city_code',

                                                                                          'subtype_code','shop_id','item_cnt_month']]]

    _X_val = val_new[[column for column in train_new.columns.values if column not in ['type_code','item_category_id','item_id','city_code','subtype_code',

                                                                                      'shop_id','item_cnt_month','date_block_num']]]

    _y_train = train_new[target_col].values

    _y_val = val_new[target_col].values

    

    

    model = XGBRegressor(

        max_depth=12,

        n_estimators=25,

        min_child_weight=40, 

        colsample_bytree=0.5, 

        subsample=0.7, 

        eta=0.1,    

        n_jobs = -1,

    seed=42,tree_method="approx")



    model.fit(_X_train, _y_train, eval_metric="rmse", eval_set=[(_X_train, _y_train), (_X_val, _y_val)], verbose=True, 

            early_stopping_rounds = 10)

    val_pred_ = model.predict(_X_val)

    train_pred_ = model.predict(_X_train)

    print('Validation root mean square for XGBoost is %f' % mean_squared_error(_y_val, val_pred_,squared=False))

    print('Train root mean square for XGBoost is %f' % mean_squared_error(_y_train, train_pred_,squared=False))

    scores.append(mean_squared_error(_y_val, val_pred_,squared=False))

    

    

    del train, val, train_new, val_new, _X_train, _X_val, _y_train, _y_val

    gc.collect()

    

print("mean score for cross validation is %f " % statistics.mean(scores))

print("worst of scores for cross validation is %f " % max(scores))

print("best of scores for cross validation is %f " % min(scores))



def expandingMeanEncodings(train,column,target,new_column):

    cumsum = train.groupby(column)[target].cumsum() - train[target]

    cumcnt = train.groupby(column).cumcount()



    train[new_column] = cumsum / cumcnt

    train[new_column].fillna(train[target].mean(), inplace=True) 
def calculateMapMeanEncodings(train,val,column,target,new_column):

    target_mean = train.groupby(column)[target].mean()

    train[new_column] = train[column].map(target_mean)

    val[new_column] = val[column].map(target_mean)

    val[new_column].fillna(train[target].mean(), inplace=True) 

import sys

def kFoldMeanEncodings(train,column,target,new_column):

    kf = KFold(n_splits = 5, shuffle = False)

    train[new_column] = train[target].mean()

    for tr_ind, val_ind in kf.split(train):

        X_tr = train.iloc[tr_ind].copy()

        X_val = train.iloc[val_ind].copy()

        means = X_val[column].map(X_tr.groupby(column)[target].mean())

        X_val[new_column] = means

        train.iloc[val_ind] = X_val

        del X_tr,X_val

        gc.collect()

    train[new_column].fillna(train[target].mean(), inplace = True)

        
def looMeanEncodings(train,column,target,new_column):

    target_sum = train.groupby(column)[target].sum()

    target_count = train.groupby(column)[target].count()

    train[column + target+'_sum'] = train[column].map(target_sum)

    train[column + target+'_count'] = train[column].map(target_count)



    train[new_column] = (train[column + target+'_sum'] - train[target]) / (train[column + target+'_count'] - 1)

    train[new_column].fillna(train[target].mean(), inplace = True)

def smoothMeanEncodings(train,column,target,new_column,alpha):

    target_mean = train.groupby(column)[target].mean()

    target_count = train.groupby(column)[target].count()



    train[column + target+'_mean'] = train[column].map(target_mean)

    train[column + target+'_count'] = train[column].map(target_count)



    train[new_column] = (train[column + target+'_mean'] *  train[column + target+'_count'] + train[target].mean() * alpha) / (alpha + train[column + target+'_count'])

    train[new_column].fillna(train[target].mean(), inplace=True)

from xgboost import XGBRegressor

from sklearn.model_selection import KFold

from xgboost import plot_importance

from tqdm import tqdm

import time

from sklearn.metrics import mean_squared_error

import statistics, gc, sys

import numpy as np

scores = {}



#initilize variables

target_col = 'item_cnt_month'

features_to_consider = ['item_id','item_category_id','type_code','city_code','subtype_code','shop_id']

corrcoefs = pd.DataFrame(columns = ['Cor'])



for cur_block_num in tqdm([33]):

    start_time = time.time()

    print('---Start applying mean-encodings---')

    #split data into train and val

    train = matrix[matrix['date_block_num'] <cur_block_num].copy(deep=True)

    val = matrix[matrix['date_block_num'] == cur_block_num].copy(deep=True)

    

    

    for feature in features_to_consider:

        #calculate mean encodings on train and map to val and train

        calculateMapMeanEncodings(train,val,feature,target_col,feature+'_expanding_target_enc')

        #1. expanding mean encodings on train

        expandingMeanEncodings(train,feature,target_col,feature+'_expanding_target_enc')

        corrcoefs.loc[feature+'_expanding_target_enc'] = np.corrcoef(train[target_col],train[feature+'_expanding_target_enc'])[0][1]

        

    

        #calculate mean encodings on train and map to val and train

        calculateMapMeanEncodings(train,val, feature,target_col,feature+'_kfold_target_enc')

        #2. expanding mean encodings on train

        kFoldMeanEncodings(train,feature,target_col,feature+'_kfold_target_enc')

        corrcoefs.loc[feature+'_kfold_target_enc'] = np.corrcoef(train[target_col],train[feature+'_kfold_target_enc'])[0][1]

        

        #calculate mean encodings on train and map to val and train

        calculateMapMeanEncodings(train,val, feature,target_col,feature+'_loo_target_enc')

        #3. leave one out mean encodings on train

        looMeanEncodings(train,feature,target_col,feature+'_loo_target_enc')

        corrcoefs.loc[feature+'_loo_target_enc'] = np.corrcoef(train[target_col],train[feature+'_loo_target_enc'])[0][1]

        

        #calculate mean encodings on train and map to val and train

        calculateMapMeanEncodings(train,val, feature,target_col,feature+'_smooth_target_enc')

        #4. smooth mean encodings on train

        smoothMeanEncodings(train,feature,target_col,feature+'_smooth_target_enc',100)

        corrcoefs.loc[feature+'_smooth_target_enc'] = np.corrcoef(train[target_col], train[feature+'_smooth_target_enc'])[0][1]

    print('---Finish mean-encodings, elapsed time in min: %0.2f---'%((time.time() - start_time)/60))

    for method in ['kfold','expanding','loo','smooth']:

        print('---Start modelling using %s mean encoding---'%(method))

        start_time = time.time()

        #validate model

        _X_train = train[[column for column in train.columns.values if method in column and column not in ['date_block_num','type_code','item_category_id',

                                                                                                           'item_id','city_code','subtype_code','shop_id',

                                                                                                           'item_cnt_month']]]

        _X_val = val[[column for column in train.columns.values if method in column and column not in ['type_code','item_category_id','item_id','city_code',

                                                                                                       'subtype_code','shop_id','item_cnt_month','date_block_num']]]

        _y_train = train[target_col].values

        _y_val = val[target_col].values





        model = XGBRegressor(

            max_depth=12,

            n_estimators=25,

            min_child_weight=40, 

            colsample_bytree=0.5, 

            subsample=0.7, 

            eta=0.1,    

            n_jobs = -1,

        seed=42,tree_method="approx")



        model.fit(_X_train, _y_train, eval_metric="rmse", eval_set=[(_X_train, _y_train), (_X_val, _y_val)], verbose=True)

        val_pred_ = model.predict(_X_val)

        train_pred_ = model.predict(_X_train)

        scores[method] = {'train': model.evals_result()['validation_0']['rmse'], 

                          'test': model.evals_result()['validation_1']['rmse']}        

        del _X_train, _X_val, _y_train, _y_val

        gc.collect()

        print('---Finish modelling using %s, elapsed time in min: %0.2f---'%(method,(time.time() - start_time)/60))

    

    del train, val

    gc.collect()

    





import matplotlib.pyplot as plt





epochs = 25

x_axis = range(0, epochs)



# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, scores['kfold']['train'], label='k_fold train error')

ax.plot(x_axis, scores['expanding']['train'], label='expanding train error')

ax.plot(x_axis, scores['loo']['train'], label='loo train error')

ax.plot(x_axis, scores['smooth']['train'], label='smooth train error')

ax.legend(loc=0, prop={'size': 8})

plt.ylabel('RMSE')

plt.xlabel('Number of Epochs')

plt.title('XGBoost Performance Change')

plt.show()



x_axis = range(0, epochs)



# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, scores['kfold']['test'], label='k_fold test error')

ax.plot(x_axis, scores['expanding']['test'], label='expanding test error')

ax.plot(x_axis, scores['loo']['test'], label='loo test error')

ax.plot(x_axis, scores['smooth']['test'], label='smooth test error')

ax.legend(loc=0, prop={'size': 8})

plt.ylabel('RMSE')

plt.xlabel('Number of Epochs')

plt.title('XGBoost Performance Change')

plt.show()
corrcoefs