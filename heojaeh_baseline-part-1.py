# preprocessing

import numpy as np

import pandas as pd 



# graph

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



# model

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb



# evaluation

from sklearn.metrics import mean_squared_error



# utils

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
seed = 223
train_act = pd.read_csv('/kaggle/input/train_activity.csv')

test1_act = pd.read_csv('/kaggle/input/test1_activity.csv')

test2_act = pd.read_csv('/kaggle/input/test2_activity.csv')

train_pay = pd.read_csv('/kaggle/input/train_payment.csv')

test1_pay = pd.read_csv('/kaggle/input/test1_payment.csv')

test2_pay = pd.read_csv('/kaggle/input/test2_payment.csv')

print('train activity shape: ',train_act.shape)

print('test1 activity shape: ',test1_act.shape)

print('test2 activity shape: ',test2_act.shape)

print('train payment shape: ',train_pay.shape)

print('test1 payment shape: ',test1_pay.shape)

print('test2 payment shape: ',test2_pay.shape)
train_label = pd.read_csv('/kaggle/input/train_label.csv')

print('train_label shape: ',train_label.shape)
validation_acc = pd.read_csv('/kaggle/input/train_valid_user_id.csv')

print('validation_acc shape: ',validation_acc.shape)
train = train_act.groupby(['acc_id','char_id']).day.count().reset_index().groupby('acc_id').agg({'char_id':'count','day':'max'})

train = pd.merge(train, train_label, on='acc_id', how='inner')

print('train shape: ',train.shape)
amount_by_acc_id = train_pay.groupby('acc_id').agg({'amount_spent':['max','median']})

amount_by_acc_id.columns = ['max_amount','median_amount']

train = pd.merge(train, amount_by_acc_id, on='acc_id', how='left')

train = train.fillna(0)

print('train shape: ',train.shape)
train_idx = validation_acc[validation_acc.set=='Train'].acc_id

valid_idx = validation_acc[validation_acc.set=='Validation'].acc_id



train_set = train[train.acc_id.isin(train_idx)]

valid_set = train[train.acc_id.isin(valid_idx)]



print('train set: ',train_set.shape)

print('valid set: ',valid_set.shape)
def survival64(y_pred, dataset):

    y_true = dataset.get_label()

    y_pred = np.array([64 if x > 64 else x for x in y_pred])

    y_pred = np.array([0 if x < 0 else x for x in y_pred])

    y_pred = np.round(y_pred)

    error = np.sqrt(mean_squared_error(y_true, y_pred))

    return 'error', error, False
lr_amount = RandomForestRegressor()

lr_amount.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

          train_set.amount_spent)

lr_amount_pred = lr_amount.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lr_amount_pred = pd.Series(lr_amount_pred).apply(lambda x: 0 if x < 0 else x)
lr_survival = RandomForestRegressor()

lr_survival.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

          train_set.survival_time)

lr_survival_pred = lr_survival.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lr_survival_pred = pd.Series(lr_survival_pred).apply(lambda x: 64 if x > 64 else x)

lr_survival_pred = lr_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
lr_pred_df = pd.DataFrame({'acc_id':valid_set.acc_id.values,

                           'survival_time':lr_survival_pred,

                           'amount_spent':lr_amount_pred})

print('lr_pred_df shape: ',lr_pred_df.shape)
rf_params = {

    'n_estimators':1000,

    'max_depth':10,

    'n_jobs':5

}
rf_amount = RandomForestRegressor(**rf_params)

rf_amount.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

              train_set.amount_spent)

rf_amount_pred = rf_amount.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_amount_pred = pd.Series(rf_amount_pred).apply(lambda x: 0 if x < 0 else x)
rf_survival = RandomForestRegressor(**rf_params)

rf_survival.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

                train_set.survival_time)

rf_survival_pred = rf_survival.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_survival_pred = pd.Series(rf_survival_pred).apply(lambda x: 64 if x > 64 else x)

rf_survival_pred = rf_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
rf_pred_df = pd.DataFrame({'acc_id':valid_set.acc_id.values,

                           'survival_time':rf_survival_pred,

                           'amount_spent':rf_amount_pred})

print('rf_pred_df shape: ',rf_pred_df.shape)
lgb_params = {

    'n_estimators':800,

    'sees':seed

}
lgb_train_amount = lgb.Dataset(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

                               train_set.amount_spent)

lgb_train_survival = lgb.Dataset(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

                                 train_set.survival_time)
lgb_amount = lgb.train(lgb_params, 

                       lgb_train_amount,

                       feval=survival64,

                       valid_sets=[lgb_train_amount],

                       verbose_eval=100)



lgb_amount_pred = lgb_amount.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lgb_amount_pred = pd.Series(lgb_amount_pred).apply(lambda x: 0 if x < 0 else x)
lgb_survival = lgb.train(lgb_params, 

                         lgb_train_survival,

                         feval=survival64,

                         valid_sets=[lgb_train_survival],

                         verbose_eval=100)



lgb_survival_pred = lgb_survival.predict(valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lgb_survival_pred = pd.Series(lgb_survival_pred).apply(lambda x: 64 if x > 64 else x)

lgb_survival_pred = lgb_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
lgb_pred_df = pd.DataFrame({'acc_id':valid_set.acc_id.values,

                           'survival_time':lgb_survival_pred,

                           'amount_spent':lgb_amount_pred})

print('lgb_pred_df shape: ',lgb_pred_df.shape)
def score_function(predict, actual):

    

    # predict = pd.read_csv(predict_label, engine='python') # 예측 답안 파일 불러오기

    # actual = pd.read_csv(actual_label,engine='python') # 실제 답안 파일 불러오기





    predict.acc_id = predict.acc_id.astype('int')

    predict = predict.sort_values(by =['acc_id'], axis = 0) # 예측 답안을 acc_id 기준으로 정렬 

    predict = predict.reset_index(drop = True)

    actual.acc_id = actual.acc_id.astype('int')

    actual = actual.sort_values(by =['acc_id'], axis = 0) # 실제 답안을 acc_id 기준으로 정렬

    actual =actual.reset_index(drop=True)

    

    if predict.acc_id.equals(actual.acc_id) == False:

        print('acc_id of predicted and actual label does not match')

        sys.exit() # 예측 답안의 acc_id와 실제 답안의 acc_id가 다른 경우 에러처리 

    else:

            

        S, alpha, L, sigma = 30, 0.01, 0.1, 15  

        cost, gamma, add_rev = 0,0,0 

        profit_result = []

        survival_time_pred = list(predict.survival_time)

        amount_spent_pred = list(predict.amount_spent)

        survival_time_actual = list(actual.survival_time)

        amount_spent_actual = list(actual.amount_spent)    

        for i in range(len(survival_time_pred)):

            if survival_time_pred[i] == 64 :                 

                cost = 0

                optimal_cost = 0

            else:

                cost = alpha * S * amount_spent_pred[i]                    #비용 계산

                optimal_cost = alpha * S * amount_spent_actual[i]          #적정비용 계산 

            

            if optimal_cost == 0:

                gamma = 0

            elif cost / optimal_cost < L:

                gamma = 0

            elif cost / optimal_cost >= 1:

                gamma = 1

            else:

                gamma = (cost)/((1-L)*optimal_cost) - L/(1-L)              #반응률 계산

            

            if survival_time_pred[i] == 64 or survival_time_actual[i] == 64:

                T_k = 0

            else:

                T_k = S * np.exp(-((survival_time_pred[i] - survival_time_actual[i])**2)/(2*(sigma)**2))    #추가 생존기간 계산

                

            add_rev = T_k * amount_spent_actual[i]                         #잔존가치 계산

    

           

            profit = gamma * add_rev - cost                                #유저별 기대이익 계산

            profit_result.append(profit)

            

        score = sum(profit_result) 

    return score
lr_valid_score = score_function(lr_pred_df, valid_set[['acc_id','survival_time','amount_spent']])

rf_valid_score = score_function(rf_pred_df, valid_set[['acc_id','survival_time','amount_spent']])

lgb_valid_score = score_function(lgb_pred_df, valid_set[['acc_id','survival_time','amount_spent']])

true_score = score_function(valid_set[['acc_id','survival_time','amount_spent']],

                            valid_set[['acc_id','survival_time','amount_spent']])



print('Linear Regression score: ',lr_valid_score)

print('Random Forest score: ',rf_valid_score)

print('Light GBM score: ',lgb_valid_score)

print('true score: ',true_score)