import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

import xgboost as xgb



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from boruta import BorutaPy

from sklearn.feature_selection import RFE

import sys
train = pd.read_csv('../input/bigcon2019083101/Train.csv')

test1 = pd.read_csv('../input/bigcon2019083101/Test1.csv')

test2 = pd.read_csv('../input/bigcon2019083101/Test2.csv')
validation_acc = pd.read_csv('../input/bigcontest2019/train_valid_user_id.csv')
train_churn = train.loc[train['survival_time'] != 64]

train_churn.head()
temp_train = train.drop(columns = ['acc_id','amount_spent','survival_time'],axis =1)

temp_target = train.survival_time



gbm = lgb.LGBMRegressor()

gbm.fit(temp_train,temp_target)

gbm.booster_.feature_importance()



fea_imp_ = pd.DataFrame({'cols':temp_train.columns, 'fea_imp':gbm.feature_importances_})

fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by = ['fea_imp'], ascending = False).head(10)
fig, ax = plt.subplots(figsize=(10,10))

lgb.plot_importance(gbm,ax=ax)

plt.show()
rf_params = {'n_estimators' : 1000,

            'max_depth' : 5}



rf = RandomForestRegressor(**rf_params,random_state = 123)

rf.fit(temp_train,temp_target)
feature_imp = pd.DataFrame({'cols':temp_train.columns, 'importance':rf.feature_importances_})

feature_imp.loc[feature_imp.importance > 0].sort_values(by = ['importance'],ascending = False).head(10)
selector = RFE(gbm,18,step = 1) #####

selector = selector.fit(temp_train,temp_target)



print(selector.support_)



fea_rank_ = pd.DataFrame({'cols':temp_train.columns,'fea_rank':selector.ranking_ })

fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by = ['fea_rank'], ascending = True).head(10)
survival_col=list(fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by = ['fea_rank'], ascending = True).head(18)['cols'])+['acc_id','amount_spent','survival_time']

survival_col_non_target = list(fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by = ['fea_rank'], ascending = True).head(18)['cols'])
temp_target = train_churn.amount_spent

temp_train = train_churn.drop(columns = ['acc_id','amount_spent','survival_time'],axis =1)



gbm = lgb.LGBMRegressor()

gbm.fit(temp_train,temp_target)



fea_imp_ = pd.DataFrame({'cols':temp_train.columns, 'fea_imp':gbm.feature_importances_})

fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by = ['fea_imp'], ascending = False).head(10)
rf = RandomForestRegressor(**rf_params,random_state = 123)

rf.fit(temp_train,temp_target)
feature_imp = pd.DataFrame({'cols':temp_train.columns,'importance':rf.feature_importances_})

feature_imp.loc[feature_imp.importance > 0].sort_values(by = ['importance']).head(10)
selector = RFE(gbm,18,step = 1)

selector = selector.fit(temp_train,temp_target)



print(selector.support_)



fea_rank_ = pd.DataFrame({'cols':temp_train.columns,'fea_rank':selector.ranking_ })

fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by = ['fea_rank'], ascending = True).head(10)
amount_spent_col=list(fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by = ['fea_rank'], ascending = True).head(18)['cols'])+['acc_id','amount_spent','survival_time']

amount_spent_col_non_target=list(fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by = ['fea_rank'], ascending = True).head(18)['cols'])
survival_feature_selection_train = train_churn[survival_col]

amount_feature_selection_train = train_churn[amount_spent_col]
def survival64(y_pred, dataset):

    y_true = dataset.get_label()

    y_pred = np.array([64 if x > 64 else x for x in y_pred])

    y_pred = np.array([0 if x < 0 else x for x in y_pred])

    y_pred = np.round(y_pred)

    error = np.sqrt(mean_squared_error(y_true, y_pred))

    return 'error', error, False
train_index = validation_acc[validation_acc.set == 'Train'].acc_id

valid_index = validation_acc[validation_acc.set == 'Validation'].acc_id



survival_train_set = survival_feature_selection_train[train['acc_id'].isin(train_index)]

survival_valid_set = survival_feature_selection_train[train['acc_id'].isin(valid_index)]



amount_train_set = amount_feature_selection_train[train['acc_id'].isin(train_index)]

amount_valid_set = amount_feature_selection_train[train['acc_id'].isin(valid_index)]



print('Survival Train set:',survival_train_set.shape)

print('Survival Valid set:',survival_valid_set.shape)



print('Amount Spent Train set:',amount_train_set.shape)

print('Amount Spent Valid set:',amount_valid_set.shape)
lgb_params = {'n_estimators': 1000,

             'seed': 123}



lgb_train_amount = lgb.Dataset(amount_train_set.drop(['acc_id','amount_spent','survival_time'],axis = 1),

                              amount_train_set.amount_spent)

lgb_train_survival = lgb.Dataset(survival_train_set.drop(['acc_id','amount_spent','survival_time'],axis = 1),

                                survival_train_set.survival_time)
lgb_amount = lgb.train(lgb_params,

                      lgb_train_amount,

                      feval = survival64,

                      valid_sets = [lgb_train_amount],

                      verbose_eval=100)



lgb_amount_pred = lgb_amount.predict(amount_valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lgb_amount_pred = pd.Series(lgb_amount_pred).apply(lambda x: 0 if x < 0 else x)
lgb_survival = lgb.train(lgb_params,

                        lgb_train_survival,

                        feval =survival64,

                        valid_sets = [lgb_train_survival],

                        verbose_eval = 100)



lgb_survival_pred = lgb_survival.predict(survival_valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

lgb_survival_pred = pd.Series(lgb_survival_pred).apply(lambda x: 64 if x > 64 else x)

lgb_survival_pred = lgb_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
lgb_pred_df = pd.DataFrame({'acc_id':survival_valid_set.acc_id.values,

                           'survival_time':lgb_survival_pred,

                           'amount_spent':lgb_amount_pred})

print('lgb_pred_df shape: ',lgb_pred_df.shape)
rf_params = {

    'n_estimators':1000,

    'max_depth':5,

}
rf_amount = RandomForestRegressor(**rf_params)

rf_amount.fit(amount_train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

              amount_train_set.amount_spent)

rf_amount_pred = rf_amount.predict(amount_valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_amount_pred = pd.Series(rf_amount_pred).apply(lambda x: 0 if x < 0 else x)
rf_survival = RandomForestRegressor(**rf_params)

rf_survival.fit(survival_train_set.drop(['acc_id','amount_spent','survival_time'], axis=1),

                survival_train_set.survival_time)

rf_survival_pred = rf_survival.predict(survival_valid_set.drop(['acc_id','amount_spent','survival_time'], axis=1))

rf_survival_pred = pd.Series(rf_survival_pred).apply(lambda x: 64 if x > 64 else x)

rf_survival_pred = rf_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
rf_pred_df = pd.DataFrame({'acc_id':survival_valid_set.acc_id.values,

                           'survival_time':rf_survival_pred,

                           'amount_spent':rf_amount_pred})

print('rf_pred_df shape: ',rf_pred_df.shape)
rf_pred_df.describe()
amount_dtrain = xgb.DMatrix(amount_train_set.drop(['acc_id','amount_spent','survival_time'], axis = 1),label = amount_train_set.amount_spent)

survival_dtrain = xgb.DMatrix(survival_train_set.drop(['acc_id','amount_spent','survival_time'], axis = 1),label = survival_train_set.survival_time)



amount_dval = xgb.DMatrix(amount_valid_set.drop(['acc_id','amount_spent','survival_time'],axis = 1))

survival_dval = xgb.DMatrix(survival_valid_set.drop(['acc_id','amount_spent','survival_time'],axis = 1))
xgb_params_amount={'eta':0.01,

            'max_depth':3,

            'objective':'reg:squarederror',

            'eval_metric':'mae',

            'min_child_samples':1}



num_round = 10000
bst = xgb.cv(xgb_params, amount_dtrain, num_round,early_stopping_rounds = 100, nfold = 5, verbose_eval = 100)

bst_round_amount = bst.index.size



print('Amount Spent Best Rounds:',bst_round_amount)
xgb_params_survival={'eta':0.01,

            'max_depth':5,

            'objective':'reg:squarederror',

            'eval_metric':'mae',

            'min_child_samples':1}
bst = xgb.cv(xgb_params_survival, survival_dtrain, num_round, early_stopping_rounds = 100, nfold = 2,verbose_eval = 100)

bst_round_survival = bst.index.size



print('Survival Time Best Rounds:',bst_round_survival)
model_survival = xgb.train(xgb_params_survival,survival_dtrain,num_boost_round = bst_round_amount)

y_pred_survival = model_survival.predict(survival_dval)
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model_survival, ax = ax)

plt.show()
model_amount = xgb.train(xgb_params_amount,amount_dtrain,num_boost_round = bst_round_amount)

y_pred_amount = model_amount.predict(amount_dval)
fig, ax = plt.subplots(figsize=(10,10))

xgb.plot_importance(model_amount, ax =ax)
xgb_pred = pd.DataFrame({'acc_id':survival_valid_set['acc_id'],

                        'survival_time':y_pred_survival,

                        'amount_spent': y_pred_amount})

xgb_pred.head()
xgb_pred['survival_time'] = xgb_pred['survival_time'].apply(lambda x : 64 if x > 64 else (1 if x < 1 else x))

xgb_pred['amount_spent'] = xgb_pred['amount_spent'].apply(lambda x : 0 if x < 0 else x)
scaler = MinMaxScaler(feature_range = [0, 39.412632])



scale_rf_pred_df = rf_pred_df.copy()

scale_lgb_pred_df = lgb_pred_df.copy()

scale_xgb_pred_df = xgb_pred.copy()



scale_rf_pred_df['amount_spent'] = scaler.fit_transform(scale_rf_pred_df['amount_spent'].values.reshape(-1,1))

scale_lgb_pred_df['amount_spent'] = scaler.fit_transform(scale_lgb_pred_df['amount_spent'].values.reshape(-1,1))

scale_xgb_pred_df['amount_spent'] = scaler.fit_transform(scale_xgb_pred_df['amount_spent'].values.reshape(-1,1))
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

            

        score = sum(profit_result)                                         #기대이익 총합 계산

    return score
lgb_valid_score = score_function(lgb_pred_df, survival_valid_set[['acc_id','survival_time','amount_spent']])

rf_valid_score = score_function(rf_pred_df, survival_valid_set[['acc_id','survival_time','amount_spent']])

xgb_valid_score = score_function(xgb_pred,survival_valid_set[['acc_id','survival_time','amount_spent']])



scale_lgb_valid_score = score_function(scale_lgb_pred_df, survival_valid_set[['acc_id','survival_time','amount_spent']])

scale_rf_valid_score = score_function(scale_rf_pred_df, survival_valid_set[['acc_id','survival_time','amount_spent']])

scale_xgb_valid_score = score_function(scale_xgb_pred_df, survival_valid_set[['acc_id','survival_time','amount_spent']])

true_score = score_function(amount_valid_set[['acc_id','survival_time','amount_spent']],

                            survival_valid_set[['acc_id','survival_time','amount_spent']])



print("Random Froest score: ", rf_valid_score)

print('Light GBM score: ',lgb_valid_score)

print('XGB score:', xgb_valid_score)

print("Scale Random Forest score",scale_rf_valid_score)

print("Scale XGB scroe",scale_xgb_valid_score)

print("Scale Light GBM score",scale_lgb_valid_score)
test1.rename(columns = {'get_ect':'get_etc'},inplace= True)

test2.rename(columns = {'get_ect':'get_etc'},inplace= True)
amount_train_churn = train_churn[amount_spent_col]

survival_train_churn = train_churn[survival_col]



amount_test1 = test1[amount_spent_col_non_target]

survival_test1 = test1[survival_col_non_target]
rf_params = {

    'n_estimators':1000,

    'max_depth':5,

}
rf_amount = RandomForestRegressor(**rf_params)

rf_amount.fit(amount_train_churn.drop(['acc_id','amount_spent','survival_time'], axis=1),

              amount_train_churn.amount_spent)

rf_amount_pred = rf_amount.predict(amount_test1)

rf_amount_pred = pd.Series(rf_amount_pred).apply(lambda x: 0 if x < 0 else x)
rf_survival = RandomForestRegressor(**rf_params)

rf_survival.fit(amount_train_churn.drop(['acc_id','amount_spent','survival_time'], axis=1),

                amount_train_churn.survival_time)

rf_survival_pred = rf_survival.predict(amount_test1)

rf_survival_pred = pd.Series(rf_survival_pred).apply(lambda x: 64 if x > 64 else x)

rf_survival_pred = rf_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
rf_pred_df = pd.DataFrame({'acc_id':test1.acc_id.values,

                           'survival_time':rf_survival_pred,

                           'amount_spent':rf_amount_pred})

print('rf_pred_df shape: ',rf_pred_df.shape)
scaler = MinMaxScaler(feature_range = [0, 39.412632])



rf_pred_df['amount_spent'] = scaler.fit_transform(rf_pred_df['amount_spent'].values.reshape(-1,1))
rf_pred_df.to_csv('test1_predict.csv', index = False) 
amount_test2 = test2[amount_spent_col_non_target]

survival_test2 = test2[survival_col_non_target]
rf_amount = RandomForestRegressor(**rf_params)

rf_amount.fit(amount_train_churn.drop(['acc_id','amount_spent','survival_time'], axis=1),

              amount_train_churn.amount_spent)

rf_amount_pred = rf_amount.predict(amount_test2)

rf_amount_pred = pd.Series(rf_amount_pred).apply(lambda x: 0 if x < 0 else x)
rf_survival = RandomForestRegressor(**rf_params)

rf_survival.fit(amount_train_churn.drop(['acc_id','amount_spent','survival_time'], axis=1),

                amount_train_churn.survival_time)

rf_survival_pred = rf_survival.predict(amount_test2)

rf_survival_pred = pd.Series(rf_survival_pred).apply(lambda x: 64 if x > 64 else x)

rf_survival_pred = rf_survival_pred.apply(lambda x: 0 if x < 0 else x).round()
rf_pred_df = pd.DataFrame({'acc_id':test2.acc_id.values,

                           'survival_time':rf_survival_pred,

                           'amount_spent':rf_amount_pred})

print('rf_pred_df shape: ',rf_pred_df.shape)
scaler = MinMaxScaler(feature_range = [0, 39.412632])



rf_pred_df['amount_spent'] = scaler.fit_transform(rf_pred_df['amount_spent'].values.reshape(-1,1))
rf_pred_df.to_csv('test2_predict.csv', index = False)