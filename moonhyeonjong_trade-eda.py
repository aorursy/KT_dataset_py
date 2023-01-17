import os



import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns



matplotlib.rcParams['axes.unicode_minus'] = False # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처

matplotlib.rcParams["font.family"] = "Malgun Gothic" 



import warnings

warnings.filterwarnings('ignore')
train_trade = pd.read_csv('../input/train_trade.csv')

test1_trade = pd.read_csv('../input/test1_trade.csv')

test2_trade = pd.read_csv('../input/test2_trade.csv')



label = pd.read_csv('../input/train_label.csv')
print('train_trade.shape :', train_trade.shape)

print('test1_trade.shape :', test1_trade.shape)

print('test2_trade.shape :', test2_trade.shape)

print()

print('label.shape :', label.shape)
train_trade['data'] = 'train'

test1_trade['data'] = 'test1'

test2_trade['data'] = 'test2'
trade = pd.concat([train_trade, test1_trade, test2_trade], axis = 0)
trade.isnull().sum()
trade.loc[trade['item_price'].isnull()].head()
trade.loc[trade['item_price'].isnull()]['type'].value_counts()
print('trade.day.nunique() : ', trade.day.nunique())

print('trade.day.unique() : ', trade.day.unique())
print('trade.time.nunique() : ', trade.time.nunique())

print('trade.time.unique() : ', trade.time.unique())
trade['hour'] = trade['time'].str.split(':', expand = True)[0]
print('trade.hour.nunique() : ', trade.hour.nunique())

print('trade.hour.unique() : ', trade.hour.unique())
print('trade.server.nunique() : ', trade.server.nunique())

print('trade.server.unique() : ', trade.server.unique())
print('trade.type.nunique() : ', trade.type.nunique())

print('trade.type.unique() : ', trade.type.unique())
print('trade.item_type.nunique() : ', trade.item_type.nunique())

print('trade.item_type.unique() : ', trade.item_type.unique())
plt.figure(figsize= (15,4))

sns.kdeplot(trade['item_amount'], shade = True)
trade['item_amount'].describe(percentiles= [0.8, 0.9, 0.95, 0.99]).round(2)
trade.columns
category = ['server', 'day', 'hour', 'type', 'item_type']

continous = ['item_amount', 'item_price']
m = [0,0,1,1,2,2]

n = [0,1,0,1,0,1]



f, ax = plt.subplots(3, 2,figsize = (18,12))

for m, n, feature in zip(m, n, category):

    sns.countplot(x = feature, data = trade, ax = ax[m][n])

    ax[m][n].set_title('Trade_{} countplot'.format(feature), fontsize = 15)

plt.xticks(rotation = 45, ha = "right")

plt.tight_layout()
m = [0,0,1,1,2,2]

n = [0,1,0,1,0,1]



f, ax = plt.subplots(3, 2,figsize = (18,12))

for m, n, feature in zip(m, n, category):

    sns.pointplot(x = feature, y = 'item_amount', data = trade, ax = ax[m][n], color='blue')

    sns.pointplot(x = feature, y = 'item_price', data = trade, ax = ax[m][n], color='red')

    ax[m][n].set_title('pointplot of Trade_{} \n blue : item_amount, red : item_price'.format(feature), fontsize = 15)

plt.xticks(rotation = 45, ha = "right")

plt.tight_layout()
print('source_acc_id 의 수 : ', trade['source_acc_id'].nunique())

print('source_char_id 의 수 : ', trade['source_char_id'].nunique())

print('target_acc_id 의 수 : ', trade['target_acc_id'].nunique())

print('targer_char_id 의 수 : ', trade['target_char_id'].nunique())
trade_buy = trade.copy()

trade_sell = trade.copy()



trade_buy['trade_type'] = 'buy'

trade_sell['trade_type'] = 'sell'



trade_buy.drop(['source_acc_id', 'source_char_id'], axis = 1, inplace = True)

trade_sell.drop(['target_acc_id', 'target_char_id'], axis = 1, inplace = True)



trade_buy = trade_buy.rename(columns = {'target_acc_id' : 'acc_id' , 'target_char_id' : 'char_id'})

trade_sell = trade_sell.rename(columns = {'source_acc_id' : 'acc_id', 'source_char_id' : 'char_id'})



total_trade = pd.concat([trade_buy, trade_sell], axis = 0)
total_trade.drop(['time'], axis = 1, inplace = True)
data = pd.merge(label, total_trade, on = 'acc_id', how = 'left')
data['isSurvival'] = data['survival_time'].apply(lambda x : 1 if x == 64 else 0)
data.head()
data.shape
print((data.isnull().sum() / len(data)))
category = ['server', 'day', 'hour', 'trade_type', 'type', 'item_type']



continous = ['item_amount', 'item_price']



target= ['survival_time', 'amount_spent', 'isSurvival']
data.loc[data['item_amount'] > 3.9, 'isSurvival']
plt.figure(figsize= (5,5))



data.loc[data['item_amount'] > 3.9, 'isSurvival'].value_counts().plot.pie(autopct = '%1.1f')
m = [0,0,1,1,2,2]

n = [0,1,0,1,0,1]



f, ax = plt.subplots(3, 2,figsize = (18,12))

for m, n, feature in zip(m, n, category):

    sns.countplot(x = feature, data = data, ax = ax[m][n], hue = 'isSurvival')

    ax[m][n].set_title('countplot of Trade_{} by isSurvival '.format(feature), fontsize = 15)

plt.xticks(rotation = 45, ha = "right")

plt.tight_layout()
m = [0,0,1,1,2,2]

n = [0,1,0,1,0,1]



f, ax = plt.subplots(3, 2,figsize = (18,12))

for m, n, feature in zip(m, n, category):

    sns.pointplot(x = feature, y = 'item_amount', data = data, ax = ax[m][n], hue = 'isSurvival')

    ax[m][n].set_title('pointplot of Trade_{} by isSurvival '.format(feature), fontsize = 15)

plt.xticks(rotation = 45, ha = "right")

plt.tight_layout()
data[['survival_time', 'amount_spent','item_amount', 'item_price', 'isSurvival']].describe().round(3)
data.loc[data['server'] == 'bg'][['survival_time', 'amount_spent','item_amount', 'item_price', 'isSurvival']].describe().round(3)
m = [0,0,1,1,2,2]

n = [0,1,0,1,0,1]



f, ax = plt.subplots(3, 2,figsize = (18,12))

for m, n, feature in zip(m, n, category):

    sns.pointplot(x = feature, y = 'item_price', data = data, ax = ax[m][n], hue = 'isSurvival')

    ax[m][n].set_title('pointplot of Trade_{} by isSurvival'.format(feature), fontsize = 15)

plt.xticks(rotation = 45, ha = "right")

plt.tight_layout()
f, ax = plt.subplots(1, 2,figsize = (16,4))



sns.kdeplot(data.loc[data['isSurvival'] == 1, 'item_amount'], ax = ax[0], label = '생존')

sns.kdeplot(data.loc[data['isSurvival'] == 0, 'item_amount'], ax = ax[0], label = '이탈')



sns.kdeplot(data.loc[data['isSurvival'] == 1, 'item_price'], ax = ax[1], label = '생존')

sns.kdeplot(data.loc[data['isSurvival'] == 0, 'item_price'], ax = ax[1], label = '이탈')



ax[0].set_title('distribution of Trade_item_amount by isSurvival', fontsize = 15)

ax[1].set_title('distribution of Trade_item_price by isSurvival', fontsize = 15)



plt.xticks(rotation = 45, ha = "right")

plt.tight_layout()

plt.legend()
# model

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb



# evaluation

from sklearn.metrics import mean_squared_error
validation_acc = pd.read_csv('../input/train_valid_user_id.csv')

print('validation_acc shape: ',validation_acc.shape)
data_gr = data.groupby(['acc_id'])['survival_time', 'amount_spent','item_amount', 'item_price'].mean().reset_index()
data_gr.shape
seed = 223
train_idx = validation_acc[validation_acc.set=='Train'].acc_id

valid_idx = validation_acc[validation_acc.set=='Validation'].acc_id



train_set = data_gr[data_gr.acc_id.isin(train_idx)]

valid_set = data_gr[data_gr.acc_id.isin(valid_idx)]



print('train set: ',train_set.shape)

print('valid set: ',valid_set.shape)
train_set.fillna(0, inplace = True)

valid_set.fillna(0, inplace = True)
def survival64(y_pred, dataset):

    y_true = dataset.get_label()

    y_pred = np.array([64 if x > 64 else x for x in y_pred])

    y_pred = np.array([0 if x < 0 else x for x in y_pred])

    y_pred = np.round(y_pred)

    error = np.sqrt(mean_squared_error(y_true, y_pred))

    return 'error', error, False
lr_amount = RandomForestRegressor()

lr_amount.fit(train_set.drop(['acc_id','amount_spent','survival_time'], axis=1), train_set.amount_spent)

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

    'n_jobs':-1

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