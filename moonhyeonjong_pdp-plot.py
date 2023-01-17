import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

plt.style.use('seaborn-bright')

import seaborn as sns



matplotlib.rcParams['axes.unicode_minus'] = False # 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처

matplotlib.rcParams["font.family"] = "Malgun Gothic" 



import warnings

warnings.filterwarnings('ignore')



import lightgbm as lgb



from sklearn.preprocessing import MinMaxScaler

seed = 42



from IPython.display import display

pd.options.display.max_columns = None
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
train = pd.read_csv('../input/bigcontest2019/preprocessing.csv')

test1 = pd.read_csv('../input/bigcontest2019/preprocessing_test1.csv')

test2 = pd.read_csv('../input/bigcontest2019/preprocessing_test2.csv')

validation_acc = pd.read_csv('../input/bigcontest2019/train_valid_user_id.csv')



print('Train set:', train.shape)

print('Test1 set:', test1.shape)

print('Test2 set:', test2.shape)

print('validation set:', validation_acc.shape)
train.head()
train_idx = validation_acc[validation_acc.set=='Train'].acc_id

valid_idx = validation_acc[validation_acc.set=='Validation'].acc_id



train_set = train[train.acc_id.isin(train_idx)]

valid_set = train[train.acc_id.isin(valid_idx)]



print('Train set:',train_set.shape)

print('Valid set:',valid_set.shape)
train_set['is_survive'] = train_set['survival_time'].apply(lambda x : 1 if x == 64 else 0)

train_set['total_amount_spent'] = train_set['survival_time'] * train_set['amount_spent']



valid_set['is_survive'] = valid_set['survival_time'].apply(lambda x: 1 if x == 64 else 0)

valid_set['total_amount_spent'] = valid_set['survival_time'] * valid_set['amount_spent']



x_train_amount = train_set.loc[train_set['total_amount_spent'] != 0].drop(columns = ['acc_id','survival_time','amount_spent','is_survive','total_amount_spent'])

x_train_survival = train_set.drop(columns = ['acc_id','survival_time','amount_spent','is_survive','total_amount_spent'])



y_train_amount = train_set.loc[train_set['total_amount_spent'] != 0].total_amount_spent

y_train_survival = train_set['is_survive']



print('Train Dataset Shape:',x_train_survival.shape)

print('Train Dataset without Total Amount Spent 0 Shape: ',x_train_survival.shape)

print('Train Target Survival Dataset Shape: ',y_train_survival.shape)

print('Train Target Total Amount Spent without 0 Shape:', y_train_amount.shape)
x_valid_survival = valid_set.drop(columns = ['acc_id','survival_time','amount_spent','is_survive','total_amount_spent'])

x_valid_amount = valid_set.loc[valid_set['total_amount_spent'] != 0].drop(columns = ['acc_id','survival_time','amount_spent','is_survive','total_amount_spent'])



y_valid_survival = valid_set.is_survive

y_valid_amount = valid_set.loc[valid_set['total_amount_spent'] != 0].total_amount_spent



print('valid Dataset Shape:',x_valid_survival.shape)

print('valid Dataset without Total Amount Spent 0 Shape: ',x_valid_survival.shape)

print('valid Target Survival Dataset Shape: ',y_valid_survival.shape)

print('valid Target Total Amount Spent without 0 Shape:', y_valid_amount.shape)
lgb_params_amount={'learning_rate':0.01,

                   'max_depth': 5,

                   'boosting': 'gbdt',

                   'seed': 42,

                   'objective': 'regression',

                   'metric':'rmse'}



lgb_params_survival={'learning_rate':0.01,

                     'max_depth': 5,

                     'boosting': 'gbdt',

                     'seed': 42,

                     'objective': 'binary',

                     'metric':'auc'}
lgb_train_amount = lgb.Dataset(x_train_amount, y_train_amount)

lgb_train_survival = lgb.Dataset(x_train_survival, y_train_survival)



lgb_valid_amount = lgb.Dataset(x_valid_amount, y_valid_amount)

lgb_valid_survival = lgb.Dataset(x_valid_survival, y_valid_survival)
lgb_amount = lgb.train(lgb_params_amount, 

                       lgb_train_amount,

                       valid_sets = [lgb_train_amount,lgb_valid_amount],

                       num_boost_round = 5000,

                       verbose_eval = 100,

                       early_stopping_rounds = 50)



lgb_amount_pred = lgb_amount.predict(valid_set.drop(columns = ['acc_id','survival_time','amount_spent','is_survive','total_amount_spent']),num_iteration = lgb_amount.best_iteration)

lgb_amount_pred = pd.Series(lgb_amount_pred).apply(lambda x: 0 if x < 0 else x)



lgb_survival = lgb.train(lgb_params_survival, 

                       lgb_train_survival,

                       valid_sets = [lgb_train_survival, lgb_valid_survival],

                       num_boost_round = 5000,

                       verbose_eval = 100,

                       early_stopping_rounds = 50)



lgb_survival_pred = lgb_survival.predict(valid_set.drop(columns = ['acc_id','survival_time','amount_spent','is_survive','total_amount_spent']),num_iteration = lgb_survival.best_iteration)

lgb_survival_pred = pd.Series(lgb_survival_pred).apply(lambda x: 0 if x < 0 else x)
lgb_pred_df = pd.DataFrame({'acc_id': valid_set.acc_id.values,

                            'survival_time': lgb_survival_pred,

                            'amount_spent': lgb_amount_pred})



print('lgb_pred_df shape: ',lgb_pred_df.shape)



scaler_amount = MinMaxScaler(feature_range = [0, 74])

scaler_survival = MinMaxScaler(feature_range = [1, 64])



lgb_pred_df['survival_time'] = scaler_survival.fit_transform(lgb_pred_df['survival_time'].values.reshape(-1,1)).round()



lgb_pred_df['amount_spent'] = lgb_pred_df['amount_spent'] / lgb_pred_df['survival_time']

lgb_pred_df['amount_spent'] = scaler_amount.fit_transform(lgb_pred_df['amount_spent'].values.reshape(-1,1))





lgb_valid_score = score_function(lgb_pred_df, valid_set[['acc_id','survival_time','amount_spent']])

true_score = score_function(valid_set[['acc_id','survival_time','amount_spent']], valid_set[['acc_id','survival_time','amount_spent']])



print('Light GBM score: ',lgb_valid_score)

print('true score: ',true_score)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
X = x_train_survival

y = y_train_survival



sns.set(rc={"axes.titlesize":20,

            'axes.labelsize':10,

            'figure.figsize':(18, 20),

            'xtick.labelsize':10,

            'ytick.labelsize':10})

sns.set_style("whitegrid")



model = GradientBoostingClassifier()

model.fit(X, y)



fig, ax = plot_partial_dependence(model,       

                                  features=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], # column numbers of plots we want to show

                                  X=X,            # raw predictors data.

                                  feature_names=['2_roll_mean2_game_money_change', '1_game_money_change','4_game_money_change',

                                                 '4_roll_mean2_level', '4_level', '4_playtime', '2_game_money_change',

                                                 '2_roll_sd2_npc_kill', '4_private_shop', '4_roll_mean2_game_money_change',

                                                 '2_roll_mean2_solo_exp','1_playtime', '1_solo_exp', '4_npc_kill', '4_roll_sd2_playtime'], # labels on graphs

                                  grid_resolution=10) # number of values to plot on x axis



fig.tight_layout()
X = x_train_amount

y = y_train_amount



sns.set(rc={"axes.titlesize":20,

            'axes.labelsize':10,

            'figure.figsize':(18.0, 20),

            'xtick.labelsize':10,

            'ytick.labelsize':10})

sns.set_style("whitegrid")



model = GradientBoostingRegressor()

model.fit(X, y)



fig, ax = plot_partial_dependence(model,       

                                   features=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], # column numbers of plots we want to show

                                   X=X,            # raw predictors data.

                                   feature_names=['3_roll_mean2_amount_spent', '1_amount_spent', '3_amount_spent',

                                                  '4_roll_mean2_amount_spent', 'buy_item_amount_max', '4_amount_spent',

                                                  '2_roll_mean2_fishing','2_roll_mean2_amount_spent' ,'2_roll_sd2_npc_kill', '3_roll_sd2_solo_exp',

                                                  '4_pledge_cnt', '4_game_money_change','4_roll_sd2_solo_exp',

                                                  '4_roll_mean2_game_money_change', '4_roll_mean2_solo_exp'], # labels on graphs

                                   grid_resolution=10) # number of values to plot on x axis



fig.tight_layout()
from pycebox.ice import ice, ice_plot
x_train_amount_df = pd.DataFrame(x_train_amount, columns=x_train_amount.columns)
x_train_amount_df.shape
ice_df = ice(data=x_train_amount_df, column='2_roll_mean2_game_money_change',

             predict = lgb_amount.predict)



rcParams['figure.figsize'] = 8,6



ice_plot(ice_df , c='dimgray', linewidth=0.3)

plt.ylabel('Pred. AV %ile')

plt.xlabel('thalach')

plt.show()
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

from matplotlib import rcParams
survival_columns = ['2_roll_mean2_game_money_change', '1_game_money_change','4_game_money_change',

                    '4_roll_mean2_level', '4_level', '4_playtime', '2_game_money_change',

                    '2_roll_sd2_npc_kill', '4_private_shop', '4_roll_mean2_game_money_change',

                    '2_roll_mean2_solo_exp','1_playtime', '1_solo_exp', '4_npc_kill', '4_roll_sd2_playtime']



amount_columns = ['3_roll_mean2_amount_spent', '1_amount_spent', '3_amount_spent',

                  '4_roll_mean2_amount_spent', 'buy_item_amount_max', '4_amount_spent',

                  '2_roll_mean2_fishing', '2_roll_sd2_npc_kill', '3_roll_sd2_solo_exp',

                  '4_roll_sd2_solo_exp', '4_roll_mean2_game_money_change', '4_roll_mean2_solo_exp']
def pdpbox_survival(features):

    # Create the data that we will plot

    pdp_goals = pdp.pdp_isolate(model=lgb_survival, dataset=x_train_survival,

                                model_features=x_valid_survival.columns.tolist(), feature=features)

    # plot it

    pdp.pdp_plot(pdp_goals, features)

    plt.show()

    

def pdpbox_amount(features):

    # Create the data that we will plot

    pdp_goals = pdp.pdp_isolate(model=lgb_amount, dataset=x_train_amount,

                                model_features=x_valid_amount.columns.tolist(), feature=features)

    # plot it

    pdp.pdp_plot(pdp_goals, features)

    plt.show()
for i in survival_columns:

    pdpbox_survival(i)
for i in amount_columns:

    pdpbox_amount(i)
# figure size in inches

rcParams['figure.figsize'] = 15,10

sns.set_style("whitegrid")

pdp_interaction = pdp.pdp_interact(model=lgb_amount, dataset=x_train_amount, model_features=x_valid_amount.columns.tolist(), features=['1_playtime', '2_playtime'])

pdp.pdp_interact_plot(pdp_interact_out=pdp_interaction, feature_names=['2_roll_mean2_game_money_change', '1_game_money_change'], plot_type='contour')

plt.show()
# figure size in inches

rcParams['figure.figsize'] = 15,10

sns.set_style("whitegrid")

pdp_interaction = pdp.pdp_interact(model=lgb_amount, dataset=x_train_amount, model_features=x_valid_amount.columns.tolist(), features=['1_playtime', '2_playtime'])

pdp.pdp_interact_plot(pdp_interact_out=pdp_interaction, feature_names=['2_roll_mean2_game_money_change', '1_game_money_change'],

                      x_quantile=True, plot_type='contour', plot_pdp=True)

plt.show()