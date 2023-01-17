import lightgbm as lgb

from sklearn.metrics import mean_absolute_error

import pandas as pd

import numpy as np 

from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")
PATH = '../input/mol-features'
type_name = {

             0: '1_1JHC_0',

             1: '1_1JHC_1',

             2: '2_2JHH',

             3: '3_1JHN',

             4: '4_2JHN',

             5: '5_2JHC',

             6: '6_3JHH',

             7: '7_3JHC',

             8: '8_3JHN', 

            }
!ls ../input/mol-features
!ls ../input/mol-features/1_1jhc-20190826t180727z-001 #compressed file
folder = {

            '1_1JHC_0': '1_1jhc_0-20190825t153133z-001',

            '1_1JHC_1': '1_1jhc_1-20190826t180741z-001',

            '2_2JHH': '2_2jhh-20190825t170952z-001',

            '3_1JHN': '3_1jhn-20190825t171137z-001',

            '4_2JHN': '4_2jhn-20190825t170517z-001',

            '5_2JHC':'5_2jhc-20190825t175318z-001',

            '6_3JHH':'6_3jhh-20190825t170153z-001',

            '7_3JHC':'7_3jhc-20190825t153045z-001',

            '8_3JHN':'8_3jhn-20190825t170413z-001'

    

}
type_params = {

    

    0: {

    'boosting_type': "gbdt",

    'objective': "huber",

    'learning_rate': 0.1,

    'num_leaves': 511,

    'sub_feature': 0.50,

    'sub_row': 0.5,

    'bagging_freq': 1,

    'metric': 'mae'},



    1: {

    'boosting_type': "gbdt",

    'objective': "huber",

    'learning_rate': 0.1,

    'num_leaves': 100,

    'sub_feature': 0.50,

    'sub_row': 0.5,

    'bagging_freq': 1,

    'metric': 'mae'},

  

    2: {    

    'boosting_type': "gbdt",

    'objective': "huber",

    'learning_rate': 0.01,

    'bagging_freq': 1,

    'metric': 'mae',

    'min_data_in_leaf': 130, 

    'num_leaves': 150, 

    'reg_alpha': 0.5, 

    'reg_lambda': 0.6000000000000001, 

    'sub_feature': 0.30000000000000004, 

    'sub_row': 0.4},

    

    

    3: {

    'boosting_type': "gbdt",

    'objective': "huber",

    'learning_rate': 0.01,

    'bagging_freq': 1,

    'metric': 'mae',

    'min_data_in_leaf': 96, 

    'num_leaves': 30, 

    'reg_alpha': 0.2, 

    'reg_lambda': 0.4, 

    'sub_feature': 0.4, 

    'sub_row': 0.5},

    

    

    4: {

    'boosting_type': "gbdt",

    'objective': "huber",

    'learning_rate': 0.1,

    'bagging_freq': 1,

    'metric': 'mae',

    'min_data_in_leaf': 21, 

    'num_leaves': 200, 

    'reg_alpha': 0.30000000000000004, 

    'reg_lambda': 0.2, 

    'sub_feature': 0.4, 

    'sub_row': 1.0},

    

    5: {

    'boosting_type': "gbdt",

    'objective': "huber",

    'learning_rate': 0.1,

    'bagging_freq': 1,

    'metric': 'mae',

    'num_leaves': 1023, 

    'sub_feature': 0.5, 

    'sub_row': 0.5},

    

    6:{

   'boosting_type': "gbdt",

   'objective': "huber",

   'learning_rate': 0.1,

   'min_data_in_leaf': 50, 

   'num_leaves': 700, 

   'reg_alpha': 0.30000000000000004, 

   'reg_lambda': 0.8, 

   'sub_feature': 0.5, 

   'sub_row': 0.5},

    

    7: {

    'boosting_type': "gbdt",

    'objective': "huber",

    'learning_rate': 0.1,

    'bagging_freq': 1,

    'metric': 'mae',

    'num_leaves': 1023, 

    'sub_feature': 0.5, 

    'sub_row': 0.5},

               



    8: {

    'boosting_type': "gbdt",

    'objective': "huber",

    'learning_rate': 0.01,

    'min_data_in_leaf': 38,

        'num_leaves': 350,

        'reg_alpha': 0.30000000000000004,

        'reg_lambda': 0.6000000000000001,

        'sub_feature': 0.6000000000000001,

        'sub_row': 0.5,

        'metric': 'mae'}





}
sub = pd.read_csv(f'../input/champs-scalar-coupling/sample_submission.csv', low_memory=False)

sub ['typei'] = pd.read_csv(f'../input/champs-scalar-coupling/test.csv', low_memory=False)['type']

sub.head()


score = []



# fit 8(type) model

for idx in type_name:



    ntype = type_name[idx]

    print(ntype)

    direct = folder[ntype]

    x_train_val = pd.read_csv(PATH+f'/{direct}/{ntype}/x_train.csv', index_col=0, low_memory=False)

    print('x_train:', x_train_val.shape)

    x_test_val =pd.read_csv(PATH+f'/{direct}/{ntype}/x_test.csv', index_col=0, low_memory=False)

    print('x_test:', x_test_val.shape)

    ID = pd.read_pickle(PATH+f'/{direct}/{ntype}/ID.csv')

    y_train_val = pd.read_pickle(PATH+f'/{direct}/{ntype}/y_train.csv')



    break

    

    print(f'------------type{ntype}------------')



    maes = []

    predictions = np.zeros(len(x_test_val))

    preds_train = np.zeros(len(x_train_val))



    n_fold = 5

    folds = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=42)



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train_val, ID)):

        strLog = "fold {}".format(fold_)

        print(strLog)



        x_tr, x_val = x_train_val.iloc[trn_idx], x_train_val.iloc[val_idx]

        y_tr, y_val = y_train_val.iloc[trn_idx], y_train_val.iloc[val_idx]



        model = lgb.LGBMRegressor(**type_params[idx], n_estimators=30000,random_state=1)

        model.fit(x_tr,

                  y_tr,

                  eval_set=[(x_tr, y_tr), (x_val, y_val)],

                  eval_metric='mae',

                  verbose=1000,

                  early_stopping_rounds=200

                  )



        # predictions

        preds = model.predict(

            x_test_val)  # , num_iteration=model.best_iteration_)

        predictions += preds / folds.n_splits

        preds = model.predict(

            x_train_val)  # , num_iteration=model.best_iteration_)

        preds_train += preds / folds.n_splits



        preds = model.predict(x_val)



        # mean absolute error

        mae = mean_absolute_error(y_val, preds)

        print('MAE: %.6f' % mae)

        print('Score: %.6f' % np.log(mae))

        maes.append(mae)

        print('')



    sub.loc[sub['typei'] == idx, 'scalar_coupling_constant'] = predictions

    score.append(np.mean(maes))

    print(f'{ntype} MAE:', np.mean(maes))

    print(f'{ntype} Score:', np.log(np.mean(maes)))



    print('')



print('')

print('----------------------')

# print('train score:', sum(np.log(score)) / 8)
#sub.to_csv(f'{config.DATA_DIR}/ano_20000.csv', index=False)
final = sub.drop(['typei'], axis=1)

final.to_csv(f'example.csv', index=False)

final.head()
submission = pd.read_csv('../input/molsubs/lgbm_final.csv')

submission.to_csv('lgbm_final.csv', index=False)

submission.head()