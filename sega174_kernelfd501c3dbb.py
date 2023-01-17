import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn import ensemble, preprocessing

from catboost import CatBoostClassifier, Pool

def one_hot_encoder(df, nan_as_category = True):

    original_columns = list(df.columns)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)

    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns
def train_test():

    

    train = pd.read_csv('../input/application_train.csv')

    test = pd.read_csv('../input/application_test.csv')

    print(train.shape), print(test.shape)

    

    ind_train = train['SK_ID_CURR']

    ind_test = test['SK_ID_CURR']

    

    new_df = train.append(test).reset_index()

    

    avg = [col for col in new_df.columns if 'AVG' in col]

    new_df['AVG_MEAN'] =  new_df[avg].sum(axis=1)

    new_df['AVG_SUM'] =  new_df[avg].sum(axis=1)

    new_df['AVG_STD'] =  new_df[avg].std(axis=1)

    new_df['AVG_MEDIAN'] =  new_df[avg].median(axis=1)

    

    mode = [col for col in new_df.columns if 'MODE' in col]

    new_df['MODE_MEAN'] =  new_df[mode].sum(axis=1)

    new_df['MODE_SUM'] =  new_df[mode].sum(axis=1)

    new_df['MODE_STD'] =  new_df[mode].std(axis=1)

    new_df['MODE_MEDIAN'] =  new_df[mode].median(axis=1)

    

    medi = [col for col in new_df.columns if 'MEDI' in col]

    new_df['MEDI_MEAN'] =  new_df[medi].sum(axis=1)

    new_df['MEDI_SUM'] =  new_df[medi].sum(axis=1)

    new_df['MEDI_STD'] =  new_df[medi].std(axis=1)

    new_df['MEDI_MEDIAN'] =  new_df[medi].median(axis=1)

    

    new_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    

    new_df['AGE_CLIENT']                = new_df['DAYS_BIRTH']/-365

    new_df['EMPLOYED_YEAR']             = new_df['DAYS_EMPLOYED']/-365

    new_df['_REGISTRATION_YEAR']        = new_df['DAYS_REGISTRATION']/-365

    new_df['ID_PUBLISH_YEAR']           = new_df['DAYS_ID_PUBLISH'] / -365

    new_df['RATIO_CHILD_MEMBERS_FAM']   = new_df['CNT_CHILDREN'] /new_df['CNT_FAM_MEMBERS']

    new_df['RATIO_INCOME_MEMBERS_FAM']  = new_df['AMT_INCOME_TOTAL'] /new_df['CNT_FAM_MEMBERS']

    new_df['RATIO_INCOME_CREDIT']       = new_df['AMT_INCOME_TOTAL'] /new_df['AMT_CREDIT']

    new_df['RATIO_INCOME_ANNUITY']      = new_df['AMT_INCOME_TOTAL'] /new_df['AMT_ANNUITY']

    new_df['RATIO_PRICE_INCOME']        = new_df['AMT_GOODS_PRICE'] /new_df['AMT_INCOME_TOTAL']

    new_df['RATIO_PRICE_CREDIT']        = new_df['AMT_GOODS_PRICE'] /new_df['AMT_CREDIT']

    new_df['EXT_SCORE_SUM']             = new_df['EXT_SOURCE_1'] +new_df['EXT_SOURCE_2'] +new_df['EXT_SOURCE_3']

    new_df['EXT_SCORE_MEAN']            = (new_df['EXT_SOURCE_1'] +new_df['EXT_SOURCE_2'] +new_df['EXT_SOURCE_3'])/3

    new_df['OBS_90_CNT_SOCIAL_CIRCLE_SUM']  = new_df['OBS_30_CNT_SOCIAL_CIRCLE'] +new_df['OBS_60_CNT_SOCIAL_CIRCLE'] 

    new_df['OBS_90_CNT_SOCIAL_CIRCLE_MEAN']  = (new_df['OBS_30_CNT_SOCIAL_CIRCLE'] +new_df['OBS_60_CNT_SOCIAL_CIRCLE'])/2 

    new_df['DEF_90_CNT_SOCIAL_CIRCLE_MEAN']  = (new_df['DEF_60_CNT_SOCIAL_CIRCLE'] +new_df['DEF_30_CNT_SOCIAL_CIRCLE'])/2 

    new_df['DEF_90_CNT_SOCIAL_CIRCLE_SUM']  = new_df['DEF_60_CNT_SOCIAL_CIRCLE'] +new_df['DEF_30_CNT_SOCIAL_CIRCLE']    

    

    flag_doc_col = [col for col in new_df.columns if 'FLAG_DOCUMENT' in col]

    new_df['FLAG_DOC_MEAN'] =  train[flag_doc_col].mean(axis=1)

    new_df['FLAG_DOC_SUM'] =  train[flag_doc_col].sum(axis=1)

    

    new_df, col = one_hot_encoder(new_df)

    

    train = new_df.loc[new_df['SK_ID_CURR'].isin(ind_train)]

    test = new_df.loc[new_df['SK_ID_CURR'].isin(ind_test)]

    print(train.shape), print(test.shape)

    return train, test

    
train,test = train_test()
def bureau_bb():

    bureau_balance = pd.read_csv('../input/bureau_balance.csv')

    bureau = pd.read_csv('../input/bureau.csv')

    print(bureau.shape)

    bureau_balance, cat_bb = one_hot_encoder(bureau_balance)

    bb_agg = {

        'MONTHS_BALANCE': ['median','min', 'max'],

             'STATUS_0':      ['sum','mean'],

             'STATUS_1':      ['sum','mean'],

             'STATUS_2':       ['sum','mean'],

             'STATUS_3':       ['sum','mean'],

             'STATUS_4':   ['sum','mean'],

             'STATUS_5':      ['sum','mean'],

             'STATUS_C':          ['sum','mean'],

             'STATUS_X':      ['sum','mean'],

             'STATUS_nan':      ['sum','mean']}

      

    bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(bb_agg)

    bureau_balance_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in  bureau_balance_agg.columns.tolist()])

    bureau_balance_agg = bureau_balance_agg.reset_index()

    bureau = bureau.merge(bureau_balance_agg, how='left',on='SK_ID_BUREAU').drop(['SK_ID_BUREAU'],axis=1)





    bureau, cat_b = one_hot_encoder(bureau)

    b_agg = {'DAYS_CREDIT':              ['median'],

             'CREDIT_DAY_OVERDUE':       ['median','min', 'max'],

             'DAYS_CREDIT_ENDDATE':      ['median'],

             'DAYS_ENDDATE_FACT':        ['median'],

             'DAYS_CREDIT_UPDATE':       ['median'],

             'AMT_CREDIT_MAX_OVERDUE':   ['min','max'],

             'CNT_CREDIT_PROLONG':       ['sum','mean','min','max'],

             'AMT_CREDIT_SUM':           ['min','mean','max'],

             'AMT_CREDIT_SUM_DEBT':      ['min','mean','max'],

             'AMT_CREDIT_SUM_LIMIT':     ['min','mean','max'],

             'AMT_CREDIT_SUM_OVERDUE':   ['min','mean','max'],

             'MONTHS_BALANCE_MEDIAN':    ['median'],

             'MONTHS_BALANCE_MIN':       ['min','median','max'],

             'MONTHS_BALANCE_MIN':       ['min','median','max'],

             'AMT_ANNUITY':              ['min','mean','max'],

            }

    cat_b_agg = {}

    

    for col in cat_b: 

        cat_b_agg[col] = ['mean'] 

        

    for col in cat_bb:

        cat_b_agg[col+'_SUM'] = ['mean'] 

        cat_b_agg[col+'_MEAN'] = ['mean'] 

        

         

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**b_agg,**cat_b_agg}) 

    bureau_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in  bureau_agg.columns.tolist()])

    bureau_agg = bureau_agg.reset_index()

    

    print(bureau_agg.shape)

    return bureau_agg
bureau = bureau_bb()
train = pd.merge(train,bureau,how='left',on='SK_ID_CURR')

test = pd.merge(test,bureau,how='left',on='SK_ID_CURR')
def previous_application():

    previous_application = pd.read_csv('../input/previous_application.csv')

    print(previous_application.shape)

    previous_application , cat_pr = one_hot_encoder(previous_application)

    

    previous_application['RATE_DOWN_PAYMENT'][previous_application['RATE_DOWN_PAYMENT']<0] = np.nan

    previous_application['AMT_DOWN_PAYMENT'][previous_application['AMT_DOWN_PAYMENT']<0] = np.nan

    previous_application['DAYS_TERMINATION'][previous_application['DAYS_TERMINATION'] == 365243] = np.nan

    previous_application['DAYS_LAST_DUE'][previous_application['DAYS_LAST_DUE'] == 365243] = np.nan

    previous_application['DAYS_FIRST_DUE'][previous_application['DAYS_FIRST_DUE'] == 365243] = np.nan

    previous_application['DAYS_FIRST_DRAWING'][previous_application['DAYS_FIRST_DRAWING'] == 365243] = np.nan

    

    

    

    pa_agg ={'AMT_ANNUITY':              ['median','min','max'],

             'AMT_APPLICATION':          ['median','min','max'],

             'AMT_CREDIT':               ['median','min','max'],

             'AMT_DOWN_PAYMENT':         ['median','min','max'],

             'AMT_GOODS_PRICE':          ['median','min','max'],

             'HOUR_APPR_PROCESS_START':  ['mean','min','max'],

             'NFLAG_LAST_APPL_IN_DAY':   ['sum'],

             'RATE_DOWN_PAYMENT':        ['mean','min','max','sum'],

             'RATE_INTEREST_PRIMARY':    ['mean','min','max','sum'],

             'RATE_INTEREST_PRIVILEGED': ['mean','min','max','sum'],

             'DAYS_DECISION':            ['median','min','max'],

             'CNT_PAYMENT':              ['median','min','max'],

             'DAYS_FIRST_DRAWING':       ['median','min','max'],

             'DAYS_FIRST_DUE':           ['median','min','max'],

             'DAYS_LAST_DUE':            ['median','min','max'],

             'DAYS_TERMINATION':         ['median','min','max'],

             'NFLAG_INSURED_ON_APPROVAL':['sum']}

    

    cat_agg = {}

    for cat in cat_pr:

        cat_agg[cat] = ['mean']

    #previous_application_sk_id = previous_application[['SK_ID_CURR','SK_ID_PREV']]

    #previous_application_sk_id = previous_application_sk_id.gr

    previous_application_agg = previous_application.groupby('SK_ID_CURR').agg({**pa_agg, **cat_agg})

    previous_application_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in  previous_application_agg.columns.tolist()])

    previous_application_agg = previous_application_agg.reset_index()

    #previous_application_agg = previous_application_agg.merge(previous_application_sk_id,how='left',on='SK_ID_CURR')

    print(previous_application_agg.shape)

    #previous_application_agg = previous_application_agg.groupby(['SK_ID_CURR']).mean()

        

    return previous_application_agg
previous_application = previous_application()
train = pd.merge(train,previous_application,how='left',on='SK_ID_CURR')

test = pd.merge(test,previous_application,how='left',on='SK_ID_CURR')
def POS_CASH_balance():

    POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')

    print(POS_CASH_balance.shape)

    POS_CASH_balance , cat_pc_b = one_hot_encoder(POS_CASH_balance)

    



    

    pc_b_agg = {'MONTHS_BALANCE':  ['median','min','max'],

        'CNT_INSTALMENT':  ['median','min','max'],

             'CNT_INSTALMENT_FUTURE':  ['median','min','max'],

             'SK_DPD':                 ['median','min','max'],

             'SK_DPD_DEF':             ['median','min','max'],

            }

    cat_agg = {}

    for cat in cat_pc_b:

        cat_agg[cat] = ['mean']

 

    

    POS_CASH_balance_agg = POS_CASH_balance.groupby(['SK_ID_CURR']).agg({**pc_b_agg, **cat_agg})

    POS_CASH_balance_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in  POS_CASH_balance_agg.columns.tolist()])

    POS_CASH_balance_agg = POS_CASH_balance_agg.reset_index()

    print(POS_CASH_balance_agg.shape)

    return POS_CASH_balance_agg



    
POS_CASH_balance = POS_CASH_balance()
train = pd.merge(train,POS_CASH_balance,how='left',on='SK_ID_CURR')

test = pd.merge(test,POS_CASH_balance,how='left',on='SK_ID_CURR')
def credit_card_balance():

    credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')

    credit_card_balance , cat_cc_b = one_hot_encoder(credit_card_balance)

    print(credit_card_balance.shape)

    credit_card_balance['AMT_DRAWINGS_ATM_CURRENT'][credit_card_balance['AMT_DRAWINGS_ATM_CURRENT']<0] = np.nan

    credit_card_balance['AMT_DRAWINGS_CURRENT'][credit_card_balance['AMT_DRAWINGS_CURRENT']<0] = np.nan

    

    cc_agg =     { 'MONTHS_BALANCE':          ['median','min','max'],

                 'AMT_BALANCE':               ['median','min','max'],

                 'AMT_CREDIT_LIMIT_ACTUAL':   ['median'],

                 'AMT_DRAWINGS_ATM_CURRENT':  ['median'],

                 'AMT_DRAWINGS_CURRENT':      ['mean'],

                 'AMT_DRAWINGS_OTHER_CURRENT':['mean'],

                 'AMT_DRAWINGS_POS_CURRENT':  ['mean'],

                 'AMT_INST_MIN_REGULARITY':   ['mean'],

                 'AMT_PAYMENT_CURRENT':       ['median'],

                 'AMT_PAYMENT_TOTAL_CURRENT': ['mean'],

                 'AMT_RECEIVABLE_PRINCIPAL':  ['mean'],

                 'AMT_RECIVABLE':             ['mean'],

                 'AMT_TOTAL_RECEIVABLE':      ['mean'],

                 'CNT_DRAWINGS_ATM_CURRENT':  ['mean','min','max'],

                 'CNT_DRAWINGS_CURRENT':      ['mean','min','max'],

                 'CNT_DRAWINGS_OTHER_CURRENT':['mean','min','max'],

                 'CNT_DRAWINGS_POS_CURRENT':  ['mean','min','max'],

                 'CNT_INSTALMENT_MATURE_CUM': ['median','min','max'],

                 'SK_DPD':                    ['mean','min','max'],

                 'SK_DPD_DEF':                ['mean','min','max']}

    

    cat_agg = {}

    for cat in cat_cc_b:

            cat_agg[cat] = ['mean']

    

    credit_card_balance_agg = credit_card_balance.groupby('SK_ID_CURR').agg({**cc_agg, **cat_agg})

    credit_card_balance_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in  credit_card_balance_agg.columns.tolist()])

    credit_card_balance_agg = credit_card_balance_agg.reset_index()

    print(credit_card_balance_agg.shape)

    return credit_card_balance_agg

    



    

    
credit_card_balance = credit_card_balance()
train = pd.merge(train,credit_card_balance,how='left',on='SK_ID_CURR')

test = pd.merge(test,credit_card_balance,how='left',on='SK_ID_CURR')
def installments_payments():

    installments_payments = pd.read_csv('../input/installments_payments.csv')

    print(installments_payments.shape)

    ip_agg =     {   'NUM_INSTALMENT_VERSION':['median','min','max'],

                 'NUM_INSTALMENT_NUMBER':     ['median','min','max'],

                 'DAYS_INSTALMENT':           ['min','max'],

                 'DAYS_ENTRY_PAYMENT':        ['min','max'],

                 'AMT_INSTALMENT':            ['median'],

                 'AMT_PAYMENT':               ['median'],}

    installments_payments_agg = installments_payments.groupby('SK_ID_CURR').agg(ip_agg)

    installments_payments_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in  installments_payments_agg.columns.tolist()])

    installments_payments_agg = installments_payments_agg.reset_index()

    print(installments_payments_agg.shape)

    return installments_payments_agg

    
installments_payments = installments_payments()
train = pd.merge(train,installments_payments,how='left',on='SK_ID_CURR')

test = pd.merge(test,installments_payments,how='left',on='SK_ID_CURR')
labels = train['TARGET']

idx = test['SK_ID_CURR']

train = train.drop(['index', 'SK_ID_CURR', 'TARGET'],axis=1) 

test = test.drop(['index', 'SK_ID_CURR','TARGET'],axis=1) 
train = train.fillna(0)

test = test.fillna(0)
train = train.values
from sklearn.model_selection import train_test_split

X_train, X_test, lab_train, lab_test = train_test_split(train, labels, test_size=0.3, random_state=43, shuffle = False )
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# Some useful parameters which will come in handy later on

SEED = 4 # for reproducibility

NFOLDS = 5# set folds for out-of-fold prediction

kf = KFold(n_splits= NFOLDS, shuffle=True, random_state=255)

i=0

for train_index, test_index in kf.split(X_train):

    print("TRAIN:", len(train_index), "TEST:", len(test_index))

#for i, (train_index, test_index) in enumerate(kf):

    #print(i,train_index, test_index)
ntrain=X_train.shape[0]

ntest=X_test.shape[0]

print(ntrain, ntest)
# Обучаем модель

from sklearn.metrics import roc_auc_score

import lightgbm

pred_train = np.zeros((ntrain) )

#pred_test=np.zeros((ntest,nb_classes))

i=0

val_loss_ar = []

loss_ar = []

parameters = {}

for train_index, test_index in  kf.split(X_train):

    

    train_data = lightgbm.Dataset(X_train[train_index], label=lab_train[train_index])

    val_train_data = lightgbm.Dataset(X_train[test_index], label=lab_train[test_index])

    #test_data = lightgbm.Dataset(test)

    

    

    parameters['objective'] = 'binary'

    parameters['boosting'] = 'gbdt'

    parameters['n_estimators'] = 1000

    parameters['learning_rate'] = 0.01

    parameters['num_leaves'] = 30

    parameters['max_depth'] = 10

    parameters['min_data_in_leaf'] = 1

    parameters['bagging_fraction'] = 0.9

    parameters['feature_fraction'] = 0.6

    parameters['max_delta_step'] = 0.6

    parameters['metric'] = 'auc'

    parameters['reg_alpha'] = 0.436193,

    parameters['reg_lambda'] =  0.479169

    parameters['seed'] =  300

    

    model = lightgbm.train(parameters,

                           train_data,

                           valid_sets=val_train_data,

                           num_boost_round=4000,

                           early_stopping_rounds=100,verbose_eval=100)



    print('Saving model_'+str(i)+'.txt ...')

    model.save_model('model_'+str(i)+'.txt')

    pr = model.predict(X_train[test_index],num_iteration=model.best_iteration)

    pred_train[test_index]=pr

    err=roc_auc_score(lab_train[test_index], pr)

    print("Fold", i, "Точность работы на тестовых данных:",  err)

    #pred_test=pred_test+model.predict(x_test)

    i+=1
err=roc_auc_score(lab_train, pred_train)

print("error", err)
wr=1./NFOLDS

predTest = np.zeros((ntest))

for i in np.arange(NFOLDS):  

    print('Loading model to predict...')

    # load model to predict

    bst = lightgbm.Booster(model_file='model_'+str(i)+'.txt')

    predTr=bst.predict(X_test,num_iteration=bst.best_iteration)

    predTest=predTest+predTr*wr

    err=roc_auc_score(lab_test, predTest)

    print("error", err)
wr=1./NFOLDS

predTest = np.zeros((len(test)))

for i in np.arange(NFOLDS):    

    print('Loading model to predict... model_'+str(i)+'.txt')

    # load model to predict

    bst = lightgbm.Booster(model_file='model_'+str(i)+'.txt')

    predTr=bst.predict(test,num_iteration=bst.best_iteration)

    predTest=predTest+predTr*wr

    
preds = pd.DataFrame({"SK_ID_CURR": idx, "TARGET": predTest})

preds.to_csv('cat.csv', index=False)