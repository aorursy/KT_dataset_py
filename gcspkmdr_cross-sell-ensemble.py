import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def generate_submission_file(probs_df):

    

    submission = pd.read_csv('../input/avcrosssell/sample_submission.csv')



    submission['Response'] = probs_df



    submission.to_csv('submission.csv',index=False)



    print(submission.head())
def get_probs_file_name(fname,model):

    

    if model == 'lgb':

        

        fpath = '../input/cross-sell-lightgbm/' + fname

        

    if model == 'xgb':

        

        fpath = '../input/lets-get-rid-of-the-patients-xgboost/probs/' + fname

        

    if model == 'catboost-32':

        

        fpath = '../input/cross-sell-cb-32/' + fname[10:]

        

    if model == 'catboost-432':

        

        fpath = '../input/cross-sell-cb-432/' + fname[10:]

    

    if model == 'catboost-73':

        

        fpath = '../input/cross-sell-cb-73/' + fname[10:]

        

    if model == 'catboost-1':

        

        fpath = '../input/cross-sell-cb-1/' + fname[10:]

        

    if model == 'catboost-2':

        

        fpath = '../input/cross-sell-cb-2/' + fname[10:]

    

    return fpath
sample_probs_file = pd.read_csv('../input/cross-sell-lightgbm/probs.csv')



sample_probs_file.head()
model_stats_catboost_32 = pd.read_csv('../input/cross-sell-cb-32/model_stats.csv')



model_stats_catboost_432 = pd.read_csv('../input/cross-sell-cb-432/model_stats.csv')



model_stats_catboost_73 = pd.read_csv('../input/cross-sell-cb-73/model_stats.csv')



model_stats_catboost_1 = pd.read_csv('../input/cross-sell-cb-1/model_stats.csv')



model_stats_catboost_2 = pd.read_csv('../input/cross-sell-cb-2/model_stats.csv')



model_stats_lgb = pd.read_csv('../input/cross-sell-lightgbm/model_stats.csv')



model_stats_xgb = pd.read_csv('../input/cross-sell-xgboost/model_stats.csv')
##ALL

probs_file_cat_32 = pd.read_csv('../input/cross-sell-cb-32/probs.csv')



probs_file_cat_432 = pd.read_csv('../input/cross-sell-cb-432/probs.csv')



probs_file_cat_2 = pd.read_csv('../input/cross-sell-cb-2/probs.csv')



probs_file_cat_1 = pd.read_csv('../input/cross-sell-cb-1/probs.csv')



probs_file_cat_73 = pd.read_csv('../input/cross-sell-cb-73/probs.csv')



probs_file_lgb = pd.read_csv('../input/cross-sell-lightgbm/probs.csv')



probs_file_xgb = pd.read_csv('../input/cross-sell-xgboost/probs.csv')
probs_cb = (probs_file_cat_73.iloc[:,1:]+probs_file_cat_32.iloc[:,1:]+probs_file_cat_432.iloc[:,1:]+probs_file_cat_1.iloc[:,1:]+probs_file_cat_2.iloc[:,1:])/5 

probs_ensemble = 0.5*probs_file_cat_73.iloc[:,1:] + 0.5*probs_file_xgb.iloc[:,1:]

probs_ensemble
generate_submission_file(probs_ensemble)
min_loss_index_lgb = np.argsort(model_stats_lgb.validation_loss.values)



min_loss_index_lgb
min_loss_index_xgb = np.argsort(model_stats_xgb.validation_loss.values)



min_loss_index_xgb
min_loss_index_cat_32 = np.argsort(model_stats_catboost_32.validation_loss.values)



min_loss_index_cat_32
min_loss_index_cat_432 = np.argsort(model_stats_catboost_432.validation_loss.values)



min_loss_index_cat_432
min_loss_index_cat_73 = np.argsort(model_stats_catboost_73.validation_loss.values)



min_loss_index_cat_73
min_loss_index_cat_1 = np.argsort(model_stats_catboost_1.validation_loss.values)



min_loss_index_cat_1
min_loss_index_cat_2 = np.argsort(model_stats_catboost_2.validation_loss.values)



min_loss_index_cat_2
max_auc_index_catboost_32 = np.argsort(-model_stats_catboost_32.oof_roc.values)



max_auc_index_catboost_32
max_auc_index_catboost_432 = np.argsort(-model_stats_catboost_432.oof_roc.values)



max_auc_index_catboost_432
max_auc_index_catboost_73 = np.argsort(-model_stats_catboost_73.oof_roc.values)



max_auc_index_catboost_73
max_auc_index_catboost_1 = np.argsort(-model_stats_catboost_1.oof_roc.values)



max_auc_index_catboost_1
max_auc_index_catboost_2 = np.argsort(-model_stats_catboost_2.oof_roc.values)



max_auc_index_catboost_2
max_auc_index_lgb = np.argsort(-model_stats_lgb.oof_roc.values)



max_auc_index_lgb
max_accuracy_index_xgb = np.argsort(-model_stats_xgb.oof_roc.values)



max_accuracy_index_xgb
probs_df_loss = pd.read_csv('../input/cross-sell-lightgbm/probs.csv')
def cal_probs_loss(model_name,model_stats_df,min_loss_idx_arr,model_iter):

    

    probs_cal_loss = np.zeros(shape=(127037,1))



    for i,idx in enumerate(min_loss_idx_arr):

    

        probs_cal_loss += pd.read_csv(get_probs_file_name(model_stats_df.iloc[idx,0],model_name)).iloc[:,1:]

    

        if i == model_iter-1:

        

            break

        

    probs_df_loss.iloc[:,1:] = probs_cal_loss



    #print(probs_df_loss.head())



    return probs_cal_loss/model_iter

    
probs_df_acc = pd.read_csv('../input/cross-sell-lightgbm/probs.csv')
def cal_probs_acc(model_name,model_stats_df,max_accuracy_idx_arr,model_iter):

    

    probs_cal_acc = np.zeros(shape=(127037,))



    for i,idx in enumerate(max_accuracy_idx_arr):



        probs_cal_acc += pd.read_csv(get_probs_file_name(model_stats_df.iloc[idx,0],model_name)).iloc[:,1:]



        if i ==model_iter-1:



            break



    #print(probs_cal_acc.head())

    

    return probs_cal_acc/model_iter

    