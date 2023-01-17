import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def generate_submission_file(probs_df):

    

    submission = pd.read_csv('../input/healthcareanalyticsii/sample_submission.csv')



    submission['Stay'] =  np.argmax(probs_df.iloc[:,1:].values,axis =1)



    submission['Stay'] = submission['Stay'].map(inv_dic)

    

    submission.to_csv('submission.csv',index=False)



    print(submission.head())
def get_probs_file_name(fname,model):

    

    if model == 'lgb':

        

        fpath = '../input/lets-get-rid-of-the-patients/probs' + fname[10:]

        

    if model == 'xgb':

        

        fpath = '../input/lets-get-rid-of-the-patients-xgboost/probs' + fname[10:]

        

    if model == 'catboost':

        

        fpath = '../input/lets-get-rid-of-the-patients-catboost/probs' + fname[10:]

        

    if model == 'dcn':

        

        fpath = '../input/a-model-built-on-garbage-deeptables-dcn/probs' + fname[10:]

    

    if model == 'dnn':

        

        fpath = '../input/a-model-built-on-garbage-deeptables/probs' + fname[10:]

        

    if model == 'pnn':

        

        fpath = '../input/a-model-built-on-garbage-deeptables-pnn/probs' + fname[10:]

        

    

    return fpath
sample_probs_file = pd.read_csv('../input/lets-get-rid-of-the-patients-catboost/probs_32_0_0.csv')



sample_probs_file.head()
inv_dic ={}



for i in range(len(sample_probs_file.columns[1:])):

    

    inv_dic[i] = sample_probs_file.columns[i+1]
model_stats_catboost = pd.read_csv('../input/lets-get-rid-of-the-patients-catboost/model_stats.csv')



model_stats_lgb = pd.read_csv('../input/lets-get-rid-of-the-patients/model_stats.csv')



model_stats_xgb = pd.read_csv('../input/lets-get-rid-of-the-patients-xgboost/model_stats.csv')



model_stats_dcn = pd.read_csv('../input/a-model-built-on-garbage-deeptables-dcn/model_stats.csv')



model_stats_dnn = pd.read_csv('../input/a-model-built-on-garbage-deeptables/model_stats.csv')



model_stats_pnn = pd.read_csv('../input/a-model-built-on-garbage-deeptables-pnn/model_stats.csv')
##ALL

probs_file_cat = pd.read_csv('../input/lets-get-rid-of-the-patients-catboost/probs.csv')



probs_file_lgb = pd.read_csv('../input/lets-get-rid-of-the-patients/probs.csv')



probs_file_xgb = pd.read_csv('../input/lets-get-rid-of-the-patients-xgboost/probs.csv')



probs_file_dcn = pd.read_csv('../input/a-model-built-on-garbage-deeptables-dcn/probs.csv')



probs_file_dnn = pd.read_csv('../input/a-model-built-on-garbage-deeptables/probs.csv')



probs_file_pnn = pd.read_csv('../input/a-model-built-on-garbage-deeptables-pnn/probs.csv')
#Public LB==44.2814928313451

#sample_probs_file.iloc[:,1:] = probs_file_dcn.iloc[:,1:] + probs_file_pnn.iloc[:,1:]



#sample_probs_file.head()
#generate_submission_file(sample_probs_file)
min_loss_index_lgb = np.argsort(model_stats_catboost.validation_loss.values)



min_loss_index_lgb
min_loss_index_xgb = np.argsort(model_stats_xgb.validation_loss.values)



min_loss_index_xgb
min_loss_index_cat = np.argsort(model_stats_catboost.validation_loss.values)



min_loss_index_cat
min_loss_index_dcn = np.argsort(model_stats_dcn.validation_loss.values)



min_loss_index_dcn
min_loss_index_dnn = np.argsort(model_stats_dnn.validation_loss.values)



min_loss_index_dnn
min_loss_index_pnn = np.argsort(model_stats_pnn.validation_loss.values)



min_loss_index_pnn
max_accuracy_index_catboost = np.argsort(-model_stats_catboost.accuracy.values)



max_accuracy_index_catboost
max_accuracy_index_lgb = np.argsort(-model_stats_lgb.accuracy.values)



max_accuracy_index_lgb
max_accuracy_index_xgb = np.argsort(-model_stats_xgb.accuracy.values)



max_accuracy_index_xgb
max_accuracy_index_dcn = np.argsort(-model_stats_dcn.accuracy.values)



max_accuracy_index_dcn
max_accuracy_index_dnn = np.argsort(-model_stats_dnn.accuracy.values)



max_accuracy_index_dnn
max_accuracy_index_pnn = np.argsort(-model_stats_pnn.accuracy.values)



max_accuracy_index_pnn
probs_df_loss = pd.read_csv('../input/lets-get-rid-of-the-patients-catboost/probs_32_0_0.csv')
def cal_probs_loss(model_name,model_stats_df,min_loss_idx_arr,model_iter):

    

    probs_cal_loss = np.zeros(shape=(137057,11))



    for i,idx in enumerate(min_loss_idx_arr):

    

        probs_cal_loss += pd.read_csv(get_probs_file_name(model_stats_df.iloc[idx,0],model_name)).iloc[:,1:]

    

        if i == model_iter:

        

            break

        

    probs_df_loss.iloc[:,1:] = probs_cal_loss



    #print(probs_df_loss.head())



    return probs_cal_loss

    
probs_df_acc = pd.read_csv('../input/lets-get-rid-of-the-patients-catboost/probs_32_0_0.csv')
def cal_probs_acc(model_name,model_stats_df,max_accuracy_idx_arr,model_iter):

    

    probs_cal_acc = np.zeros(shape=(137057,11))



    for i,idx in enumerate(max_accuracy_idx_arr):



        probs_cal_acc += pd.read_csv(get_probs_file_name(model_stats_df.iloc[idx,0],model_name)).iloc[:,1:]



        if i ==model_iter:



            break



    #print(probs_cal_acc.head())

    

    return probs_cal_acc

    
#probs_df_acc.iloc[:,1:] = cal_probs_acc('dcn',model_stats_dcn,max_accuracy_index_dcn,75)



#probs_df_acc.head()
#generate_submission_file(probs_df_acc)
#probs_acc_dcn = cal_probs_acc('dcn',model_stats_dcn,max_accuracy_index_dcn,75) 

    

#probs_acc_pnn = cal_probs_acc('pnn',model_stats_pnn,max_accuracy_index_pnn,6) 

    

#probs_acc_lgb = cal_probs_acc('lgb',model_stats_lgb,max_accuracy_index_lgb,1) 

    

#probs_df_acc.iloc[:,1:] = probs_acc_dcn + probs_acc_pnn #+probs_acc_lgb



#probs_df_acc.head()
#generate_submission_file(probs_df_acc)
# Best Public LB == 44.3508080698

probs_loss_dcn = cal_probs_loss('dcn',model_stats_dcn,min_loss_index_dcn,2) 

    

probs_loss_pnn = cal_probs_loss('pnn',model_stats_pnn,min_loss_index_pnn,2)



#probs_loss_dnn = cal_probs_loss('dnn',model_stats_dnn,min_loss_index_dnn,1) 

    

#probs_loss_cat = cal_probs_loss('catboost',model_stats_catboost,min_loss_index_cat,1) 

    

probs_df_loss.iloc[:,1:] = 1.075*probs_loss_dcn + 1.01*probs_loss_pnn #+ probs_loss_cat



probs_df_loss.head()
generate_submission_file(probs_df_loss)