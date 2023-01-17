import lightgbm as lgb

import numpy as np

import pandas as pd

import itertools

import time

import pprint



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, train_test_split

from sklearn.cluster import KMeans



SEEDS = 42
def rmse(y_true, y_pred):

    return (mean_squared_error(y_true, y_pred))** .5
# load_data

# I prepared dataset beforehand and skipped this process in this notebook.

# You can easily generate augmented dataset. 

# Plese refer to tito's notebook.



train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
"""

def read_bpps_sum(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))

    return bpps_arr



def read_bpps_max(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))

    return bpps_arr



def read_bpps_nb(df):

    # normalized non-zero number

    # from https://www.kaggle.com/symyksr/openvaccine-deepergcn 

    bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data

    bpps_nb_std = 0.08914   # std of bpps_nb across all training data

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")

        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]

        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std

        bpps_arr.append(bpps_nb)

    return bpps_arr 



train['bpps_sum'] = read_bpps_sum(train)

test['bpps_sum'] = read_bpps_sum(test)

train['bpps_max'] = read_bpps_max(train)

test['bpps_max'] = read_bpps_max(test)

train['bpps_nb'] = read_bpps_nb(train)

test['bpps_nb'] = read_bpps_nb(test)

"""
"""

train_data = []

for mol_id in train['id'].unique():

    sample_data = train.loc[train['id'] == mol_id]

    sample_seq_length = sample_data.seq_length.values[0]

    

    for i in range(68):

        sample_dict = {'id' : sample_data['id'].values[0],

                       'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),

                       'sequence' : sample_data['sequence'].values[0][i],

                       'structure' : sample_data['structure'].values[0][i],

                       'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],

                       'reactivity' : sample_data['reactivity'].values[0][i],

                       'reactivity_error' : sample_data['reactivity_error'].values[0][i],

                       'deg_Mg_pH10' : sample_data['deg_Mg_pH10'].values[0][i],

                       'deg_error_Mg_pH10' : sample_data['deg_error_Mg_pH10'].values[0][i],

                       'deg_pH10' : sample_data['deg_pH10'].values[0][i],

                       'deg_error_pH10' : sample_data['deg_error_pH10'].values[0][i],

                       'deg_Mg_50C' : sample_data['deg_Mg_50C'].values[0][i],

                       'deg_error_Mg_50C' : sample_data['deg_error_Mg_50C'].values[0][i],

                       'deg_50C' : sample_data['deg_50C'].values[0][i],

                       'deg_error_50C' : sample_data['deg_error_50C'].values[0][i],

                       'bpps_sum' : sample_data['bpps_sum'].values[0][i],

                       'bpps_max' : sample_data['bpps_max'].values[0][i],

                       'bpps_nb' : sample_data['bpps_nb'].values[0][i],

                       'SN_filter' : sample_data['SN_filter'].values}

        

        

        shifts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        shift_cols = ['sequence', 'structure', 'predicted_loop_type']

        for shift,col in itertools.product(shifts, shift_cols):

            if i - shift >= 0:

                sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]

            else:

                sample_dict['b'+str(shift)+'_'+col] = -1

            

            if i + shift <= sample_seq_length - 1:

                sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]

            else:

                sample_dict['a'+str(shift)+'_'+col] = -1



        shift_cols_2 = ['bpps_sum', 'bpps_max', 'bpps_nb']

        for shift,col in itertools.product(shifts, shift_cols_2):

            if i - shift >= 0:

                sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]

            else:

                sample_dict['b'+str(shift)+'_'+col] = -999

            

            if i + shift <= sample_seq_length - 1:

                sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]

            else:

                sample_dict['a'+str(shift)+'_'+col] = -999

        

        

        train_data.append(sample_dict)

train_data = pd.DataFrame(train_data)

train_data.head()

"""
"""

test_data = []

for mol_id in test['id'].unique():

    sample_data = test.loc[test['id'] == mol_id]

    sample_seq_length = sample_data.seq_length.values[0]

    for i in range(sample_seq_length):

        sample_dict = {'id' : sample_data['id'].values[0],

                       'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),

                       'sequence' : sample_data['sequence'].values[0][i],

                       'structure' : sample_data['structure'].values[0][i],

                       'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],

                       'bpps_sum' : sample_data['bpps_sum'].values[0][i],

                       'bpps_max' : sample_data['bpps_max'].values[0][i],

                       'bpps_nb' : sample_data['bpps_nb'].values[0][i]}

        

        shifts = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        shift_cols = ['sequence', 'structure', 'predicted_loop_type']

        for shift,col in itertools.product(shifts, shift_cols):

            if i - shift >= 0:

                sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]

            else:

                sample_dict['b'+str(shift)+'_'+col] = -1

            

            if i + shift <= sample_seq_length - 1:

                sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]

            else:

                sample_dict['a'+str(shift)+'_'+col] = -1



        shift_cols_2 = ['bpps_sum', 'bpps_max', 'bpps_nb']

        for shift,col in itertools.product(shifts, shift_cols_2):

            if i - shift >= 0:

                sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]

            else:

                sample_dict['b'+str(shift)+'_'+col] = -999

            

            if i + shift <= sample_seq_length - 1:

                sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]

            else:

                sample_dict['a'+str(shift)+'_'+col] = -999

        

        test_data.append(sample_dict)

test_data = pd.DataFrame(test_data)

test_data.head()

"""
"""

# label_encoding



sequence_encmap = {'A': 0, 'G' : 1, 'C' : 2, 'U' : 3}

structure_encmap = {'.' : 0, '(' : 1, ')' : 2}

looptype_encmap = {'S':0, 'E':1, 'H':2, 'I':3, 'X':4, 'M':5, 'B':6}



enc_targets = ['sequence','structure', 'predicted_loop_type']

enc_maps = [sequence_encmap, structure_encmap, looptype_encmap]



for t,m in zip(enc_targets, enc_maps):

    for c in [c for c in train_data.columns if t in c]:



        train_data[c] = train_data[c].replace(m)

        test_data[c] = test_data[c].replace(m)

        

"""
"""

train_data.to_csv("train.csv",index=False)

test_data.to_csv("./test.csv",index=False)

"""
# I recommend you save both the dataset. Because, it takes long to generate these.

#test_data.to_cav('aug_test_df_2.csv', index=False)

#train_data.to_csv('aug_train_df_2.csv' index=False)
# load prepared train data and test data

train_data = pd.read_csv("../input/covid-dataset/train.csv")

test_data = pd.read_csv("../input/covid-dataset/test.csv")

submission = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
train_data[features]
# to save time, I only use 3 columns as targets.

targets = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

#targets = ['reactivity', 'deg_pH10', 'deg_50C']
# features for training

features = ['sequence', 'structure', 'predicted_loop_type', 'bpps_sum', 'bpps_max', 'bpps_nb', 

            'b1_sequence', 'a1_sequence', 'b1_structure', 'a1_structure', 'b1_predicted_loop_type', 

            'a1_predicted_loop_type', 'b2_sequence', 'a2_sequence', 'b2_structure', 'a2_structure', 

            'b2_predicted_loop_type', 'a2_predicted_loop_type', 'b3_sequence', 'a3_sequence', 

            'b3_structure', 'a3_structure', 'b3_predicted_loop_type', 'a3_predicted_loop_type', 

            'b4_sequence', 'a4_sequence', 'b4_structure', 'a4_structure', 'b4_predicted_loop_type', 

            'a4_predicted_loop_type', 'b5_sequence', 'a5_sequence', 'b5_structure', 'a5_structure', 

            'b5_predicted_loop_type', 'a5_predicted_loop_type', 'b6_sequence', 'a6_sequence', 

            'b6_structure', 'a6_structure', 'b6_predicted_loop_type', 'a6_predicted_loop_type', 

            'b7_sequence', 'a7_sequence', 'b7_structure', 'a7_structure', 'b7_predicted_loop_type', 

            'a7_predicted_loop_type', 'b8_sequence', 'a8_sequence', 'b8_structure', 'a8_structure', 

            'b8_predicted_loop_type', 'a8_predicted_loop_type', 'b9_sequence', 'a9_sequence', 

            'b9_structure', 'a9_structure', 'b9_predicted_loop_type', 'a9_predicted_loop_type', 

            'b10_sequence', 'a10_sequence', 'b10_structure', 'a10_structure', 

            'b10_predicted_loop_type', 'a10_predicted_loop_type', 'b11_sequence', 'a11_sequence', 

            'b11_structure', 'a11_structure', 'b11_predicted_loop_type', 'a11_predicted_loop_type', 

            'b12_sequence', 'a12_sequence', 'b12_structure', 'a12_structure', 

            'b12_predicted_loop_type', 'a12_predicted_loop_type', 'b13_sequence', 'a13_sequence', 

            'b13_structure', 'a13_structure', 'b13_predicted_loop_type', 'a13_predicted_loop_type', 

            'b14_sequence', 'a14_sequence', 'b14_structure', 'a14_structure', 

            'b14_predicted_loop_type', 'a14_predicted_loop_type', 'b15_sequence', 'a15_sequence', 

            'b15_structure', 'a15_structure', 'b15_predicted_loop_type', 'a15_predicted_loop_type', 

            'b1_bpps_sum', 'a1_bpps_sum', 'b1_bpps_max', 'a1_bpps_max', 'b1_bpps_nb', 'a1_bpps_nb', 

            'b2_bpps_sum', 'a2_bpps_sum', 'b2_bpps_max', 'a2_bpps_max', 'b2_bpps_nb', 'a2_bpps_nb', 

            'b3_bpps_sum', 'a3_bpps_sum', 'b3_bpps_max', 'a3_bpps_max', 'b3_bpps_nb', 'a3_bpps_nb', 

            'b4_bpps_sum', 'a4_bpps_sum', 'b4_bpps_max', 'a4_bpps_max', 'b4_bpps_nb', 'a4_bpps_nb', 

            'b5_bpps_sum', 'a5_bpps_sum', 'b5_bpps_max', 'a5_bpps_max', 'b5_bpps_nb', 'a5_bpps_nb', 

            'b6_bpps_sum', 'a6_bpps_sum', 'b6_bpps_max', 'a6_bpps_max', 'b6_bpps_nb', 'a6_bpps_nb', 

            'b7_bpps_sum', 'a7_bpps_sum', 'b7_bpps_max', 'a7_bpps_max', 'b7_bpps_nb', 'a7_bpps_nb', 

            'b8_bpps_sum', 'a8_bpps_sum', 'b8_bpps_max', 'a8_bpps_max', 'b8_bpps_nb', 'a8_bpps_nb', 

            'b9_bpps_sum', 'a9_bpps_sum', 'b9_bpps_max', 'a9_bpps_max', 'b9_bpps_nb', 'a9_bpps_nb', 

            'b10_bpps_sum', 'a10_bpps_sum', 'b10_bpps_max', 'a10_bpps_max', 'b10_bpps_nb', 'a10_bpps_nb', 

            'b11_bpps_sum', 'a11_bpps_sum', 'b11_bpps_max', 'a11_bpps_max', 'b11_bpps_nb', 'a11_bpps_nb', 

            'b12_bpps_sum', 'a12_bpps_sum', 'b12_bpps_max', 'a12_bpps_max', 'b12_bpps_nb', 'a12_bpps_nb', 

            'b13_bpps_sum', 'a13_bpps_sum', 'b13_bpps_max', 'a13_bpps_max', 'b13_bpps_nb', 'a13_bpps_nb', 

            'b14_bpps_sum', 'a14_bpps_sum', 'b14_bpps_max', 'a14_bpps_max', 'b14_bpps_nb', 'a14_bpps_nb', 

            'b15_bpps_sum', 'a15_bpps_sum', 'b15_bpps_max', 'a15_bpps_max', 'b15_bpps_nb', 'a15_bpps_nb']

# categorical features

cols_cat = ['sequence', 'structure', 'predicted_loop_type', 'b1_sequence', 'a1_sequence', 

            'b1_structure', 'a1_structure', 'b1_predicted_loop_type', 'a1_predicted_loop_type', 

            'b2_sequence', 'a2_sequence', 'b2_structure', 'a2_structure', 'b2_predicted_loop_type', 

            'a2_predicted_loop_type', 'b3_sequence', 'a3_sequence', 'b3_structure', 'a3_structure', 

            'b3_predicted_loop_type', 'a3_predicted_loop_type', 'b4_sequence', 'a4_sequence', 

            'b4_structure', 'a4_structure', 'b4_predicted_loop_type', 'a4_predicted_loop_type', 

            'b5_sequence', 'a5_sequence', 'b5_structure', 'a5_structure', 'b5_predicted_loop_type', 

            'a5_predicted_loop_type', 'b6_sequence', 'a6_sequence', 'b6_structure', 'a6_structure', 

            'b6_predicted_loop_type', 'a6_predicted_loop_type', 'b7_sequence', 'a7_sequence', 

            'b7_structure', 'a7_structure', 'b7_predicted_loop_type', 'a7_predicted_loop_type', 

            'b8_sequence', 'a8_sequence', 'b8_structure', 'a8_structure', 'b8_predicted_loop_type', 

            'a8_predicted_loop_type', 'b9_sequence', 'a9_sequence', 'b9_structure', 'a9_structure', 

            'b9_predicted_loop_type', 'a9_predicted_loop_type', 'b10_sequence', 'a10_sequence', 

            'b10_structure', 'a10_structure', 'b10_predicted_loop_type', 'a10_predicted_loop_type', 

            'b11_sequence', 'a11_sequence', 'b11_structure', 'a11_structure', 

            'b11_predicted_loop_type', 'a11_predicted_loop_type', 'b12_sequence', 'a12_sequence', 

            'b12_structure', 'a12_structure', 'b12_predicted_loop_type', 'a12_predicted_loop_type', 

            'b13_sequence', 'a13_sequence', 'b13_structure', 'a13_structure', 

            'b13_predicted_loop_type', 'a13_predicted_loop_type', 'b14_sequence', 'a14_sequence', 

            'b14_structure', 'a14_structure', 'b14_predicted_loop_type', 'a14_predicted_loop_type', 

            'b15_sequence', 'a15_sequence', 'b15_structure', 'a15_structure', 

            'b15_predicted_loop_type', 'a15_predicted_loop_type']



train_data[cols_cat] = train_data[cols_cat].astype('category')

test_data[cols_cat] = test_data[cols_cat].astype('category')



# filtering train_data

#train_data = train_data[train_data.SN_filter == 1]
#lgb prediction

start = time.time()



FOLD_N = 5

kf = KFold(n_splits=FOLD_N)



# hyperparameters tuned by optuna lightgbm tuner

params_reactivity = {'bagging_fraction': 1.0,

 'bagging_freq': 0,

 'boosting': 'gbdt',

 'feature_fraction': 0.7,

 'feature_pre_filter': False,

 'lambda_l1': 3.5734348044581306,

 'lambda_l2': 0.05576820758877974,

 'metric': 'rmse',

 'min_child_samples': 50,

 'num_leaves': 71,

 'objective': 'regression',

 'seed': 42}

params_Mg_pH10 = {'bagging_fraction': 0.9832004107802255,

 'bagging_freq': 3,

 'boosting': 'gbdt',

 'feature_fraction': 0.9,

 'feature_pre_filter': False,

 'lambda_l1': 6.661763922280738,

 'lambda_l2': 1.599668434334185e-08,

 'metric': 'rmse',

 'min_child_samples': 100,

 'num_leaves': 256,

 'objective': 'regression',

 'seed': 42}

params_pH10 = {'bagging_fraction': 1.0,

 'bagging_freq': 0,

 'boosting': 'gbdt',

 'feature_fraction': 0.8,

 'feature_pre_filter': False,

 'lambda_l1': 8.414317420167844,

 'lambda_l2': 1.1849676601679822e-08,

 'metric': 'rmse',

 'min_child_samples': 20,

 'num_leaves': 254,

 'objective': 'regression',

 'seed': 42}

params_Mg_50C = {'bagging_fraction': 0.9650708892784845,

 'bagging_freq': 6,

 'boosting': 'gbdt',

 'feature_fraction': 0.7,

 'feature_pre_filter': False,

 'lambda_l1': 6.771757093085835,

 'lambda_l2': 1.1080640906004635e-08,

 'metric': 'rmse',

 'min_child_samples': 20,

 'num_leaves': 100,

 'objective': 'regression',

 'seed': 42}

params_50C = {'bagging_fraction': 1.0,

 'bagging_freq': 0,

 'boosting': 'gbdt',

 'feature_fraction': 0.92,

 'feature_pre_filter': False,

 'lambda_l1': 9.061504176030757e-06,

 'lambda_l2': 7.459533216331888e-05,

 'metric': 'rmse',

 'min_child_samples': 100,

 'num_leaves': 231,

 'objective': 'regression',

 'seed': 42}



params_list = [params_reactivity, params_Mg_pH10, params_pH10, params_Mg_50C, params_50C]



result = {}

oof_df = pd.DataFrame(train_data.id_seqpos)



for i, target in enumerate(targets):

    params = params_list[i]

    oof = pd.DataFrame()

    preds = np.zeros(len(test_data))

    print(f'- predict {target}')

    print('')

    

    for n, (tr_idx, vl_idx) in enumerate(kf.split(train_data[features])):

        tr_x, tr_y = train_data[features].iloc[tr_idx], train_data[target].iloc[tr_idx]

        vl_x, vl_y = train_data[features].iloc[vl_idx], train_data[target].iloc[vl_idx]





        vl_id = train_data['id_seqpos'].iloc[vl_idx]



        t_data = lgb.Dataset(tr_x, label=tr_y) 

        v_data = lgb.Dataset(vl_x, label=vl_y)



        model = lgb.train(params, 

                      t_data, 

                      num_boost_round=15000,

                      valid_sets=[t_data, v_data], 

                      verbose_eval=500, 

                      early_stopping_rounds=150, 

                     )



        vl_pred = model.predict(vl_x)

        score = rmse(vl_y, vl_pred)

        print(f'score : {score}')



        oof = oof.append(pd.DataFrame({'id_seqpos':vl_id, target:vl_pred}))

        pred = model.predict(test_data[features])

        preds += pred / FOLD_N   

    

    oof_df = oof_df.merge(oof, on='id_seqpos', how='inner')

    rmse_score = rmse(train_data[target], oof_df[target])

    print(f'{target} rmse: {rmse_score}')

    submission[target] = preds

    result[target] = rmse_score

    

print(time.time()-start, 'seconds has passed')
import pickle

filename = 'lgbm_model.pickle'

pickle.dump(model, open(filename, 'wb'))
'''

Ver1 result:

{'reactivity': 0.21540200147225563,

 'deg_Mg_pH10': 0.27175680226193105,

 'deg_Mg_50C': 0.2279103378674039}

'total : 0.23835638053386354'

'''



display(result)

display(f'total : {np.mean(list(result.values()))}')
#oof_df.to_csv('oof_df.csv', index=False)

submission.to_csv('submission.csv', index=False)


#!pip install optuna

#import optuna.integration.lightgbm as opt_lgb



#train_opt, val_opt = train_test_split(train_data)

#bestps = []

#targets = ['deg_pH10', 'deg_Mg_50C', 'deg_50C']

#num_round = 1500

#

#for t in targets:

#  params = {'objective': 'regression',

#          'boosting': 'gbdt',

#          'metric': 'rmse',

#          'seed' : SEEDS}

#  print(f'- tuning for {t}')

#  lgb_train = lgb.Dataset(train_opt[features], train_opt[t])

#  lgb_valid = lgb.Dataset(val_opt[features], val_opt[t])

#

#  best = opt_lgb.train(params, lgb_train, num_boost_round=num_round,

#                        valid_names=["train", "valid"], valid_sets=[lgb_train, lgb_valid],

#                        verbose_eval = 0,  early_stopping_rounds = 150)

#  pprint.pprint(best.params)

#  bestps.append(best.params)

#  print('')

#  
"""

for i in range(3):

    pprint.pprint(bestps[i])

"""