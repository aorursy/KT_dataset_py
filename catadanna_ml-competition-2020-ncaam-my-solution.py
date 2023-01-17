# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import csv

import math

import pickle



import category_encoders as ce

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, TheilSenRegressor, HuberRegressor

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier 

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingRegressor 

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis



import xgboost

from xgboost import XGBRegressor

from catboost import CatBoostClassifier, CatBoostRegressor, Pool



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
NAN_STRING_TO_REPLACE = 'zz'

NAN_VALUE_FLOAT = 8888.0

NAN_VALUE_INT = 8888

NAN_VALUE_STRING = '8888'



BATCH_SIZE = 1000

EPOCHS = 5

N_NEURONS = 10



SEED = 8888

SPLITS = 20



SMOOTHING = 0.2

OTHER_NAN = 0



IMPUTING_STRATEGY = 'median'



PARAMS_CATBOOST = dict()

PARAMS_CATBOOST['logging_level'] = 'Silent'

PARAMS_CATBOOST['eval_metric'] = 'Logloss'

PARAMS_CATBOOST['custom_metric'] = 'Logloss'

PARAMS_CATBOOST['loss_function'] = 'Logloss'

PARAMS_CATBOOST['iterations'] = 125 # best 125

PARAMS_CATBOOST['od_type'] = 'Iter' # IncToDec, Iter

PARAMS_CATBOOST['random_seed'] = SEED

PARAMS_CATBOOST['learning_rate'] = 0.003 # alpha, default 0.03 if no l2_leaf_reg

PARAMS_CATBOOST['task_type'] = 'CPU'

PARAMS_CATBOOST['use_best_model']: True

PARAMS_CATBOOST['l2_leaf_reg'] = 3.0 # lambda, default 3, S: 300



PARAMS_CATBOOST_REGRESSOR = dict()

PARAMS_CATBOOST_REGRESSOR['logging_level'] = 'Silent'

PARAMS_CATBOOST_REGRESSOR['eval_metric'] = 'RMSE'

PARAMS_CATBOOST_REGRESSOR['custom_metric'] = 'RMSE'

PARAMS_CATBOOST_REGRESSOR['loss_function'] = 'RMSE'

PARAMS_CATBOOST_REGRESSOR['iterations'] = 5

PARAMS_CATBOOST_REGRESSOR['od_type'] = 'Iter' # IncToDec, Iter

PARAMS_CATBOOST_REGRESSOR['random_seed'] = SEED

PARAMS_CATBOOST_REGRESSOR['learning_rate'] = 0.003 # alpha, default 0.03 if no l2_leaf_reg

PARAMS_CATBOOST_REGRESSOR['task_type'] = 'CPU'

PARAMS_CATBOOST_REGRESSOR['use_best_model']: True

PARAMS_CATBOOST_REGRESSOR['l2_leaf_reg'] = 3.0 # lambda, default 3, S: 300



PARAMS_XGB = dict()

PARAMS_XGB['objective']='binary:logistic'

PARAMS_XGB['eval_metric'] = 'mae'

PARAMS_XGB['booster'] = 'gbtree'

PARAMS_XGB['eta'] = 0.02

PARAMS_XGB['subsample'] = 0.35

PARAMS_XGB['colsample_bytree'] = 0.7

PARAMS_XGB['num_parallel_tree'] = 10

PARAMS_XGB['min_child_weight'] = 40

PARAMS_XGB['gamma'] = 10

PARAMS_XGB['max_depth'] = 3





W_FEATURES = [

    'WTeamID', 

    'WFGM', 

    'WFGA', 

    'WFGM3', 

    'WFGA3', 

    'WFTM', 

    'WFTA', 

    'WOR', 

    'WDR', 

    'WAst', 

    'WTO', 

    'WStl', 

    'WBlk', 

    'WPF', 

    'WScore', 

    'Final_WTeam', 

    #'Semi_Final_WTeam', 

    'WTeam_W_count', 

    'WScore_mean',

    'WScore_median', 

    'WScore_sum',

    'WTeam_Seed',

    'WTeam_PerCent',

    'Diff_WTeam',

    'WFGA_min', 

    #'WFGA_max', 

    'WFGA_mean',

    'WFGA_median'

    #'WAst_mean',

    #'WBlk_mean'

]



L_FEATURES = [

    'LTeamID', 

    'LFGM', 

    'LFGA', 

    'LFGM3', 

    'LFGA3', 

    'LFTM', 

    'LFTA', 

    'LOR', 

    'LDR', 

    'LAst', 

    'LTO', 

    'LStl', 

    'LBlk', 

    'LPF', 

    'LScore',

    'Final_LTeam', 

    #'Semi_Final_LTeam', 

    'LTeam_L_count', 

    'LScore_mean',  

    'LScore_median', 

    'LScore_sum',

    'LTeam_Seed',

    'LTeam_PerCent',

    'Diff_LTeam',

    'LFGA_min', 

    #'LFGA_max', 

    'LFGA_mean', 

    'LFGA_median'

    #'LAst_mean',

    #'LBlk_mean'

]



'''

Description: Read Data from CSV file into Pandas DataFrame

'''

def read_data(inFile, sep=','):

    df_op = pd.read_csv(filepath_or_buffer=inFile, low_memory=False, encoding='utf-8', sep=sep)

    return df_op



#Description: Write Pandas DataFrame into CSV file

def write_data(df, outFile):

    f = open(outFile+'.csv', 'w')

    r = df.to_csv(index=False, path_or_buf=f)

    f.close()



# Description: Create submission file:    

def print_submission_into_file(y_pred, df_test_id, algo=""):

    l = []

    for myindex in range(y_pred.shape[0]):

        Y0 = y_pred[myindex]

        l.insert(myindex, Y0)

    

    df_pred = pd.DataFrame(pd.Series(l), columns=["Pred"])

    df_result = pd.concat([df_test_id, df_pred], axis=1, sort=False)

     

    f = open('submission'+algo+'.csv', 'w')

    r = df_result.to_csv(index=False, path_or_buf=f)

    f.close()



    return df_result





# Description: Generate string in the format of submission ID

def concat_row(r):

    if r['WTeamID'] < r['LTeamID']:

        res = str(r['Season'])+"_"+str(r['WTeamID'])+"_"+str(r['LTeamID'])

    else:

        res = str(r['Season'])+"_"+str(r['LTeamID'])+"_"+str(r['WTeamID'])

    return res





# Delete leaked from train

def delete_leaked_from_df_train(df_train, df_test):

    # Delete leaked from train

    dft = df_train.loc[:, ['Season','WTeamID','LTeamID']]

    df_train['Concats'] = df_train.apply(concat_row, axis=1)

    df2 = df_test[df_test['ID'].isin(df_train['Concats'].unique())]



    df_train_duplicates = df_train[df_train['Concats'].isin(df_test['ID'].unique())]

    df_train_idx = df_train_duplicates.index.values

    

    df_train = df_train.drop(df_train_idx)

    df_train = df_train.drop('Concats', axis=1)

    

    return df_train



# Convert seed to numeric:

def replace_seed_only(s):

    s = s.replace('W', '')

    s = s.replace('X', '')

    s = s.replace('Y', '')

    s = s.replace('Z', '')

    

    if re.search('(a|b)', s):

        s = s.replace('a', '')

        s = s.replace('b', '')

    else:

        s = s+'0'

     

    return int(s)



       

def log_loss(y_01, y_p):

    n = y_01.shape[0]

    v = np.multiply(y_01, np.log(y_p)) + np.multiply((1-y_01), np.log(1-y_p))

    

    res = -(np.sum(v)/float(n)) 

    return res



def set_aggregation(row, se_agg, se_col, r_col, op_col):

    df_s = se_agg[se_agg[se_col] == row[r_col]]

    df = df_s[df_s['Season']==row['Season']].reset_index(drop=True)

    if df.shape[0] == 0:

        return 0

    else:

        return df.at[0, op_col]

    

# Get value for count features for a team, and replace NaNs withe zero:    

def get_value_for_count(team, team_name, team_count):

    if team in team_count.index:

        return team_count.loc[team, 'Count']

    else:

        return 0

        

def set_WLoc(row):

    if row==1:

        return 2

    elif row==2:

        return 1

    else:

        return 0

    

def write_label(r):

    if r['WTeamID'] < r['LTeamID']:

        return 1

    else:

        return 0

        

    

def get_labels_df_train(df_train, df_test):

    df_train['Concats'] = df_train.apply(concat_row, axis=1)

    df_train_good = df_train[df_train['Concats'].isin(df_test['ID'].unique())]

    df_train_good['Label'] = df_train_good.apply(write_label, axis=1)

    df_train_good['Concats'] = df_train_good.apply(concat_row, axis=1)

    return df_train_good  



PATH = "/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/"

df_mncaa_tourney_detailed_results = read_data(PATH+"MDataFiles_Stage1/MNCAATourneyDetailedResults.csv")

df_regular_season_detailed_results = read_data(PATH+"MDataFiles_Stage1/MRegularSeasonDetailedResults.csv")

df_tourney_seeds = read_data(PATH+"MDataFiles_Stage1/MNCAATourneySeeds.csv")



df_test = read_data(PATH+"MSampleSubmissionStage1_2020.csv")



df_train = df_mncaa_tourney_detailed_results

labels = get_labels_df_train(df_train, df_test)

df_train = delete_leaked_from_df_train(df_train, df_test)



'''

Prepare seeds :

'''



df_tourney_seeds['SeedID'] = df_tourney_seeds['Seed'].apply(replace_seed_only)



mapping_WLoc = {'N':0, 'A':1, 'H':2}

df_train['WLoc'] = df_train.loc[df_train.WLoc.notnull(), 'WLoc'].map(mapping_WLoc)



# Features to parse

features = df_train.columns



df_train_features = df_train[features]

df_train_features = df_train_features.fillna(NAN_VALUE_INT)



# Create simple imputer

si_mf = SimpleImputer(missing_values=NAN_VALUE_INT, strategy=IMPUTING_STRATEGY)

ar_train = si_mf.fit_transform(df_train_features)

df_train = pd.DataFrame(ar_train, columns=features)



df_train_tcr = df_train.copy()

df_train_tcr = df_train_tcr.fillna(NAN_VALUE_INT)

features_tcr = ['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore', 'NumOT']



# Count number of times each team arrived in Final game

df_train_tcr_final = df_train_tcr[df_train_tcr['DayNum']==154]

ar_tcr_final_teams = df_train_tcr_final.loc[:,['WTeamID', 'LTeamID']].to_numpy()

ar_tcr_final_teams = np.unique(ar_tcr_final_teams)

ar_final_teams_count = np.array(np.unique(ar_tcr_final_teams.flatten(), return_counts=True)).T

df_final_teams_count = pd.DataFrame(ar_final_teams_count, columns=['TeamID','Count'])



# Count number of times each team arrived in Semi Final game

df_train_semi_final = df_train_tcr[df_train_tcr['DayNum']==152]

ar_semi_final_teams = df_train_semi_final.loc[:,['WTeamID', 'LTeamID']].to_numpy()

ar_semi_final_teams = np.unique(ar_semi_final_teams)

ar_semi_final_teams_count = np.array(np.unique(ar_semi_final_teams.flatten(), return_counts=True)).T

df_semi_final_teams_count = pd.DataFrame(ar_semi_final_teams_count, columns=['TeamID','Count'])





'''

AGREGATES:

'''

wt_se_agg = df_train_tcr.groupby(['Season', 'WTeamID']).agg({'WScore':['sum', 'mean', 'median', 'count']})

lt_se_agg = df_train_tcr.groupby(['Season', 'LTeamID']).agg({'WScore':['sum', 'mean', 'median', 'count']})



wt_se_agg.columns = ['sum', 'mean', 'median', 'count']

wt_se_agg = wt_se_agg.reset_index()



lt_se_agg.columns = ['sum', 'mean', 'median', 'count']

lt_se_agg = lt_se_agg.reset_index()



# Sum and mean

wt_mean = df_train_tcr.groupby('WTeamID').mean()

wt_sum = df_train_tcr.groupby('WTeamID').sum()

wt_median = df_train_tcr.groupby('WTeamID').median()

lt_mean = df_train_tcr.groupby('LTeamID').mean()

lt_sum = df_train_tcr.groupby('LTeamID').sum()

lt_median = df_train_tcr.groupby('LTeamID').median()



# Nb wins, lose

wt_count = df_train_tcr.groupby('WTeamID').size().to_frame()

lt_count = df_train_tcr.groupby('LTeamID').size().to_frame()



wt_count.columns = ['Count']

lt_count.columns = ['Count']



# Min

wt_min = df_train_tcr.groupby('WTeamID').min()

lt_min = df_train_tcr.groupby('LTeamID').min()



# Max

wt_max = df_train_tcr.groupby('WTeamID').max()

lt_max = df_train_tcr.groupby('LTeamID').max()



#Insert seed information for the training and test set : 

df_train['WTeam_Seed'] = df_train.loc[:,['Season','WTeamID']].apply(lambda row: set_aggregation(row, df_tourney_seeds, 'TeamID', 'WTeamID', 'SeedID'), axis=1)

df_train['LTeam_Seed'] = df_train.loc[:,['Season','LTeamID']].apply(lambda row: set_aggregation(row, df_tourney_seeds, 'TeamID', 'LTeamID', 'SeedID'), axis=1)



df_test_id = df_test["ID"]

df_test = df_test["ID"].apply(lambda x: pd.Series(x.split("_"))).astype('int16')

df_test.columns = ['Season', 'WTeamID', 'LTeamID']



df_test['WTeam_Seed'] = df_test.loc[:,['Season','WTeamID']].apply(lambda row: set_aggregation(row, df_tourney_seeds, 'TeamID', 'WTeamID', 'SeedID'), axis=1)

df_test['LTeam_Seed'] = df_test.loc[:,['Season','LTeamID']].apply(lambda row: set_aggregation(row, df_tourney_seeds, 'TeamID', 'LTeamID', 'SeedID'), axis=1)



df_test['DayNum'] = NAN_VALUE_INT

df_test['NumOT'] = df_train_tcr['NumOT'].max()

df_test['WLoc'] = 0

features = df_train.columns

df_train_features = df_train.fillna(NAN_VALUE_INT)



# Create simple imputer

si_mf = SimpleImputer(missing_values=NAN_VALUE_INT, strategy=IMPUTING_STRATEGY)

si_mf.fit(df_train_features)



imputation = 0



w_features = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'WScore']

l_features = ['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'LScore']



features = df_train.columns

if imputation == 0:

    for cn in features:

        if cn in ['Season', 'WTeamID', 'LTeamID', 'WLoc', 'DayNum', 'WTeam_Seed', 'LTeam_Seed']:

            continue

        df_test[cn] = NAN_VALUE_INT



    # Impute to df_test

    df_test = df_test.fillna(NAN_VALUE_INT)

    ar_test = si_mf.transform(df_test)

    df_test = pd.DataFrame(ar_test, columns=features).astype('float64')        

elif imputation == 1:

    df_test['DayNum'] = df_train['DayNum'].median()

    df_test['NumOT']  = df_train['NumOT'].median()

     

    agg_strategy = 'mean'

    

    for i in range(len(w_features)):

        cn_w = w_features[i]

        cn_l = l_features[i]

        print(cn_w)

        

        wt_agg = df_train.groupby(['Season', 'WTeamID']).agg({cn_w:['sum', 'mean', 'median']})

        wt_agg.columns = ['sum', 'mean', 'median']

        wt_agg = wt_agg.reset_index()

        df_test[cn_w] = df_test.loc[:, ['Season', 'WTeamID']].apply(lambda row: set_aggregation(row, wt_agg, 'WTeamID', 'WTeamID', agg_strategy), axis=1)

        

        lt_agg = df_train.groupby(['Season', 'LTeamID']).agg({cn_l:['sum', 'mean', 'median']})

        lt_agg.columns = ['sum', 'mean', 'median']

        lt_agg = lt_agg.reset_index()

        df_test[cn_l] = df_test.loc[:,['Season','LTeamID']].apply(lambda row: set_aggregation(row, lt_agg, 'LTeamID', 'LTeamID',agg_strategy), axis=1)

       

elif imputation == 2: # mean

    df_test['DayNum'] = df_train['DayNum'].mean()

    df_test['NumOT']  = df_train['NumOT'].mean()

 

    for i in range(len(w_features)):

        cn_w = w_features[i]

        cn_l = l_features[i]

        print(cn_w)



        df_test[cn_w] = df_test['WTeamID'].map(wt_mean[cn_w])

        df_test[cn_l] = df_test['LTeamID'].map(lt_mean[cn_l])

   

    df_test = df_test.fillna(0)



elif imputation == 3: # median

    df_test['DayNum'] = df_train['DayNum'].median()

    df_test['NumOT']  = df_train['NumOT'].median()

    

    for i in range(len(w_features)):

        cn_w = w_features[i]

        cn_l = l_features[i]

        print(cn_w)



        df_test[cn_w] = df_test['WTeamID'].map(wt_median[cn_w])

        df_test[cn_l] = df_test['LTeamID'].map(lt_median[cn_l])

   

    df_test = df_test.fillna(0)



'''

PREPROCESS TRAIN

'''

    

# Final df_train

df_train_final = df_train.loc[df_train['WTeamID'].isin(ar_tcr_final_teams)]

df_train_final_indexes = df_train_final.index.values

df_train = df_train.assign(Final_WTeam=0)

df_train_final = df_train_final.assign(Final_WTeam=0)

df_train_final.loc[df_train_final_indexes, 'Final_WTeam'] = df_train_final['WTeamID'].map(df_final_teams_count.set_index('TeamID')['Count'])

df_train.update(df_train_final)



df_train_final = df_train.loc[df_train['LTeamID'].isin(ar_tcr_final_teams)]

df_train_final_indexes = df_train_final.index.values

df_train = df_train.assign(Final_LTeam= 0)

df_train_final = df_train_final.assign(Final_LTeam=0)

df_train_final.loc[df_train_final_indexes, 'Final_LTeam'] = df_train_final['LTeamID'].map(df_final_teams_count.set_index('TeamID')['Count'])

df_train.update(df_train_final)



# Semi final df_train

df_train_final = df_train.loc[df_train['WTeamID'].isin(ar_semi_final_teams)]

df_train_final_indexes = df_train_final.index.values

df_train['Semi_Final_WTeam'] = 0

df_train_final['Semi_Final_WTeam'] = 0

df_train_final.loc[df_train_final_indexes, 'Semi_Final_WTeam'] = df_train_final['WTeamID'].map(df_semi_final_teams_count.set_index('TeamID')['Count'])

df_train.update(df_train_final)



df_train_final = df_train.loc[df_train['LTeamID'].isin(ar_semi_final_teams)]

df_train_final_indexes = df_train_final.index.values

df_train['Semi_Final_LTeam'] = 0

df_train_final['Semi_Final_LTeam'] = 0

df_train_final.loc[df_train_final_indexes, 'Semi_Final_LTeam'] = df_train_final['LTeamID'].map(df_semi_final_teams_count.set_index('TeamID')['Count'])

df_train.update(df_train_final)





# I tested these features but removed them as they lowered the score :

parse_these = False



if parse_these:

    df_train['WScore_mean'] = df_train['WTeamID'].map(wt_mean['WScore'])

    df_train['LScore_mean'] = df_train['LTeamID'].map(wt_mean['LScore'])

    df_train['WScore_median'] = df_train['WTeamID'].map(wt_median['WScore'])

    df_train['LScore_median'] = df_train['LTeamID'].map(wt_median['LScore'])

    df_train['WScore_sum'] = df_train['WTeamID'].map(wt_sum['WScore'])

    df_train['LScore_sum'] = df_train['LTeamID'].map(wt_sum['LScore'])



    df_train['WAst_mean'] = df_train['WTeamID'].map(wt_mean['WAst'])

    df_train['LAst_mean'] = df_train['LTeamID'].map(lt_mean['LAst'])



    df_train['WBlk_mean'] = df_train['WTeamID'].map(wt_mean['WBlk'])

    df_train['LBlk_mean'] = df_train['LTeamID'].map(lt_mean['LBlk'])



    df_train['WFGA_max'] = df_train['WTeamID'].map(wt_max['WFGA'])

    df_train['LFGA_max'] = df_train['LTeamID'].map(lt_max['LFGA'])





df_train['WFGA_mean'] = df_train['WTeamID'].map(wt_mean['WFGA'])

df_train['LFGA_mean'] = df_train['LTeamID'].map(lt_mean['LFGA'])



df_train['WFGA_median'] = df_train['WTeamID'].map(wt_median['WFGA'])

df_train['LFGA_median'] = df_train['LTeamID'].map(lt_median['LFGA'])



df_train['WFGA_min'] = df_train['WTeamID'].map(wt_min['WFGA'])

df_train['LFGA_min'] = df_train['LTeamID'].map(lt_min['LFGA'])



df_train['WScore_mean'] = df_train.loc[:,['Season','WTeamID']].apply(lambda row: set_aggregation(row, wt_se_agg, 'WTeamID', 'WTeamID', 'mean'), axis=1)

df_train['LScore_mean'] = df_train.loc[:,['Season','LTeamID']].apply(lambda row: set_aggregation(row, lt_se_agg, 'LTeamID', 'LTeamID', 'mean'), axis=1)

df_train['WScore_median'] = df_train.loc[:,['Season','WTeamID']].apply(lambda row: set_aggregation(row, wt_se_agg, 'WTeamID', 'WTeamID', 'median'), axis=1)

df_train['LScore_median'] = df_train.loc[:,['Season','LTeamID']].apply(lambda row: set_aggregation(row, lt_se_agg, 'LTeamID', 'LTeamID', 'median'), axis=1)

df_train['WScore_sum'] = df_train.loc[:,['Season','WTeamID']].apply(lambda row: set_aggregation(row, wt_se_agg, 'WTeamID', 'WTeamID', 'sum'), axis=1)

df_train['LScore_sum'] = df_train.loc[:,['Season','LTeamID']].apply(lambda row: set_aggregation(row, lt_se_agg, 'LTeamID', 'LTeamID', 'sum'), axis=1)



# Counts

df_train['WTeam_W_count'] = OTHER_NAN

df_train['LTeam_L_count'] = OTHER_NAN



count_wt_win = df_train['WTeamID'].map(wt_count['Count'])

count_lt_lose = df_train['LTeamID'].map(lt_count['Count'])

count_wt_lose = df_train['WTeamID'].apply(lambda row: get_value_for_count(row, 'LTeamID', lt_count))

count_lt_win = df_train['LTeamID'].apply(lambda row: get_value_for_count(row, 'WTeamID', wt_count))



df_train['WTeam_W_count'] = count_wt_win

df_train['LTeam_L_count'] = count_lt_lose



df_train['Diff_WTeam'] = count_wt_win - count_wt_lose

df_train['Diff_LTeam'] = count_lt_win - count_lt_lose



df_train['WTeam_PerCent'] = count_wt_win / (count_wt_win + count_wt_lose)

df_train['LTeam_PerCent'] = count_lt_win / (count_lt_win + count_lt_lose)



df_train['WTeam_W_count'] = df_train['WTeam_W_count'].fillna(OTHER_NAN)

df_train['LTeam_L_count'] = df_train['LTeam_L_count'].fillna(OTHER_NAN)



'''

PREPROCESS TEST

'''



# Final df_test

df_test_final = df_test.loc[df_test['WTeamID'].isin(ar_tcr_final_teams)]

df_train_final_indexes = df_test_final.index.values

df_test = df_test.assign(Final_WTeam=0)

df_test_final = df_test_final.assign(Final_WTeam=0)

df_test_final.loc[df_train_final_indexes, 'Final_WTeam'] = df_test_final['WTeamID'].map(df_final_teams_count.set_index('TeamID')['Count'])

df_test.update(df_test_final)



df_test_final = df_test[df_test['LTeamID'].isin(ar_tcr_final_teams)]

df_train_final_indexes = df_test_final.index.values

df_test = df_test.assign(Final_LTeam=0)

df_test_final = df_test_final.assign(Final_LTeam=0)

df_test_final.loc[df_train_final_indexes, 'Final_LTeam'] = df_test_final['LTeamID'].map(df_final_teams_count.set_index('TeamID')['Count'])

df_test.update(df_test_final)



# Semi final df_test

df_test_semi_final = df_test.loc[df_test['WTeamID'].isin(ar_semi_final_teams)]

df_test_final_indexes = df_test_semi_final.index.values

df_test['Semi_Final_WTeam'] = 0

df_test_semi_final['Semi_Final_WTeam'] = 0

df_test_semi_final.loc[df_test_final_indexes, 'Semi_Final_WTeam'] = df_test_semi_final['WTeamID'].map(df_semi_final_teams_count.set_index('TeamID')['Count'])

df_test.update(df_test_semi_final)



df_test_semi_final = df_test[df_test['LTeamID'].isin(ar_semi_final_teams)]

df_test_final_indexes = df_test_semi_final.index.values

df_test['Semi_Final_LTeam'] = 0

df_test_semi_final['Semi_Final_LTeam'] = 0

df_test_semi_final.loc[df_test_final_indexes, 'Semi_Final_LTeam'] = df_test_semi_final['LTeamID'].map(df_semi_final_teams_count.set_index('TeamID')['Count'])

df_test.update(df_test_semi_final)





# I tested these features but removed them as they lowered the score :

parse_these_test = False



if parse_these_test:

    df_test['WScore_mean'] = df_test['WTeamID'].map(wt_mean['WScore'])

    df_test['LScore_mean'] = df_test['LTeamID'].map(wt_mean['LScore'])

    df_test['WScore_median'] = df_test['WTeamID'].map(wt_median['WScore'])

    df_test['LScore_median'] = df_test['LTeamID'].map(wt_median['LScore'])

    df_test['WScore_sum'] = df_test['WTeamID'].map(wt_sum['WScore'])

    df_test['LScore_sum'] = df_test['LTeamID'].map(wt_sum['LScore'])



    df_test['WAst_mean'] = df_test['WTeamID'].map(wt_mean['WAst'])

    df_test['LAst_mean'] = df_test['LTeamID'].map(lt_mean['LAst'])



    df_test['WBlk_mean'] = df_test['WTeamID'].map(wt_mean['WBlk'])

    df_test['LBlk_mean'] = df_test['LTeamID'].map(lt_mean['LBlk'])

    

    df_test['WFGA_max'] = df_test['WTeamID'].map(wt_max['WFGA'])

    df_test['LFGA_max'] = df_test['LTeamID'].map(lt_max['LFGA'])



df_test['WFGA_mean'] = df_test['WTeamID'].map(wt_mean['WFGA'])

df_test['LFGA_mean'] = df_test['LTeamID'].map(lt_mean['LFGA'])



df_test['WFGA_median'] = df_test['WTeamID'].map(wt_median['WFGA'])

df_test['LFGA_median'] = df_test['LTeamID'].map(lt_median['LFGA'])



df_test['WFGA_min'] = df_test['WTeamID'].map(wt_min['WFGA'])

df_test['LFGA_min'] = df_test['LTeamID'].map(lt_min['LFGA'])



df_test['WScore_mean'] = df_test.loc[:,['Season','WTeamID']].apply(lambda row: set_aggregation(row, wt_se_agg, 'WTeamID', 'WTeamID', 'mean'), axis=1)

df_test['LScore_mean'] = df_test.loc[:,['Season','LTeamID']].apply(lambda row: set_aggregation(row, lt_se_agg, 'LTeamID', 'LTeamID', 'mean'), axis=1)

df_test['WScore_median'] = df_test.loc[:,['Season','WTeamID']].apply(lambda row: set_aggregation(row, wt_se_agg, 'WTeamID', 'WTeamID', 'median'), axis=1)

df_test['LScore_median'] = df_test.loc[:,['Season','LTeamID']].apply(lambda row: set_aggregation(row, lt_se_agg, 'LTeamID', 'LTeamID', 'median'), axis=1)

df_test['WScore_sum'] = df_test.loc[:,['Season','WTeamID']].apply(lambda row: set_aggregation(row, wt_se_agg, 'WTeamID', 'WTeamID', 'sum'), axis=1)

df_test['LScore_sum'] = df_test.loc[:,['Season','LTeamID']].apply(lambda row: set_aggregation(row, lt_se_agg, 'LTeamID', 'LTeamID', 'sum'), axis=1)



# Counts

count_wt_win = df_test['WTeamID'].map(wt_count['Count'])

count_lt_lose = df_test['LTeamID'].map(lt_count['Count'])

count_wt_lose = df_test['WTeamID'].apply(lambda row: get_value_for_count(row, 'LTeamID', lt_count))

count_lt_win = df_test['LTeamID'].apply(lambda row: get_value_for_count(row, 'WTeamID', wt_count))



df_test['WTeam_W_count'] = OTHER_NAN

df_test['LTeam_L_count'] = OTHER_NAN



df_test['WTeam_W_count'] = count_wt_win

df_test['LTeam_L_count'] = count_lt_lose



df_test['WTeam_W_count'] = df_test['WTeam_W_count'].fillna(OTHER_NAN)

df_test['LTeam_L_count'] = df_test['LTeam_L_count'].fillna(OTHER_NAN)



df_test['Diff_WTeam'] = count_wt_win - count_wt_lose

df_test['Diff_LTeam'] = count_lt_win - count_lt_lose



df_test['Diff_WTeam'] = df_test['Diff_WTeam'].fillna(OTHER_NAN)

df_test['Diff_LTeam'] = df_test['Diff_LTeam'].fillna(OTHER_NAN)



df_test['WTeam_PerCent'] = count_wt_win / (count_wt_win + count_wt_lose)

df_test['LTeam_PerCent'] = count_lt_win / (count_lt_win + count_lt_lose)



df_test['WTeam_PerCent'] = df_test['WTeam_PerCent'].fillna(OTHER_NAN)

df_test['LTeam_PerCent'] = df_test['LTeam_PerCent'].fillna(OTHER_NAN)
features = df_train.columns



category_features_names = ['Season', 'DayNum', 'WLoc', 'WTeamID', 'LTeamID']



df_train = df_train.fillna(OTHER_NAN)

df_test = df_test.fillna(OTHER_NAN)



x1 = df_train.shape[0]

df_train_inverse = df_train.copy()



for i in range(len(W_FEATURES)):

    v_w = W_FEATURES[i]

    v_l = L_FEATURES[i]

    df_train_inverse[v_w] = df_train[v_l]

    df_train_inverse[v_l] = df_train[v_w]



df_train_inverse['WLoc'] = df_train_inverse['WLoc'].apply(set_WLoc)

df_train = df_train.append(df_train_inverse, ignore_index=True)



X_train = df_train[features]



X_test = df_test[features]



x0 = df_train_inverse.shape[0]



y1 = np.ones((x1,), dtype=int)

y0 = np.zeros((x0,), dtype=int)



Y = np.concatenate((y1, y0), axis=None)

Y_df = pd.DataFrame(Y)

Y = Y_df



X_train[category_features_names] = X_train[category_features_names].astype('int64').astype('category')

X_test[category_features_names] = X_test[category_features_names].astype('int64').astype('category')

final_encoding = 1

cat_features = []



if final_encoding==0: # all data encoded with TE, cat_features empty

    X_train = X_train.applymap(lambda x: str(x))

    X_test = X_test.applymap(lambda x: str(x))

    te = ce.TargetEncoder(smoothing=0.2)

    te.fit(X_train, Y)

    X_train = te.transform(X_train, Y)

    X_test = te.transform(X_test)

elif final_encoding==1: # TE only on category features, cat_features = empty

    te = ce.TargetEncoder(cols=category_features_names, smoothing=0.2)

    te.fit(X_train, Y)

    X_train = te.transform(X_train, Y)

    X_test = te.transform(X_test)

    

    # cat_features = category_features_names

elif final_encoding==2:

    non_cat = [cn for cn in X_train.columns if cn not in category_features_names]

    

    X_train_numeric = X_train[non_cat].applymap(lambda x: str(x))

    X_test_numeric = X_test[non_cat].applymap(lambda x: str(x))



    te = ce.TargetEncoder(smoothing=0.2)

    te.fit(X_train_numeric, Y)

    X_train_numeric = te.transform(X_train_numeric, Y)

    X_test_numeric = te.transform(X_test_numeric)

    

    X_train[non_cat] = X_train_numeric

    X_test.update(X_test_numeric)

    cat_features = category_features_names

elif final_encoding==3: # all data without cat features

    X_train = X_train.drop(['Season', 'DayNum', 'WTeamID', 'LTeamID'], axis=1)

    X_test = X_test.drop(['Season', 'DayNum', 'WTeamID', 'LTeamID'], axis=1)

    #cat_features = ['WLoc']

else:

    cat_features = category_features_names



X = X_train

X_testset= X_test

Y_train = Y


names = [

         "Ridge",

         "RidgeCV",

         "XGB_Regressor", 

         "GBC_Classifier",

         "GBC_Regressor",

         "HGBC_Classifier",

         "HGBC_Regressor",

         "ETC_Classifier",

         "ETC_Regressor",

         "LDA",

         "QDA",

         "DecisionTree",

         "RandomForest_Classifier",

         "RandomForest_Regressor",

         "AdaBoost_Classifier",

         "AdaBoost_Regressor",

         "LogisticRegression",

         "CatBoost_Classifier",

         "CatBoost_Regressor",

         "Huber_Regressor",

         "Theil_Regressor"

    ]



classifiers = [

        RidgeClassifier(),

        RidgeClassifierCV(),

        XGBRegressor(),

        GradientBoostingClassifier(verbose=0),

        GradientBoostingRegressor(verbose=0),

        HistGradientBoostingClassifier(verbose=0),

        HistGradientBoostingRegressor(verbose=0),

        ExtraTreesClassifier(verbose=0),

        ExtraTreesRegressor(verbose=0),

        LinearDiscriminantAnalysis(),

        QuadraticDiscriminantAnalysis(),

        DecisionTreeClassifier(max_depth=5),

        RandomForestClassifier(max_depth=5, n_estimators=500, verbose=0),

        RandomForestRegressor(max_depth=5, n_estimators=500, verbose=0),

        AdaBoostClassifier(),

        AdaBoostRegressor(),

        LogisticRegression(max_iter=10000, verbose=0),

        CatBoostClassifier(**PARAMS_CATBOOST),

        CatBoostRegressor(**PARAMS_CATBOOST_REGRESSOR),

        HuberRegressor(),

        TheilSenRegressor(verbose=False)

    ]





kf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)

format_data = 'df'



for name, clf in zip(names, classifiers):

    print("Classifier "+name)

        

    test_preds = 0

    test_score = 0

    train_score = 0

    count = 0

    

    for train_index, test_index in kf.split(X, Y):

        count = count+1

        #print("Split "+str(count)+" ... ")

        

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        

        if name in ["CatBoost_Classifier", "CatBoost_Regressor"]:

            train_dataset = Pool(data=X_train, label=y_train, cat_features=cat_features)

            eval_dataset = Pool(data=X_test, label=y_test, cat_features=cat_features)

            clf.fit(train_dataset, use_best_model=True, eval_set=[eval_dataset]) # Get predicted classes

            print("Count of trees in model = {}".format(clf.tree_count_))

        else:

            clf.fit(X_train, y_train.values.ravel())

        

        if name in ["XGB_Regressor", "Ridge", "RidgeCV", "HGBC_Regressor", "GBC_Regressor", "ETC_Regressor", "CatBoost_Regressor", "RandomForest_Regressor", "AdaBoost_Regressor", "Huber_Regressor", "Theil_Regressor"]:

            y_train_predict = clf.predict(X_train)

            y_test_predict = clf.predict(X_test)

            y_pred_proba = clf.predict(X_testset) 

        else:

            y_train_predict = clf.predict_proba(X_train)[:,0]

            y_test_predict = clf.predict_proba(X_test)[:,0]

            y_pred_proba = clf.predict_proba(X_testset)[:,0]

            

        if name in ["Ridge", "RidgeCV", "HuberRegressor", "TheilSenRegressor"]:

            y_train_predict = y_train_predict / float(10)

            y_test_predict = y_test_predict / float(10)

            y_pred_proba = y_pred_proba / float(10)

        

        '''

        # Cross validation, save the model to disk, for each split

        filename = 'model_ALL_'+str(SPLITS)+'_splits_'+name+'_'+str(count)+'.sav'

        pickle.dump(clf, open(filename, 'wb'))

        

        y_test_predict = y_test_predict.reshape(-1, 1)

        y01 = y_test.to_numpy().reshape((y_test.shape[0], 1))

        p = log_loss(y01, y_test_predict)

        

        y_train_predict = y_train_predict.reshape(-1, 1)

        y01 = y_train.to_numpy().reshape((y_train.shape[0], 1))

        pp = log_loss(y01, y_train_predict)

        

        # Coss validation, print score for each split:

        print("Score Test : "+str(p))

        print("Score Train : "+str(pp))

        

        # Generate submission for the split

        print_submission_into_file(y_pred_proba, df_test_id, "_ALL_"+str(name)+'_'+str(SPLITS)+'_splits_'+str(count))

        '''

        

        test_preds += y_pred_proba/float(SPLITS)

        

    # Generate submission for the whole data:

    df = print_submission_into_file(test_preds, df_test_id, "_"+str(name))

    

    # DataFrame labels : 

    # ID of the format of ID in the submission file and 

    # Label with 1 if WTeam wins and 0 otherwise

    

    labels_good = labels["Label"]

    

    df_predict = df[df["ID"].isin(labels["Concats"])]

    predictions = df_predict["Pred"]

    p11 = log_loss(labels_good.astype('float').to_numpy(), predictions.astype('float').to_numpy())

    print("Score : "+str(p11))

   
