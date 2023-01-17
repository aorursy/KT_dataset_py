# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tourney_results=pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')
regular_results=pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')
regular_results.head()
def prepare_data(df):

    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT', 

    'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 

    'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]



    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'

    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'

    df.columns.values[6] = 'location'

    dfswap.columns.values[6] = 'location'    

      

    df.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(df.columns)]

    dfswap.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(dfswap.columns)]



    output = pd.concat([df, dfswap]).reset_index(drop=True)

    output.loc[output.location=='N','location'] = '0'

    output.loc[output.location=='H','location'] = '1'

    output.loc[output.location=='A','location'] = '-1'

    output.location = output.location.astype(int)

    

    output['PointDiff'] = output['T1_Score'] - output['T2_Score']

    

    return output
regular_data=prepare_data(regular_results)

tourney_data=prepare_data(tourney_results)
tourney_data.loc[0:1115,'Result']=1

tourney_data.loc[1115:2230,'Result']=0
stat_cols=[col for col in regular_data.columns if col not in ['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID', 'T2_Score',

       'location', 'NumOT']]
stat_seasons=regular_data.groupby(['Season','T1_TeamID'])[stat_cols].agg([np.mean]).reset_index()
stat_seasons
stat_seasons.columns.values
stat_seasons.columns=[''.join(col).strip() for col in stat_seasons.columns.values]
stat_seasons.head()
stat_seasons_T1=stat_seasons.copy()

stat_seasons_T2=stat_seasons.copy()
stat_seasons_T1.columns=['T1'+ x.replace('T1','').replace('T2','oppponent') for x in stat_seasons_T1.columns]

stat_seasons_T2.columns=['T2'+ x.replace('T1','').replace('T2','oppponent') for x in stat_seasons_T2.columns]

stat_seasons_T1.columns.values[0]="Season"

stat_seasons_T2.columns.values[0]="Season"
stat_seasons_T1.head()
tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score','Result']]

tourney_data=pd.merge(tourney_data,stat_seasons_T1,on=['Season','T1_TeamID'],how='left')

tourney_data=pd.merge(tourney_data,stat_seasons_T2,on=['Season','T2_TeamID'],how='left')

seeds_data=pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')
seeds_data.head()
def extra(x):

    if len(x)==3:

        return -1

    elif x[3]=='a':

        return 1

    else:

        return 0
seeds_data['seed']=seeds_data['Seed'].apply(lambda x : int(x[1:3]))
seeds_data['seed_2']=seeds_data['Seed'].apply(extra)
seeds_T1 = seeds_data[['Season','TeamID','seed','seed_2']].copy()

seeds_T2 = seeds_data[['Season','TeamID','seed','seed_2']].copy()
seeds_T1.columns = ['Season','T1_TeamID','T1_seed','T1_seed_2']

seeds_T2.columns = ['Season','T2_TeamID','T2_seed','T2_seed_2']
seeds_T2.head()
tourney_data = pd.merge(tourney_data, seeds_T1, on = ['Season', 'T1_TeamID'], how = 'left')

tourney_data = pd.merge(tourney_data, seeds_T2, on = ['Season', 'T2_TeamID'], how = 'left')
tourney_data["Seed_diff"] = tourney_data["T1_seed"].astype(int) - tourney_data["T2_seed"].astype(int)
tourney_data.dtypes
tourney_data.columns
features=[col for col in tourney_data.columns.values if col not in ['Season', 'DayNum', 'T1_TeamID', 'T2_TeamID','Result','T1_Score','T2_Score']]
len(features)
y = tourney_data['Result']

X = tourney_data[features].values
import xgboost
dtrain=xgboost.DMatrix(X,label=y)
def cauchyobj(preds, dtrain):

    labels = dtrain.get_label()

    c = 5000 

    x =  preds-labels    

    grad = x / (x**2/c**2+1)

    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2

    return grad, hess
param = {} 

#param['objective'] = 'reg:linear'

param['eval_metric'] =  'mae'

param['booster'] = 'gbtree'

param['eta'] = 0.05 #change to ~0.02 for final run

param['subsample'] = 0.35

param['colsample_bytree'] = 0.7

param['num_parallel_tree'] = 3 #recommend 10

param['min_child_weight'] = 40

param['gamma'] = 10

param['max_depth'] =  3

param['silent'] = 1
from sklearn.model_selection import KFold
xgb_cv = []

repeat_cv = 10 # recommend 10



for i in range(repeat_cv): 

    print(f"Fold repeater {i}")

    xgb_cv.append(

        xgboost.cv(

          params = param,

          dtrain = dtrain,

          obj = cauchyobj,

          num_boost_round = 3000,

          folds = KFold(n_splits = 5, shuffle = True, random_state = i),

          early_stopping_rounds = 25,

          verbose_eval = 50

        )

    )
iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]

val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]

iteration_counts, val_mae
oof_preds = []

for i in range(10):

    print(f"Fold repeater {i}")

    preds = y.copy()

    kfold = KFold(n_splits = 5, shuffle = True, random_state = i+100)    

    for train_index, val_index in kfold.split(X,y):

 

        dtrain_i = xgboost.DMatrix(X[train_index], label = y[train_index])

        dval_i = xgboost.DMatrix(X[val_index], label = y[val_index])  

        model = xgboost.train(

              params = param,

              dtrain = dtrain_i,

              num_boost_round = iteration_counts[i],

              verbose_eval = 50

        )

        preds[val_index] = model.predict(dval_i)

    oof_preds.append(preds)
oof_preds[0]
sub=pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
sub['Season'] = sub['ID'].map(lambda x: int(x[:4]))

sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)

sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)


sub = pd.merge(sub, stat_seasons_T1, on = ['Season', 'T1_TeamID'])

sub = pd.merge(sub, stat_seasons_T2, on = ['Season', 'T2_TeamID'])

sub = pd.merge(sub, seeds_T1, on = ['Season', 'T1_TeamID'])

sub = pd.merge(sub, seeds_T2, on = ['Season', 'T2_TeamID'])

sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]
Xsub=sub[features].values

dtest = xgboost.DMatrix(Xsub)
dtest = xgboost.DMatrix(Xsub)

sub_models = []

for i in range(repeat_cv):

    print(f"Fold repeater {i}")

    sub_models.append(

        xgboost.train(

          params = param,

          dtrain = dtrain,

          num_boost_round = int(iteration_counts[i] * 1.05),

          verbose_eval = 50

        )

    )

    

sub_preds = []

for i in range(repeat_cv):

    sub_preds.append(sub_models[i].predict(dtest))

    

sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)

sub[['ID','Pred']]
sub_preds = []

for i in range(repeat_cv):

    sub_preds.append(sub_models[i].predict(dtest))

    

sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)

sub[['ID','Pred']].to_csv("submission.csv", index = False)

tourney_data.to_csv('tourney_data.csv',index=False)