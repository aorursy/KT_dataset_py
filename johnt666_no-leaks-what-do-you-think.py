import pandas as pd

import numpy as np

import seaborn as sn
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold



def light_gbm_try(X,y,X_test):

    kf = KFold(n_splits=200,shuffle=True,random_state=42)



    predictions_lgb=pd.DataFrame()

    preds = pd.DataFrame()



    f1=[]

    i=1

    importances=pd.DataFrame()

    importances['Features']=X.columns

    for a,b in kf.split(X,y):

        X_tr=X.iloc[a,:]

        X_te=X.iloc[b,:]

        y_train=y[a]

        y_test=y[b]



        train_data=lgb.Dataset(X_tr, y_train)

        test_data=lgb.Dataset(X_te, y_test, reference=train_data)



        print('---------- Training fold Nº {} ----------'.format(i))



        params = {'num_leaves': 7,

             'min_data_in_leaf': 20,

             'objective': 'binary',

             'max_depth': 20,

             #'colsample_bytree':0.2,

             'learning_rate': 0.1,

             'boosting': 'gbdt',

             'bagging_freq': 5,

             'bagging_fraction': 1,

             'feature_fraction': 1,

             'bagging_seed': 11,

             'random_state': 42,

             'metric': 'binary_logloss',

             'verbosity': -1,}



        model = lgb.train(params,train_data,num_boost_round=10000,valid_sets = [test_data],

                          verbose_eval=1000,early_stopping_rounds = 300)

        #model.fit(X=X_train,y=y_train,eval_set=(X_test,y_test),verbose=100,early_stopping_rounds=300)

        predictions_lgb[str(i)]=model.predict(X_test,num_iterations=model.best_iteration)



        name = 'importance_'+str(i)

        importances[name]=model.feature_importance()



        f1.append(list(model.best_score.items())[0][1]['binary_logloss'])



        i+=1

        #gc.collect()

    print('MEAN F1 LIGHTGBM: {}'.format(np.mean(f1)))

        

    return np.mean(f1), predictions_lgb, importances
!ls
#Data Section 1 

Tourney_Results=pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')

sub=pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WSampleSubmissionStage1_2020.csv')
Tourney_Results.head()
winers=Tourney_Results[['WTeamID','WScore']].rename(columns={'WTeamID':'Team','WScore':'Score'})

losers=Tourney_Results[['LTeamID','LScore']].rename(columns={'LTeamID':'Team','LScore':'Score'})
total_scores=pd.concat([winers,losers])

total_scores.head()
aggs=['mean','max','min']
scores_agg=total_scores.groupby('Team').agg(aggs)

scores_agg.columns = scores_agg.columns.map('_'.join)

scores_agg=scores_agg.reset_index()

scores_agg.head()


train=Tourney_Results[['Season','WTeamID','LTeamID']].copy()

train.rename(columns={'WTeamID':'Team1','LTeamID':'Team2'},inplace=True)

train['Pred']=1

train.head()
train=pd.merge(train,scores_agg,how='left',left_on='Team1',right_on='Team').drop('Team',axis=1)

train.rename(columns={'Score_mean':'Score_mean1','Score_max':'Score_max1','Score_min':'Score_min1'},inplace=True)

train=pd.merge(train,scores_agg,how='left',left_on='Team2',right_on='Team').drop('Team',axis=1)

train.rename(columns={'Score_mean':'Score_mean2','Score_max':'Score_max2','Score_min':'Score_min2'},inplace=True)

train.head()
train_loser=train.copy()

train_loser['Team1']=train['Team2'].copy()

train_loser['Team2']=train['Team1'].copy()

train_loser['Score_mean1']=train['Score_mean2'].copy()

train_loser['Score_max1']=train['Score_max2'].copy()

train_loser['Score_min1']=train['Score_min2'].copy()

train_loser['Score_mean2']=train['Score_mean1'].copy()

train_loser['Score_max2']=train['Score_max1'].copy()

train_loser['Score_min2']=train['Score_min1'].copy()

train_loser['Pred']=0

train_loser.head()
sub = pd.concat([sub, sub['ID'].str.split('_', expand=True).rename(columns={0: 'Season', 1: 'Team1', 2: 'Team2'}).astype(np.int64)], axis=1)

sub.head()
sub=pd.merge(sub,scores_agg,how='left',left_on='Team1',right_on='Team').drop('Team',axis=1)

sub.rename(columns={'Score_mean':'Score_mean1','Score_max':'Score_max1','Score_min':'Score_min1'},inplace=True)

sub=pd.merge(sub,scores_agg,how='left',left_on='Team2',right_on='Team').drop('Team',axis=1)

sub.rename(columns={'Score_mean':'Score_mean2','Score_max':'Score_max2','Score_min':'Score_min2'},inplace=True)

sub.head()
sub.Season.unique()
train.Season.unique()
train=pd.concat([train,train_loser],axis=0)

train=train[train['Season']<2015] 

train.shape
X=train.drop('Pred',axis=1).reset_index().drop('index',axis=1)

a,b,c=light_gbm_try(X,y=train.Pred.values,X_test=sub.drop(['Pred','ID'],axis=1))
sub['Pred']=b.mean(axis=1)

sub.drop(['Season','Team1','Team2'],axis=1,inplace=True)

sub.head()
sub.to_csv('try_taco_alone_resetdrop.csv',index=False)
c['importance']=c.drop('Features',axis=1).mean(axis=1)
c[['Features','importance']].sort_values(by='importance',ascending=False)