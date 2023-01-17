import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

import os
print(os.listdir("../input"))
def import_training_data(nrows=None):
    df=pd.read_csv('../input/train_V2.csv', sep=',', encoding='utf-8', nrows=nrows)
    df=df.drop(['rankPoints'],axis=1)

    return df

def import_test_data(nrows=None):
    df_test=pd.read_csv('../input/test_V2.csv', sep=',', encoding='utf-8', nrows=nrows)
    df_test=df_test.drop(['rankPoints'],axis=1)

    return df_test
df=import_training_data()
df_squad=df[(df['matchType'].str.contains('squad')) | (df['matchType'].str.contains('flare'))].drop('matchType',axis=1)
df_duo=df[df['matchType'].str.contains('duo')].drop('matchType',axis=1)
df_solo=df[df['matchType'].str.contains('solo')].drop(columns=['DBNOs','revives']).drop('matchType',axis=1)
df_crash=df[df['matchType'].str.contains('crash')].drop('matchType',axis=1).drop('killPoints', axis=1).drop('winPoints', axis=1) # all values are 0
df_test=import_test_data()
df_test_squad=df_test[(df_test['matchType'].str.contains('squad')) | (df_test['matchType'].str.contains('flare'))].drop('matchType',axis=1)
df_test_duo=df_test[df_test['matchType'].str.contains('duo')].drop('matchType',axis=1)
df_test_solo=df_test[df_test['matchType'].str.contains('solo')].drop('matchType',axis=1).drop(columns=['DBNOs','revives'])
df_test_crash=df_test[df_test['matchType'].str.contains('crash')].drop('matchType',axis=1).drop('killPoints', axis=1).drop('winPoints', axis=1) # all values are 0
len_df_test=len(df_test)
del df
del df_test
def combine_training_teams(df):
    # split up test DataFrame according to matchId's
    matches = list(dict(tuple(df.groupby('matchId',axis=0))).values())
    
    # reduce teams to one entry per team by averaging their values
    matches_combined_teams=[]
    for i in matches:
        num_memb=i['groupId'].value_counts(sort=False).to_frame().reset_index().rename(columns={"index": "groupId", "groupId": "numMembers"})
        df_red_avg_tmp=i.groupby('groupId').agg('mean').sort_values(by='groupId')
        df_red_avg_tmp.sort_values(by='groupId', inplace=True)
        df_red_avg_tmp['numMembers']=num_memb.sort_values(by='groupId')['numMembers'].values
        df_red_avg_tmp=df_red_avg_tmp.drop(['matchDuration','numGroups'],axis=1)

        matches_combined_teams.append(df_red_avg_tmp.reset_index())
        
    return matches_combined_teams

def combine_testing_teams(df):
    # split up test DataFrame according to matchId's
    matches_test = list(dict(tuple(df.groupby('matchId',axis=0))).values())

    # Make dictionary of Id's and groupId's to reassign winPlacePerc to their respective Id's
#     groupid_to_id=[]
#     for i in matches_test:

    # reduce teams to one entry per team by averaging their values
    matches_combined_teams_test=[]
    for i in matches_test:
        dict_groupid_to_id={}  # {'Id' : 'groupId'}
        for idx, row in i.iterrows():
            dict_groupid_to_id[row['Id']]=row['groupId']
#         groupid_to_id.append(dict_groupid_to_id)
        
        num_memb_tmp=i['groupId'].value_counts(sort=False).to_frame().reset_index().rename(columns={"index": "groupId", "groupId": "numMembers"})
        df_red_avg_tmp=i.groupby('groupId').agg('mean').sort_values(by='groupId')
        df_red_avg_tmp.sort_values(by='groupId', inplace=True)
        df_red_avg_tmp['numMembers']=num_memb_tmp.sort_values(by='groupId')['numMembers'].values
        df_red_avg_tmp=df_red_avg_tmp.drop(['matchDuration','numGroups'],axis=1)

        matches_combined_teams_test.append([df_red_avg_tmp.reset_index(),dict_groupid_to_id])
        
    return matches_combined_teams_test
def filter_small_teams(list_of_matches):
    df_small_matches_1=[]
    small_matches_2=[]
    rest=[]
    for i in list_of_matches:
        if len(i)<2:
            df_small_matches_1.append(i)
        elif len(i)>=2 and len(i)<3:
            small_matches_2.append(i)
        else:
            rest.append(i)
    return df_small_matches_1, small_matches_2, rest

def filter_small_teams_test(list_of_matches):
    df_small_matches_1=[]
    small_matches_2=[]
    rest=[]
    for df, dic in list_of_matches:
        if len(df)<2:
            df_small_matches_1.append([df,dic])
        elif len(df)>=2 and len(df)<3:
            small_matches_2.append([df,dic])
        else:
            rest.append([df,dic])
    return df_small_matches_1, small_matches_2, rest
def prepare_pca(matches_combined_teams, in_components=None):
    print(len(matches_combined_teams[1].drop(['winPlacePerc'],axis=1).columns))
    ipca = IncrementalPCA(n_components=in_components)#, whiten=True)
    for i in matches_combined_teams:
        ipca.partial_fit(i.drop(['winPlacePerc'],axis=1).values)
    print(ipca.explained_variance_ratio_)
    print(ipca.n_components_)
    
    return ipca

def prepare_pca_input(df_train, df_test, dfeatures=None):
    # training data
    #df_train_in=df_train.drop(['winPlacePerc'],axis=1).reindex(sorted(df_train.drop(['winPlacePerc'],axis=1).columns), axis=1)
    df_train_in=[]
    df_train_groupids=[]
    for i in df_train:
        df_train_groupids.append(i[['groupId','maxPlace']])
        if len(dfeatures)>0:
            df_train_in.append(i.drop(['groupId','maxPlace'],axis=1).drop(dfeatures,axis=1).reindex(sorted(i.drop(['groupId','maxPlace'],axis=1).drop(dfeatures,axis=1).columns), axis=1))
        else:
            df_train_in.append(i.drop(['groupId','maxPlace'],axis=1).reindex(sorted(i.drop(['groupId','maxPlace'],axis=1).columns), axis=1))

    # test data
    df_test_in=[]
    df_test_groupids=[]
    for df, dic in df_test:
        df_test_groupids.append([df[['groupId','maxPlace']],dic])
        if len(dfeatures)>0:
            df_test_in.append(df.drop(['groupId','maxPlace'],axis=1).drop(dfeatures,axis=1).reindex(sorted(df.drop(['groupId','maxPlace'],axis=1).drop(dfeatures,axis=1).columns), axis=1))
        else:
            df_test_in.append(df.drop(['groupId','maxPlace'],axis=1).reindex(sorted(df.drop(['groupId','maxPlace'],axis=1).columns), axis=1))

    return df_train_in, df_train_groupids, df_test_in, df_test_groupids

def pca_transform_data(matches_combined_teams, matches_combined_teams_test, ipca):
    x_train_pca = []
    for i in matches_combined_teams:
        x_train_pca.append(ipca.transform(i.drop('winPlacePerc',axis=1).values))

    y_train_pca=[]
    for i in matches_combined_teams:
        y_train_pca.append(i['winPlacePerc'].values.reshape(-1, 1))

    x_test_pca = []
    for i in matches_combined_teams_test:
        x_test_pca.append(ipca.transform(i.values))
        
    return x_train_pca, y_train_pca, x_test_pca
def scale_data(x_train_pca, x_test_pca):
    scaler = MinMaxScaler()
    
    for i in x_train_pca:
        scaler.partial_fit(i)
    x_train_scaled=[]
    for i in x_train_pca:
        x_train_scaled.append(scaler.transform(i))

    x_test_scaled=[]
    for i in x_test_pca:
        x_test_scaled.append(scaler.transform(i))
        
    return x_train_scaled, x_test_scaled
def mlp_evaluate(x_train, y_train, df_train_groupids, training_level=100, verbose=False):
    working_sample=np.random.choice(len(x_train), training_level*5)
    
    mlp=MLPRegressor(solver='adam',hidden_layer_sizes=(50,50,))    
    
    train_counter=0
    lenxtrain=len(x_train)
    kfold_score=[]
    
    for k in range(training_level):
        for i in range(4):
            mlp.partial_fit(x_train[working_sample[train_counter]], y_train[working_sample[train_counter]].ravel())
            train_counter+=1

        y_pred_kfold=pd.DataFrame(mlp.predict(x_train[working_sample[train_counter]]), columns=['ranks'])

        y_pred_kfold=y_pred_kfold.join(df_train_groupids[working_sample[train_counter]]).sort_values(by='ranks', ascending=False)
        y_test=pd.DataFrame(y_train[working_sample[train_counter]], columns=['winPlacePerc']).join(df_train_groupids[working_sample[train_counter]]).sort_values(by='groupId', ascending=True)

        tmp_winPlacePerc=[]
        for i in range(1,len(y_pred_kfold)+1):
            tmp_winPlacePerc.append([y_pred_kfold['groupId'].values[i-1],(len(y_pred_kfold)-i)/(len(y_pred_kfold)-1)])

        tmp_winPlacePerc=pd.DataFrame(tmp_winPlacePerc, columns=['groupId','winPlacePerc']).sort_values(by='groupId', ascending=True)
        kfold_score.append(mean_absolute_error(y_test['winPlacePerc'],tmp_winPlacePerc['winPlacePerc']))
        train_counter+=1
        
        if verbose:
            print("Evaluating "+str(k)+"/"+str(training_level))
            print("Current mean bsolut error: "+str(np.mean(kfold_score)))

    print("Average mean absolute error: "+str(np.mean(kfold_score)))
    
    return kfold_score
def mlp_train(x_train, y_train):
    mlp=MLPRegressor(solver='adam',hidden_layer_sizes=(50,50,))
    lenxtrain=len(x_train)
    for i in range(lenxtrain):
        mlp.partial_fit(x_train[i], y_train[i].ravel())
    
    return mlp
def predict_winPlacePerc(x_test, df_test_groupids, mlp, verbose=False):
    winPlacePerc=[]
    num_matches=len(x_test)
    for k in range(num_matches):
        if verbose and k%100==0:
            print("Predicting match "+str(k)+"/"+str(num_matches))
            
        y_pred=pd.DataFrame(mlp.predict(x_test[k]), columns=['ranks'])

        y_pred=y_pred.join(df_test_groupids[k][0]).sort_values(by='ranks', ascending=False)

        tmp_winPlacePerc=[]
        for i in range(1,len(y_pred)+1):
#             tmp_winPlacePerc.append([y_pred['groupId'].values[i-1],(y_pred['maxPlace'].values[i-1]-i)/(y_pred['maxPlace'].values[i-1]-1)])
            tmp_winPlacePerc.append([y_pred['groupId'].values[i-1],(len(y_pred)-i)/(len(y_pred)-1)])

        winPlacePerc.append(pd.DataFrame(tmp_winPlacePerc, columns=['groupId','winPlacePerc']).sort_values(by='groupId', ascending=True))

    result_list=[]
    counter=0
    for df, el in df_test_groupids:
        match_results=[]
        for i, gid in el.items():
            wpp=winPlacePerc[counter][winPlacePerc[counter]['groupId']==gid]['winPlacePerc'].values
            match_results.append([i,wpp])
        result_list.append(pd.DataFrame(match_results, columns=['Id','winPlacePerc']))
        counter+=1

    winPlacePerc_tot=pd.concat(result_list)

    winPlacePerc_tot['winPlacePerc']=winPlacePerc_tot['winPlacePerc'].str[0]
    
    return winPlacePerc_tot
def predict_one_team_matches(df):
    result_list=[]
    counter=0
    for df, dic in df:
        match_results=[]
        for i, gid in dic.items():
            match_results.append([i,0.0])
        result_list.append(pd.DataFrame(match_results, columns=['Id','winPlacePerc']))
        counter+=1
    if len(result_list)>0:
        return pd.concat(result_list)
    else:
        return []

def predict_two_team_matches(df):
    result_list=[]
    counter=0
    for df, dic in df:
        match_results=[]
        killpointsMax=np.max([df['killPlace'].values[0],df['killPlace'].values[1]])
        maxKillpointsGId=df[df['killPlace']==killpointsMax]['groupId'].values[0]
        for i, gid in dic.items():
            if maxKillpointsGId in gid:
                match_results.append([i,0.0])
            else:
                match_results.append([i,1.0])
        result_list.append(pd.DataFrame(match_results, columns=['Id','winPlacePerc']))
        counter+=1
    if len(result_list)>0:
        return pd.concat(result_list)
    else:
        return []
df_red_squad=combine_training_teams(df_squad)
df_red_squad_1, df_red_squad_2, df_red_squad=filter_small_teams(df_red_squad)
df_test_per_match_squad = combine_testing_teams(df_test_squad)
df_test_per_match_squad_1, df_test_per_match_squad_2, df_test_per_match_squad=filter_small_teams_test(df_test_per_match_squad)
df_squad.drop(['killPoints','maxPlace','roadKills','teamKills','vehicleDestroys','winPoints','swimDistance'],axis=1).corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
del df_squad
del df_test_squad
drop_features_squad=['killPoints','roadKills','teamKills','vehicleDestroys','winPoints']
df_train_in_squad, df_train_groupids_squad, df_test_in_squad, df_test_groupids_squad = prepare_pca_input(df_red_squad, df_test_per_match_squad, drop_features_squad)
pca_squad=prepare_pca(df_train_in_squad, in_components=None)
x_train_pca_squad, y_train_pca_squad, x_test_pca_squad=pca_transform_data(df_train_in_squad, df_test_in_squad, pca_squad)
del df_red_squad
del df_test_in_squad
del df_train_in_squad
x_train_scaled_squad, x_test_scaled_squad=scale_data(x_train_pca_squad, x_test_pca_squad)
eval_score_squad=mlp_evaluate(x_train_scaled_squad, y_train_pca_squad, df_train_groupids_squad, training_level=10000, verbose=False)
mlp_squad=mlp_train(x_train_scaled_squad, y_train_pca_squad)
winPlacePerc_one_team_squad=predict_one_team_matches(df_test_per_match_squad_1)
winPlacePerc_two_teams_squad=predict_two_team_matches(df_test_per_match_squad_2)
winPlacePerc_squad = predict_winPlacePerc(x_test_scaled_squad, df_test_groupids_squad, mlp_squad, verbose=False)
winPlacePerc_squad = winPlacePerc_squad.append(winPlacePerc_two_teams_squad)
winPlacePerc_squad.head()
df_red_duo=combine_training_teams(df_duo)
df_red_duo_1, df_red_duo_2, df_red_duo=filter_small_teams(df_red_duo)
df_test_per_match_duo = combine_testing_teams(df_test_duo)
df_test_per_match_duo_1, df_test_per_match_duo_2, df_test_per_match_duo=filter_small_teams_test(df_test_per_match_duo)
df_duo.drop(['killPoints','roadKills','teamKills','vehicleDestroys','winPoints'],axis=1).corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
del df_duo
del df_test_duo
drop_features_duo=['killPoints','roadKills','teamKills','vehicleDestroys','winPoints']
df_train_in_duo, df_train_groupids_duo, df_test_in_duo, df_test_groupids_duo = prepare_pca_input(df_red_duo, df_test_per_match_duo, dfeatures=drop_features_duo)
pca_duo=prepare_pca(df_train_in_duo, in_components=None)
x_train_pca_duo, y_train_pca_duo, x_test_pca_duo=pca_transform_data(df_train_in_duo, df_test_in_duo, pca_duo)
del df_red_duo
del df_test_in_duo
del df_train_in_duo
x_train_scaled_duo, x_test_scaled_duo=scale_data(x_train_pca_duo, x_test_pca_duo)
eval_score_duo=mlp_evaluate(x_train_scaled_duo, y_train_pca_duo, df_train_groupids_duo, training_level=10000, verbose=False)
mlp_duo=mlp_train(x_train_scaled_duo, y_train_pca_duo)
winPlacePerc_one_team_duo=predict_one_team_matches(df_test_per_match_duo_1)
winPlacePerc_two_teams_duo=predict_two_team_matches(df_test_per_match_duo_2)
winPlacePerc_duo = predict_winPlacePerc(x_test_scaled_duo, df_test_groupids_duo, mlp_duo, verbose=False)
winPlacePerc_duo=winPlacePerc_duo.append(winPlacePerc_one_team_duo)
winPlacePerc_duo.head()
df_solo['winPlacePerc'].fillna(0.0, inplace=True)
df_red_solo=combine_training_teams(df_solo)
df_red_solo_1, df_red_solo_2, df_red_solo=filter_small_teams(df_red_solo)
df_test_per_match_solo = combine_testing_teams(df_test_solo)
df_test_per_match_solo_1, df_test_per_match_solo_2, df_test_per_match_solo=filter_small_teams_test(df_test_per_match_solo)
df_solo.drop(['killPoints','roadKills','teamKills','winPoints'],axis=1).corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
del df_solo
del df_test_solo
drop_features_solo=['killPoints','roadKills','teamKills','winPoints']
df_train_in_solo, df_train_groupids_solo, df_test_in_solo, df_test_groupids_solo = prepare_pca_input(df_red_solo, df_test_per_match_solo, dfeatures=drop_features_solo)
pca_solo=prepare_pca(df_train_in_solo, in_components=None)
x_train_pca_solo, y_train_pca_solo, x_test_pca_solo=pca_transform_data(df_train_in_solo, df_test_in_solo, pca_solo)
del df_red_solo
del df_test_in_solo
del df_train_in_solo
x_train_scaled_solo, x_test_scaled_solo=scale_data(x_train_pca_solo, x_test_pca_solo)
eval_score_solo=mlp_evaluate(x_train_scaled_solo, y_train_pca_solo, df_train_groupids_solo, training_level=10000, verbose=False)
mlp_solo=mlp_train(x_train_scaled_solo, y_train_pca_solo)
winPlacePerc_one_teams_solo=predict_one_team_matches(df_test_per_match_solo_1)
winPlacePerc_two_teams_solo=predict_two_team_matches(df_test_per_match_solo_2)
winPlacePerc_solo = predict_winPlacePerc(x_test_scaled_solo, df_test_groupids_solo, mlp_solo, verbose=False)
winPlacePerc_solo = winPlacePerc_solo.append(winPlacePerc_one_teams_solo).append(winPlacePerc_two_teams_solo)
winPlacePerc_solo.head()
df_red_crash=combine_training_teams(df_crash)
df_red_crash_1, df_red_crash_2, df_red_crash=filter_small_teams(df_red_crash)
df_test_per_match_crash = combine_testing_teams(df_test_crash)
df_test_per_match_crash_1, df_test_per_match_crash_2, df_test_per_match_crash=filter_small_teams_test(df_test_per_match_crash)
df_crash.drop(['headshotKills','numGroups','swimDistance','teamKills'],axis=1).corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
# df_test_per_match_crash[0].corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
del df_crash
del df_test_crash
drop_features_crash=['headshotKills','swimDistance','teamKills']
df_train_in_crash, df_train_groupids_crash, df_test_in_crash, df_test_groupids_crash = prepare_pca_input(df_red_crash, df_test_per_match_crash, dfeatures=drop_features_crash)
pca_crash=prepare_pca(df_train_in_crash, in_components=None)
x_train_pca_crash, y_train_pca_crash, x_test_pca_crash=pca_transform_data(df_train_in_crash, df_test_in_crash, pca_crash)
del df_red_crash
del df_test_in_crash
del df_train_in_crash
x_train_scaled_crash, x_test_scaled_crash=scale_data(x_train_pca_crash, x_test_pca_crash)
eval_score_crash=mlp_evaluate(x_train_scaled_crash, y_train_pca_crash, df_train_groupids_crash, training_level=len(x_train_scaled_crash), verbose=False)
mlp_crash=mlp_train(x_train_scaled_crash, y_train_pca_crash)
winPlacePerc_one_teams_crash=predict_one_team_matches(df_test_per_match_crash_1)
winPlacePerc_two_teams_crash=predict_two_team_matches(df_test_per_match_crash_2)
winPlacePerc_crash = predict_winPlacePerc(x_test_scaled_crash, df_test_groupids_crash, mlp_crash, verbose=False)
winPlacePerc_crash.head()
eval_score=np.concatenate((np.array(eval_score_squad),np.array(eval_score_duo),np.array(eval_score_solo),np.array(eval_score_crash)))
eval_score_avg=np.mean(eval_score)
print("Evaluation mean absolute error: "+str(eval_score_avg))
df_sub=winPlacePerc_squad.append(winPlacePerc_duo).append(winPlacePerc_solo).append(winPlacePerc_crash)
df_sub=df_sub.reset_index().drop('index',axis=1)
df_sub.to_csv('submission.csv', index=False)