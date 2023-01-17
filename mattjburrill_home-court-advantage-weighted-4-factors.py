import numpy as np 

import pandas as pd

import os

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#import data

in_path = '../input/'

RegularSeasonDetailedResults = pd.read_csv(in_path + 'wdatafiles/WRegularSeasonDetailedResults.csv')

NCAATourneyCompactResults = pd.read_csv(in_path + 'wdatafiles/WNCAATourneyCompactResults.csv')



RegularSeasonDetailedResults.head()
def Get4FactorStats(df):

    

    df ['WEFG']       = (df ['WFGM'] + (df ['WFGM3']*0.5))/df ['WFGA']

    df ['LEFG']       = (df ['LFGM'] + (df ['LFGM3']*0.5))/df ['LFGA']

    df ['WTOV']       = df ['WTO']/((df ['WFGA'] + 0.44) + (df ['WFTA']+df ['WTO']))

    df ['LTOV']       = df ['LTO']/((df ['LFGA'] + 0.44) + (df ['LFTA']+df ['LTO']))

    df ['WORB']       = df ['WOR']/(df ['WOR'] + df ['LDR'])

    df ['LORB']       = df ['LOR']/(df ['LOR'] + df ['WDR'])

    df ['WFTAR']      = df ['WFTA']/(df ['WFGA'])

    df ['LFTAR']      = df ['LFTA']/(df ['LFGA'])

    

    return df





#Get 4 factors on the game level 

reg_season_4_factor = Get4FactorStats(RegularSeasonDetailedResults)



#Split into two datasets of the winners and loosers because you want every team to have a line for a single game.

reg_season_4_factor_w = reg_season_4_factor[['Season','WTeamID','DayNum','WEFG','WTOV','WORB','WFTAR']]





reg_season_4_factor_w = reg_season_4_factor_w.rename(columns = {

                                               'WEFG': 'EFG',

                                               'WTOV': 'TOV',

                                               'WORB': 'ORB',

                                               'WFTAR': 'FTAR',

                                               'WLoc': 'Loc',

                                               'WTeamID':'TeamID'})





reg_season_4_factor_l = reg_season_4_factor[['Season','LTeamID','DayNum','LEFG','LTOV','LORB','LFTAR']]





reg_season_4_factor_l = reg_season_4_factor_l.rename(columns = {

                                               'LEFG': 'EFG',

                                               'LTOV': 'TOV',

                                               'LORB': 'ORB',

                                               'LFTAR': 'FTAR',

                                               'LTeamID':'TeamID'})









#Set the data back together

reg_season_4_factor_set = (reg_season_4_factor_w, reg_season_4_factor_l)

reg_season_4_factor_set = pd.concat(reg_season_4_factor_set, ignore_index = True)

reg_season_4_factor_set.drop_duplicates()



#Group by so we get the avg 4 factor results at the season team level.

reg_season_4_factor_overall_avg =  reg_season_4_factor_set.groupby(['Season','TeamID'])[['EFG','TOV','ORB','FTAR']].mean().reset_index()



reg_season_4_factor_overall_avg.head()



#Get the outcome of the tournament games.

def NCAASetWinAndLoseTeamsRecords(NCAATourneyCompactResults):

    NCAA_res_w = NCAATourneyCompactResults.rename(columns = {'WTeamID': 'NCAA_TEAMID',

                                                           'LTeamID': 'NCAA_O_TEAMID', 

                                                           'WScore':'NCAA_SCORE',

                                                           'LScore':'NCAA_O_SCORE'

                                                             })

    NCAA_res_l = NCAATourneyCompactResults.rename(columns = {'LTeamID': 'NCAA_TEAMID',

                                                           'WTeamID': 'NCAA_O_TEAMID', 

                                                           'LScore':'NCAA_SCORE',

                                                           'WScore':'NCAA_O_SCORE'

                                                             })

        

    NCAA_Ses = (NCAA_res_w, NCAA_res_l)

    NCAA_Ses = pd.concat(NCAA_Ses, ignore_index = True,sort = True)

    NCAA_Ses ['OUTCOME'] = np.where(NCAA_Ses['NCAA_SCORE']>NCAA_Ses['NCAA_O_SCORE'], 1, 0)

    NCAA_Ses = NCAA_Ses[['Season','NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME']]

    return NCAA_Ses



Tourney_Results = NCAASetWinAndLoseTeamsRecords(NCAATourneyCompactResults)



Tourney_Results.head()
#merge 4 factor stats on tournament-outcome data.

Tourney_Results_1 = pd.merge(Tourney_Results, reg_season_4_factor_overall_avg, how='inner', 

                    left_on=['Season','NCAA_TEAMID'],

                    right_on=['Season','TeamID'] )







Tourney_Results_2 = pd.merge(Tourney_Results_1, reg_season_4_factor_overall_avg, how='inner', 

                    left_on=['Season','NCAA_O_TEAMID'],

                    right_on=['Season','TeamID'],suffixes = ['','_op'] )







Non_weighted_4_factors = Tourney_Results_2[['Season','NCAA_TEAMID','NCAA_O_TEAMID','OUTCOME','EFG','TOV','ORB','FTAR','EFG_op','TOV_op','ORB_op','FTAR_op']]



Non_weighted_4_factors.head()


def print_score(m):

                

    print ("train log loss :", metrics.log_loss(y_train.tolist(),m.predict_proba(X_train).tolist(), eps=1e-15))

    print ("test log loss :", metrics.log_loss(y_valid.tolist(),m.predict_proba(X_valid).tolist(), eps=1e-15))

    

    if hasattr(m, 'oob_score_'): print ("oob_score : ", m.oob_score_)

    





train = Non_weighted_4_factors[Non_weighted_4_factors.Season <= 2017]

valid = Non_weighted_4_factors[Non_weighted_4_factors.Season == 2018]





X_train = train.drop(['Season', 'NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME'], axis=1)

y_train = train['OUTCOME']

X_valid = valid.drop(['Season', 'NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME'], axis=1)

y_valid = valid['OUTCOME']





m = RandomForestClassifier(n_estimators=500, n_jobs=-1, oob_score=True, random_state=0)

m.fit(X_train, y_train)

print_score(m)

#get 4 factors on the game level 

reg_season_h = Get4FactorStats(RegularSeasonDetailedResults)



#split into two datasets of the winners and loosers because you want every

# team to have a line for a single game.  one game = two lines, one for each team

reg_season_h_w = reg_season_h[['Season','WTeamID','DayNum','WEFG','WTOV','WORB','WFTAR','WLoc']].copy()





reg_season_h_w_1 = reg_season_h_w.loc[reg_season_h_w.WLoc == 'N', 'WLoc'] = 'A'





reg_season_h_w = reg_season_h_w.rename(columns = {

                                               'WEFG': 'EFG',

                                               'WTOV': 'TOV',

                                               'WORB': 'ORB',

                                               'WFTAR': 'FTAR',

                                               'WLoc': 'Loc',

                                               'WTeamID':'TeamID'})





reg_season_h_l = reg_season_h[['Season','LTeamID','DayNum','LEFG','LTOV','LORB','LFTAR','WLoc']].copy()

#Home vs away is opposite for the LTeamID



reg_season_h_l.loc[reg_season_h_l.WLoc == 'H', 'Loc'] = 'A'

reg_season_h_l.loc[reg_season_h_l.WLoc == 'A', 'Loc'] = 'H'

reg_season_h_l.loc[reg_season_h_l.WLoc == 'N', 'Loc'] = 'A'





reg_season_h_l = reg_season_h_l.drop(['WLoc'], axis=1)



reg_season_h_l = reg_season_h_l.rename(columns = {

                                               'LEFG': 'EFG',

                                               'LTOV': 'TOV',

                                               'LORB': 'ORB',

                                               'LFTAR': 'FTAR',

                                               'LTeamID':'TeamID'})

#set the data back together

reg_season_4_factor_h_a = (reg_season_h_w, reg_season_h_l)

reg_season_4_factor_h_a = pd.concat(reg_season_4_factor_h_a, ignore_index = True)

reg_season_4_factor_h_a.drop_duplicates()



#group by for calculating home stats. 

reg_season_4_factor_home_avg =  reg_season_4_factor_h_a.groupby(['Season','TeamID','Loc'])[['EFG','TOV','ORB','FTAR']].mean().reset_index()



reg_season_4_factor_home_avg = reg_season_4_factor_home_avg.drop(reg_season_4_factor_home_avg[reg_season_4_factor_home_avg.Loc == 'A'].index)





reg_season_4_factor_home_avg = reg_season_4_factor_home_avg.rename(columns = {

                                               'EFG': 'EFG_home',

                                               'TOV': 'TOV_home',

                                               'ORB': 'ORB_home',

                                               'FTAR': 'FTAR_home'

                                               })

reg_season_4_factor_home_avg = reg_season_4_factor_home_avg.drop(['Loc'],axis=1)



#group by for calculating whole season stats

reg_season_4_factor_overall_avg =  reg_season_4_factor_h_a.groupby(['Season','TeamID'])[['EFG','TOV','ORB','FTAR']].mean().reset_index()



reg_season_4_factor_overall_avg = reg_season_4_factor_overall_avg.rename(columns = {

                                               'EFG': 'EFG_overall',

                                               'TOV': 'TOV_overall',

                                               'ORB': 'ORB_overall',

                                               'FTAR': 'FTAR_overall'

                                               })





reg_season_4_factor_before_calc = pd.merge(reg_season_4_factor_home_avg, reg_season_4_factor_overall_avg, how='inner', 

                    left_on=['Season','TeamID'],

                    right_on=['Season','TeamID'] )





#calc for the weights on the team season level. This calculation comes from the article mentioned in the intro.

reg_season_4_factor_before_calc ['EFG_weight'] = (reg_season_4_factor_before_calc ['EFG_home']/reg_season_4_factor_before_calc ['EFG_overall'])**0.5

reg_season_4_factor_before_calc ['TOV_weight'] = (reg_season_4_factor_before_calc ['TOV_home']/reg_season_4_factor_before_calc ['TOV_overall'])**0.5

reg_season_4_factor_before_calc ['ORB_weight'] = (reg_season_4_factor_before_calc ['ORB_home']/reg_season_4_factor_before_calc ['ORB_overall'])**0.5

reg_season_4_factor_before_calc ['FTAR_weight'] = (reg_season_4_factor_before_calc ['FTAR_home']/reg_season_4_factor_before_calc ['FTAR_overall'])**0.5







#dataset that will be joined to full data set.

home_weights = reg_season_4_factor_before_calc [['Season','TeamID','EFG_home','TOV_home','ORB_home','FTAR_home']]



#merge weights in

Non_weighted_4_factors_with_weights = pd.merge(Non_weighted_4_factors,home_weights , how='inner', 

                    left_on=['Season','NCAA_TEAMID'],

                    right_on=['Season','TeamID'] )



Non_weighted_4_factors_with_weights = pd.merge(Non_weighted_4_factors_with_weights,home_weights , how='inner', 

                    left_on=['Season','NCAA_O_TEAMID'],

                    right_on=['Season','TeamID'], suffixes =['', '_op'] )



Non_weighted_4_factors_with_weights.head()
#find home/away.



NCAA_res_w = NCAATourneyCompactResults.rename(columns = {'WTeamID': 'NCAA_TEAMID',

                                                           'LTeamID': 'NCAA_O_TEAMID', 

                                                           'WScore':'NCAA_SCORE',

                                                           'LScore':'NCAA_O_SCORE'

                                                             })

NCAA_res_l = NCAATourneyCompactResults.rename(columns = {'LTeamID': 'NCAA_TEAMID',

                                                           'WTeamID': 'NCAA_O_TEAMID', 

                                                           'LScore':'NCAA_SCORE',

                                                           'WScore':'NCAA_O_SCORE'

                                                             })

        

NCAA_Ses = (NCAA_res_w, NCAA_res_l)

NCAA_Ses = pd.concat(NCAA_Ses, ignore_index = True,sort = True)

NCAA_Ses ['OUTCOME'] = np.where(NCAA_Ses['NCAA_SCORE']>NCAA_Ses['NCAA_O_SCORE'], 1, 0)

home_game = NCAA_Ses[['Season','NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME','WLoc']]



home_game.head()




#merge WLoc in (Home,Away,Neutral) 



Weighted_4_factor = pd.merge(Non_weighted_4_factors_with_weights,home_game , how='inner', 

                    left_on=['Season','NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME'],

                    right_on=['Season','NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME'] )



#multiply the four factor by the  teams's home weights that we calculated above.

Weighted_4_factor.loc[(Weighted_4_factor.WLoc == 'H') &(Weighted_4_factor.OUTCOME == 1), 'EFG'] = Weighted_4_factor['EFG'] * Weighted_4_factor['EFG_home']

Weighted_4_factor.loc[(Weighted_4_factor.WLoc == 'H') &(Weighted_4_factor.OUTCOME == 1), 'TOV'] = Weighted_4_factor['TOV'] * Weighted_4_factor['TOV_home']

Weighted_4_factor.loc[(Weighted_4_factor.WLoc == 'H') &(Weighted_4_factor.OUTCOME == 1), 'ORB'] = Weighted_4_factor['EFG'] * Weighted_4_factor['ORB_home']

Weighted_4_factor.loc[(Weighted_4_factor.WLoc == 'H') &(Weighted_4_factor.OUTCOME == 1), 'FTAR'] = Weighted_4_factor['TOV'] * Weighted_4_factor['FTAR_home']



Weighted_4_factor.loc[(Weighted_4_factor.WLoc == 'H') &(Weighted_4_factor.OUTCOME == 0), 'EFG_op'] = Weighted_4_factor['EFG_op'] * Weighted_4_factor['EFG_home_op']

Weighted_4_factor.loc[(Weighted_4_factor.WLoc == 'H') &(Weighted_4_factor.OUTCOME == 0), 'TOV_op'] = Weighted_4_factor['TOV_op'] * Weighted_4_factor['TOV_home_op']

Weighted_4_factor.loc[(Weighted_4_factor.WLoc == 'H') &(Weighted_4_factor.OUTCOME == 0), 'EFG_op'] = Weighted_4_factor['EFG_op'] * Weighted_4_factor['ORB_home_op']

Weighted_4_factor.loc[(Weighted_4_factor.WLoc == 'H') &(Weighted_4_factor.OUTCOME == 0), 'TOV_op'] = Weighted_4_factor['TOV_op'] * Weighted_4_factor['FTAR_home_op']







Weighted_4_factor = Weighted_4_factor.drop(['EFG_home', 'TOV_home', 

                 'ORB_home', 'FTAR_home', 'EFG_home_op', 'TOV_home_op',

                 'ORB_home_op', 'FTAR_home_op', 'WLoc', 'TeamID_op','TeamID'], axis=1)



Weighted_4_factor.head()
train = Weighted_4_factor[Weighted_4_factor.Season <= 2017]

valid = Weighted_4_factor[Weighted_4_factor.Season == 2018]





X_train = train.drop(['Season', 'NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME'], axis=1)

y_train = train['OUTCOME']

X_valid = valid.drop(['Season', 'NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME'], axis=1)

y_valid = valid['OUTCOME']





m = RandomForestClassifier(n_estimators=500, n_jobs=-1, oob_score=True, random_state=0)

m.fit(X_train, y_train)

print_score(m)