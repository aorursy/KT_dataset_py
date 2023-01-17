import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
directory = '/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/'

tourney_result = pd.read_csv(directory+'MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

tourney_seed = pd.read_csv(directory+'MDataFiles_Stage1/MNCAATourneySeeds.csv')
# deleting unnecessary columns

tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

def get_seed(x):

    return int(x[1:3])



tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))

tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))

season_result = pd.read_csv(directory+'MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
season_win_result = season_result[['Season', 'WTeamID', 'WScore']]

season_lose_result = season_result[['Season', 'LTeamID', 'LScore']]

season_win_result.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)

season_lose_result.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)

season_result = pd.concat((season_win_result, season_lose_result)).reset_index(drop=True)

season_score = season_result.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()

tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Score':'WScoreT'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_result = pd.merge(tourney_result, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

tourney_result.rename(columns={'Score':'LScoreT'}, inplace=True)

tourney_result = tourney_result.drop('TeamID', axis=1)

tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)

tourney_win_result.rename(columns={'WSeed':'Seed1', 'LSeed':'Seed2', 'WScoreT':'ScoreT1', 'LScoreT':'ScoreT2'}, inplace=True)

tourney_lose_result = tourney_win_result.copy()

tourney_lose_result['Seed1'] = tourney_win_result['Seed2']

tourney_lose_result['Seed2'] = tourney_win_result['Seed1']

tourney_lose_result['ScoreT1'] = tourney_win_result['ScoreT2']

tourney_lose_result['ScoreT2'] = tourney_win_result['ScoreT1']

tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']

tourney_win_result['ScoreT_diff'] = tourney_win_result['ScoreT1'] - tourney_win_result['ScoreT2']

tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']

tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']
tourney_win_result['result'] = 1

tourney_lose_result['result'] = 0

tourney_result = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)

test_df = pd.read_csv(directory+'MSampleSubmissionStage1_2020.csv')
test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))

test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))

test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))

test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Seed':'Seed1'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Seed':'Seed2'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, season_score, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, season_score, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))

test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))

test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']

test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']

test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)

X = tourney_result.drop('result', axis=1)

y = tourney_result.result
X.head()
test_df.head()
for a in X.columns:

    print(a,X[a].isnull().sum())
ss = StandardScaler()

ss.fit(pd.concat([X[X.columns],test_df[X.columns]],sort=False))

X[X.columns] = ss.transform(X[X.columns])

test_df[X.columns] = ss.transform(test_df[X.columns])
def Output(p):

    return 1.0/(1.+np.exp(-p))



def GPI(data):

    return Output(  0.100000*np.tanh((((((((((13.17023181915283203)) * ((((-((data["Seed_diff"])))) * 2.0)))) - (((data["Seed1"]) + (data["ScoreT2"]))))) * 2.0)) - (np.cos(((-((data["Seed_diff"])))))))) +

                    0.100000*np.tanh(((((((((((data["ScoreT_diff"]) + (((((data["Seed_diff"]) * ((-3.0)))) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) + (data["Seed_diff"]))) +

                    0.100000*np.tanh((((((14.09049892425537109)) * (((((-((((((np.cos((data["Seed2"]))) + (data["ScoreT_diff"]))) * 2.0))))) + ((((14.09049892425537109)) * ((-((data["Seed_diff"])))))))/2.0)))) * 2.0)) +

                    0.100000*np.tanh(((data["Seed2"]) + (((((data["ScoreT_diff"]) + ((((((((((-((data["Seed_diff"])))) + (((np.cos((data["Seed2"]))) / 2.0)))) * 2.0)) * 2.0)) * 2.0)))) * 2.0)))) +

                    0.100000*np.tanh(((((((((data["ScoreT_diff"]) + (((((((data["Seed2"]) + (((-0.994007))))) - (data["Seed1"]))) * 2.0)))) * 2.0)) - (((data["Seed2"]) + (data["Seed1"]))))) * 2.0)))



def GPII(data):

    return Output(  0.100000*np.tanh(((((((((((((((data["ScoreT_diff"]) + ((((((-2.0)) * (data["Seed_diff"]))) * 2.0)))) * 2.0)) - (data["Seed_diff"]))) * 2.0)) - (data["Seed_diff"]))) * 2.0)) * 2.0)) +

                    0.100000*np.tanh(((data["Seed_diff"]) * (((data["ScoreT_diff"]) + ((((((data["Seed2"]) <= (np.cos((((data["Seed1"]) * 2.0)))))*1.)) - ((((3.0)) * 2.0)))))))) +

                    0.100000*np.tanh(((((((((((((data["Seed_diff"]) * (((((-1.411442))) * 2.0)))) * 2.0)) - (((np.sin((data["ScoreT_diff"]))) * 2.0)))) * 2.0)) * 2.0)) - (np.sin((data["Seed_diff"]))))) +

                    0.100000*np.tanh((((((((((0.501484))) + (data["Seed_diff"]))/2.0)) + ((((-((((data["Seed_diff"]) - ((((((0.411693))) + (data["ScoreT_diff"]))/2.0))))))) * ((12.19133663177490234)))))) * ((10.94865894317626953)))) +

                    0.100000*np.tanh(((((((((((((data["ScoreT_diff"]) - ((((np.cos((data["Seed2"]))) > (data["Seed2"]))*1.)))) - (data["Seed_diff"]))) * 2.0)) - (data["Seed_diff"]))) * 2.0)) - (data["Seed1"]))))

def GPIII(data):

    return Output(  0.100000*np.tanh(((((((((np.where(data["Seed2"] <= (-((data["Seed_diff"]))), ((((-1.411442))) - ((2.0))), 0 )) * (((data["Seed_diff"]) * 2.0)))) - (data["Seed_diff"]))) * 2.0)) * 2.0)) +

                    0.100000*np.tanh(((np.sin((np.minimum(((data["ScoreT2"])), ((((data["Seed2"]) - (data["Seed_diff"])))))))) + (((((((data["Seed1"]) - ((11.37786579132080078)))) - ((11.37786579132080078)))) * (data["Seed_diff"]))))) +

                    0.100000*np.tanh(((data["Seed_diff"]) * (((((-(((13.00282287597656250))))) + (((((((((np.cos((((data["Seed1"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) + (((data["ScoreT1"]) * 2.0)))))/2.0)))) +

                    0.100000*np.tanh((((((((((((-((data["Seed_diff"])))) - (data["Seed_diff"]))) + (np.tanh(((((data["Seed2"]) <= (data["Seed_diff"]))*1.)))))) + (data["ScoreT_diff"]))) * 2.0)) * 2.0)) +

                    0.100000*np.tanh((((((((((((((((((((((-3.0)) * (data["Seed_diff"]))) + (data["ScoreT_diff"]))) + (((-1.435063))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))

def GPIV(data):

    return Output(  0.100000*np.tanh((((((-3.0)) * ((((9.0)) + (data["ScoreT1"]))))) * (np.minimum(((data["Seed_diff"])), (((((data["ScoreT1"]) <= ((((5.0)) + (((-1.221382))))))*1.))))))) +

                    0.100000*np.tanh(((((((data["ScoreT_diff"]) * (data["ScoreT_diff"]))) * (data["ScoreT_diff"]))) - (((data["Seed_diff"]) * ((4.93693542480468750)))))) +

                    0.100000*np.tanh(((((((((((((((((data["ScoreT_diff"]) - (((data["Seed_diff"]) * 2.0)))) * 2.0)) * 2.0)) - (data["Seed_diff"]))) * 2.0)) * (data["Seed_diff"]))) * (data["Seed_diff"]))) * 2.0)) +

                    0.100000*np.tanh(((((((np.minimum(((data["Seed2"])), (((((((-((data["Seed_diff"])))) * 2.0)) * (data["Seed_diff"])))))) * (((((data["Seed_diff"]) * 2.0)) * 2.0)))) * 2.0)) * 2.0)) +

                    0.100000*np.tanh((((((((data["Seed_diff"]) * 2.0)) > ((((data["ScoreT1"]) + (((((data["Seed_diff"]) * 2.0)) * (data["ScoreT1"]))))/2.0)))*1.)) - (((((data["Seed_diff"]) * 2.0)) * 2.0)))))

def GPV(data):

    return Output(  0.100000*np.tanh(((((((((data["ScoreT_diff"]) + (np.tanh(((((-3.0)) * (((((((((data["Seed_diff"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)))))))) - (data["Seed_diff"]))) * 2.0)) * 2.0)) +

                    0.100000*np.tanh((((((((-3.0)) * (np.cos(((((-3.0)) - (data["Seed1"]))))))) - ((((-3.0)) * (((((data["Seed_diff"]) * ((-3.0)))) * 2.0)))))) * 2.0)) +

                    0.100000*np.tanh(((((((((((((((((((((data["ScoreT_diff"]) / 2.0)) - (data["Seed_diff"]))) + (((-0.152974))))) * 2.0)) * 2.0)) + (((-0.152974))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +

                    0.100000*np.tanh(((((data["Seed_diff"]) * ((((-2.0)) - (np.sin(((((((-3.0)) - (((((data["Seed2"]) * ((-3.0)))) * 2.0)))) / 2.0)))))))) * 2.0)) +

                    0.100000*np.tanh((((((((((((-2.0)) * (data["Seed_diff"]))) + (data["ScoreT_diff"]))) + ((((((data["Seed2"]) - (data["Seed_diff"]))) <= (data["ScoreT_diff"]))*1.)))) * 2.0)) * 2.0)))

def GPVI(data):

    return Output(  0.100000*np.tanh(np.real(((np.where(np.abs(((((data["Seed2"]) - (data["Seed1"]))) * (complex(14.38402175903320312)))) >= np.abs(((((data["ScoreT1"]) + (data["ScoreT_diff"]))) + (data["Seed1"]))),((((data["Seed2"]) - (data["Seed1"]))) * (complex(14.38402175903320312))), ((((data["ScoreT1"]) + (data["ScoreT_diff"]))) + (data["Seed1"])) )) * 2.0))) +

                    0.100000*np.tanh(np.real(np.where(np.abs(((data["ScoreT_diff"]) - (((complex(-3.0)) * (data["Seed_diff"]))))) >= np.abs(((complex(3.0)) * (((complex(-3.0)) * (data["Seed_diff"]))))),((data["ScoreT_diff"]) - (((complex(-3.0)) * (data["Seed_diff"])))), ((complex(3.0)) * (((complex(-3.0)) * (data["Seed_diff"])))) ))) +

                    0.100000*np.tanh(np.real((((((((((((((-((((data["Seed_diff"]) + (((((((complex(0.026572)) * 2.0)) * 2.0)) / (data["Seed2"])))))))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0))) +

                    0.100000*np.tanh(np.real(((((np.where(np.abs(data["ScoreT_diff"]) <= np.abs(((complex(7.0)) * (np.sin((data["Seed_diff"]))))),data["ScoreT_diff"], ((complex(7.0)) * (np.sin((data["Seed_diff"])))) )) - (np.sinh((data["Seed_diff"]))))) - (np.sinh((np.sinh((np.sinh((data["Seed_diff"])))))))))) +

                    0.100000*np.tanh(np.real(np.where(np.abs(((data["Seed_diff"]) - (np.where(np.abs(complex(-0.665339)) > np.abs(data["ScoreT2"]),complex(1.), complex(0.) )))) >= np.abs(((np.where(np.abs(data["Seed2"]) >= np.abs((((-((data["Seed_diff"])))) - (data["Seed1"]))),data["Seed2"], (((-((data["Seed_diff"])))) - (data["Seed1"])) )) - (data["Seed_diff"]))),((data["Seed_diff"]) - (np.where(np.abs(complex(-0.665339)) > np.abs(data["ScoreT2"]),complex(1.), complex(0.) ))), ((np.where(np.abs(data["Seed2"]) >= np.abs((((-((data["Seed_diff"])))) - (data["Seed1"]))),data["Seed2"], (((-((data["Seed_diff"])))) - (data["Seed1"])) )) - (data["Seed_diff"])) ))))



def GPVII(data):

    return Output(  0.100000*np.tanh(np.real(((((((((((((np.where(np.abs(data["ScoreT_diff"]) > np.abs(complex(-0.665339)), data["ScoreT_diff"], complex(0.) )) - (data["Seed_diff"]))) * 2.0)) - (data["ScoreT_diff"]))) - (data["Seed_diff"]))) * 2.0)) * (complex(7.19583606719970703))))) +

                    0.100000*np.tanh(np.real(((data["Seed_diff"]) * (((((-((((np.where(np.abs(complex(-1.411442)) <= np.abs(data["ScoreT2"]), np.cosh((complex(7.86849117279052734))), complex(0.) )) + (np.cosh((complex(-3.0))))))))) + (data["ScoreT2"]))/2.0))))) +

                    0.100000*np.tanh(np.real((((-((np.sinh((data["Seed2"])))))) + ((((((((-((data["Seed1"])))) + (np.sinh((np.sinh((data["Seed2"]))))))) * 2.0)) + (data["ScoreT_diff"])))))) +

                    0.100000*np.tanh(np.real(np.where(np.abs(((data["Seed_diff"]) * 2.0)) > np.abs(complex(-0.361695)), ((complex(0.411693)) + (((((((data["ScoreT_diff"]) + ((((-((data["Seed_diff"])))) * 2.0)))) * 2.0)) * 2.0))), complex(0.) ))) +

                    0.100000*np.tanh(np.real(((np.where(np.abs(((complex(0.355140)) * (np.sinh((data["ScoreT_diff"]))))) >= np.abs(np.where(np.abs((((-((data["Seed1"])))) * 2.0)) >= np.abs(np.sinh((data["Seed2"]))),(((-((data["Seed1"])))) * 2.0), np.sinh((data["Seed2"])) )),((complex(0.355140)) * (np.sinh((data["ScoreT_diff"])))), np.where(np.abs((((-((data["Seed1"])))) * 2.0)) >= np.abs(np.sinh((data["Seed2"]))),(((-((data["Seed1"])))) * 2.0), np.sinh((data["Seed2"])) ) )) - (data["Seed_diff"])))))



def GPVIII(data):

    return Output(  0.100000*np.tanh(np.real(((np.where(np.abs(((np.where(np.abs(((complex(0,1)*np.conjugate(data["Seed_diff"])) * (complex(0,1)*np.conjugate(complex(14.20872020721435547))))) >= np.abs(np.sinh((data["ScoreT_diff"]))),((complex(0,1)*np.conjugate(data["Seed_diff"])) * (complex(0,1)*np.conjugate(complex(14.20872020721435547)))), np.sinh((data["ScoreT_diff"])) )) * 2.0)) >= np.abs(np.sinh((complex(0,1)*np.conjugate(complex(14.20872020721435547))))),((np.where(np.abs(((complex(0,1)*np.conjugate(data["Seed_diff"])) * (complex(0,1)*np.conjugate(complex(14.20872020721435547))))) >= np.abs(np.sinh((data["ScoreT_diff"]))),((complex(0,1)*np.conjugate(data["Seed_diff"])) * (complex(0,1)*np.conjugate(complex(14.20872020721435547)))), np.sinh((data["ScoreT_diff"])) )) * 2.0), np.sinh((complex(0,1)*np.conjugate(complex(14.20872020721435547)))) )) * 2.0))) +

                    0.100000*np.tanh(np.real(((((((np.where(np.abs(((((((data["ScoreT_diff"]) - (data["Seed_diff"]))) - (data["Seed_diff"]))) * 2.0)) >= np.abs((((-((complex(0.355140))))) / (data["Seed_diff"]))),((((((data["ScoreT_diff"]) - (data["Seed_diff"]))) - (data["Seed_diff"]))) * 2.0), (((-((complex(0.355140))))) / (data["Seed_diff"])) )) * 2.0)) * 2.0)) * 2.0))) +

                    0.100000*np.tanh(np.real(((((data["Seed_diff"]) - (((np.where(np.abs(data["ScoreT2"]) <= np.abs(data["Seed2"]),complex(1.), complex(0.) )) + (np.where(np.abs(data["ScoreT2"]) <= np.abs(data["ScoreT1"]),complex(1.), complex(0.) )))))) - (((complex(9.63826560974121094)) * (data["Seed_diff"])))))) +

                    0.100000*np.tanh(np.real(np.where(np.abs((((-((data["Seed_diff"])))) * (np.cosh((((data["Seed2"]) * (complex(8.06685638427734375)))))))) >= np.abs(np.cos((np.cosh((((data["Seed2"]) * (complex(8.06685638427734375)))))))),(((-((data["Seed_diff"])))) * (np.cosh((((data["Seed2"]) * (complex(8.06685638427734375))))))), np.cos((np.cosh((((data["Seed2"]) * (complex(8.06685638427734375))))))) ))) +

                    0.100000*np.tanh(np.real(np.where(np.abs(complex(-1.435063)) <= np.abs(((data["Seed_diff"]) * (((complex(-3.0)) * 2.0)))), ((((((((data["ScoreT_diff"]) - (np.sinh((data["Seed_diff"]))))) * 2.0)) * 2.0)) - (data["Seed_diff"])), complex(0.) ))))



def GPIX(data):

    return Output(  0.100000*np.tanh(np.real(((np.where(np.abs(((np.where(np.abs(np.where(np.abs(np.sqrt((complex(-3.0)))) > np.abs(complex(0.0)),complex(1.), complex(0.) )) > np.abs(data["ScoreT1"]),complex(1.), complex(0.) )) / (data["Seed1"]))) >= np.abs(((np.sinh((complex(-3.0)))) * (data["Seed_diff"]))),((np.where(np.abs(np.where(np.abs(np.sqrt((complex(-3.0)))) > np.abs(complex(0.0)),complex(1.), complex(0.) )) > np.abs(data["ScoreT1"]),complex(1.), complex(0.) )) / (data["Seed1"])), ((np.sinh((complex(-3.0)))) * (data["Seed_diff"])) )) * 2.0))) +

                    0.100000*np.tanh(np.real(((((((((((((np.where(np.abs(np.tanh((data["Seed1"]))) <= np.abs(data["Seed_diff"]), data["ScoreT_diff"], complex(0.) )) - (((data["Seed_diff"]) * 2.0)))) * 2.0)) * 2.0)) * 2.0)) - (data["Seed1"]))) * 2.0))) +

                    0.100000*np.tanh(np.real(np.where(np.abs(((data["ScoreT_diff"]) / (complex(-0.452108)))) >= np.abs(np.sinh((np.where(np.abs(np.where(np.abs(((((data["Seed_diff"]) * 2.0)) / (complex(-0.452108)))) >= np.abs(np.sinh((data["ScoreT_diff"]))),((((data["Seed_diff"]) * 2.0)) / (complex(-0.452108))), np.sinh((data["ScoreT_diff"])) )) >= np.abs(complex(-0.452108)),np.where(np.abs(((((data["Seed_diff"]) * 2.0)) / (complex(-0.452108)))) >= np.abs(np.sinh((data["ScoreT_diff"]))),((((data["Seed_diff"]) * 2.0)) / (complex(-0.452108))), np.sinh((data["ScoreT_diff"])) ), complex(-0.452108) )))),((data["ScoreT_diff"]) / (complex(-0.452108))), np.sinh((np.where(np.abs(np.where(np.abs(((((data["Seed_diff"]) * 2.0)) / (complex(-0.452108)))) >= np.abs(np.sinh((data["ScoreT_diff"]))),((((data["Seed_diff"]) * 2.0)) / (complex(-0.452108))), np.sinh((data["ScoreT_diff"])) )) >= np.abs(complex(-0.452108)),np.where(np.abs(((((data["Seed_diff"]) * 2.0)) / (complex(-0.452108)))) >= np.abs(np.sinh((data["ScoreT_diff"]))),((((data["Seed_diff"]) * 2.0)) / (complex(-0.452108))), np.sinh((data["ScoreT_diff"])) ), complex(-0.452108) ))) ))) +

                    0.100000*np.tanh(np.real(np.sinh((np.sinh((np.where(np.abs(((data["Seed_diff"]) / (complex(-0.452108)))) >= np.abs(np.where(np.abs(complex(0,1)*np.conjugate(np.sin((np.cosh((data["ScoreT_diff"])))))) >= np.abs(data["ScoreT_diff"]),complex(0,1)*np.conjugate(np.sin((np.cosh((data["ScoreT_diff"]))))), data["ScoreT_diff"] )),((data["Seed_diff"]) / (complex(-0.452108))), np.where(np.abs(complex(0,1)*np.conjugate(np.sin((np.cosh((data["ScoreT_diff"])))))) >= np.abs(data["ScoreT_diff"]),complex(0,1)*np.conjugate(np.sin((np.cosh((data["ScoreT_diff"]))))), data["ScoreT_diff"] ) ))))))) +

                    0.100000*np.tanh(np.real(np.where(np.abs(np.where(np.abs(np.sinh((data["ScoreT_diff"]))) >= np.abs(((complex(-3.0)) * (np.sinh((data["Seed_diff"]))))),np.sinh((data["ScoreT_diff"])), ((complex(-3.0)) * (np.sinh((data["Seed_diff"])))) )) >= np.abs(np.sqrt((complex(-3.0)))),np.where(np.abs(np.sinh((data["ScoreT_diff"]))) >= np.abs(((complex(-3.0)) * (np.sinh((data["Seed_diff"]))))),np.sinh((data["ScoreT_diff"])), ((complex(-3.0)) * (np.sinh((data["Seed_diff"])))) ), np.sqrt((complex(-3.0))) ))))



def GPX(data):

    return Output(  0.100000*np.tanh(np.real(((((data["Seed_diff"]) / (np.where(np.abs(np.where(np.abs(complex(-0.079116)) <= np.abs(((data["Seed_diff"]) / (np.sinh((complex(0.499892)))))),complex(-0.079116), ((data["Seed_diff"]) / (np.sinh((complex(0.499892))))) )) <= np.abs(((data["Seed_diff"]) / (np.sinh((data["ScoreT_diff"]))))),np.where(np.abs(complex(-0.079116)) <= np.abs(((data["Seed_diff"]) / (np.sinh((complex(0.499892)))))),complex(-0.079116), ((data["Seed_diff"]) / (np.sinh((complex(0.499892))))) ), ((data["Seed_diff"]) / (np.sinh((data["ScoreT_diff"])))) )))) * 2.0))) +

                    0.100000*np.tanh(np.real(np.where(np.abs(((((data["Seed_diff"]) * (((((complex(-3.0)) - (complex(1.368404)))) + (np.where(np.abs(data["Seed_diff"]) <= np.abs(complex(-0.513372)), complex(1.368404), complex(0.) )))))) + (data["ScoreT_diff"]))) >= np.abs(data["ScoreT_diff"]),((((data["Seed_diff"]) * (((((complex(-3.0)) - (complex(1.368404)))) + (np.where(np.abs(data["Seed_diff"]) <= np.abs(complex(-0.513372)), complex(1.368404), complex(0.) )))))) + (data["ScoreT_diff"])), data["ScoreT_diff"] ))) +

                    0.100000*np.tanh(np.real(((((np.where(np.abs(np.where(np.abs(data["Seed2"]) >= np.abs(data["ScoreT_diff"]),data["Seed2"], data["ScoreT_diff"] )) >= np.abs(((((data["Seed_diff"]) / (complex(-0.260113)))) - (((np.tanh((data["Seed2"]))) - (data["ScoreT_diff"]))))),np.where(np.abs(data["Seed2"]) >= np.abs(data["ScoreT_diff"]),data["Seed2"], data["ScoreT_diff"] ), ((((data["Seed_diff"]) / (complex(-0.260113)))) - (((np.tanh((data["Seed2"]))) - (data["ScoreT_diff"])))) )) * 2.0)) * 2.0))) +

                    0.100000*np.tanh(np.real(np.where(np.abs(data["Seed_diff"]) >= np.abs(((np.where(np.abs(np.sinh((((complex(-2.0)) * (((complex(-3.0)) * (data["Seed_diff"]))))))) <= np.abs(data["ScoreT_diff"]),np.sinh((((complex(-2.0)) * (((complex(-3.0)) * (data["Seed_diff"])))))), data["ScoreT_diff"] )) + (((complex(-3.0)) * (data["Seed_diff"]))))),data["Seed_diff"], ((np.where(np.abs(np.sinh((((complex(-2.0)) * (((complex(-3.0)) * (data["Seed_diff"]))))))) <= np.abs(data["ScoreT_diff"]),np.sinh((((complex(-2.0)) * (((complex(-3.0)) * (data["Seed_diff"])))))), data["ScoreT_diff"] )) + (((complex(-3.0)) * (data["Seed_diff"])))) ))) +

                    0.100000*np.tanh(np.real(np.sinh((((data["Seed_diff"]) * (((((np.cos((((((np.cosh((np.cos((data["Seed2"]))))) - (complex(4.0)))) * (data["Seed2"]))))) * 2.0)) - (complex(4.0))))))))))



def GP(data):

    return (GPI(data)+GPII(data)+GPIII(data)+GPIV(data)+GPV(data)+

            GPVI(data.astype(complex))+GPVII(data.astype(complex))+

            GPVIII(data.astype(complex))+GPIX(data.astype(complex))+

            GPX(data.astype(complex)))/10
log_loss(y,GP(X))
submission_df = pd.read_csv(directory+'MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = GP(test_df).values

submission_df.to_csv('submission.csv', index=False)