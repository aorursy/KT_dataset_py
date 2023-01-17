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
import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

from xgboost import XGBClassifier

import gc

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor
Tourney_Compact_Results = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

Tourney_Seeds = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')
RegularSeason_Compact_Results = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')

MSeasons = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSeasons.csv')

MTeams=pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv')
Tourney_Compact_Results.head(65)
Tourney_Compact_Results.shape
Tourney_Compact_Results.info()
RegularSeason_Compact_Results
RegularSeason_Compact_Results.shape
RegularSeason_Compact_Results.info()
RegularSeason_Compact_Results.head()
RegularSeason_Compact_Results.info()
RegularSeason_Compact_Results.shape
Tourney_Seeds.head()
Tourney_Seeds.info()
Tourney_Seeds.shape
Tourney_Compact_Results.groupby('Season').mean().head()
Tourney_Results_Compact=pd.merge(Tourney_Compact_Results, Tourney_Seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

Tourney_Results_Compact.rename(columns={'Seed':'WinningSeed'},inplace=True)

Tourney_Results_Compact=Tourney_Results_Compact.drop(['TeamID'],axis=1)



Tourney_Results_Compact = pd.merge(Tourney_Results_Compact, Tourney_Seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

Tourney_Results_Compact.rename(columns={'Seed':'LoosingSeed'}, inplace=True)

Tourney_Results_Compact=Tourney_Results_Compact.drop(['TeamID','NumOT','WLoc'],axis=1)





Tourney_Results_Compact
Tourney_Results_Compact=Tourney_Results_Compact.drop(['WScore','LScore'],axis=1)

Tourney_Results_Compact.head()
Tourney_Results_Compact.info()
Tourney_Results_Compact['WinningSeed'] = Tourney_Results_Compact['WinningSeed'].str.extract('(\d+)', expand=True)

Tourney_Results_Compact['LoosingSeed'] = Tourney_Results_Compact['LoosingSeed'].str.extract('(\d+)', expand=True)

Tourney_Results_Compact.WinningSeed = pd.to_numeric(Tourney_Results_Compact.WinningSeed, errors='coerce')

Tourney_Results_Compact.LoosingSeed = pd.to_numeric(Tourney_Results_Compact.LoosingSeed, errors='coerce')
RegularSeason_Compact_Results.head()
season_winning_team = RegularSeason_Compact_Results[['Season', 'WTeamID', 'WScore']]

season_losing_team = RegularSeason_Compact_Results[['Season', 'LTeamID', 'LScore']]

season_winning_team.rename(columns={'WTeamID':'TeamID','WScore':'Score'}, inplace=True)

season_losing_team.rename(columns={'LTeamID':'TeamID','LScore':'Score'}, inplace=True)

RegularSeason_Compact_Results = pd.concat((season_winning_team, season_losing_team)).reset_index(drop=True)

RegularSeason_Compact_Results
RegularSeason_Compact_Results_Final = RegularSeason_Compact_Results.groupby(['Season', 'TeamID'])['Score'].sum().reset_index()

RegularSeason_Compact_Results_Final
Tourney_Results_Compact = pd.merge(Tourney_Results_Compact, RegularSeason_Compact_Results_Final, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

Tourney_Results_Compact.rename(columns={'Score':'WScoreTotal'}, inplace=True)

Tourney_Results_Compact
Tourney_Results_Compact = Tourney_Results_Compact.drop('TeamID', axis=1)

Tourney_Results_Compact = pd.merge(Tourney_Results_Compact, RegularSeason_Compact_Results_Final, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

Tourney_Results_Compact.rename(columns={'Score':'LScoreTotal'}, inplace=True)

Tourney_Results_Compact = Tourney_Results_Compact.drop('TeamID', axis=1)

Tourney_Results_Compact.to_csv('Tourney_Win_Results_Train.csv', index=False)

Tourney_Results_Compact=Tourney_Results_Compact[Tourney_Results_Compact['Season'] < 2015] 

Tourney_Results_Compact
Tourney_Win_Results=Tourney_Results_Compact.drop(['Season','WTeamID','LTeamID','DayNum'],axis=1)

Tourney_Win_Results
Tourney_Win_Results.rename(columns={'WinningSeed':'Seed1', 'LoosingSeed':'Seed2', 'WScoreTotal':'ScoreT1', 'LScoreTotal':'ScoreT2'}, inplace=True)
RegularSeason_Compact_Results.head(200)
Tourney_Win_Results
tourney_lose_result = Tourney_Win_Results.copy()

tourney_lose_result['Seed1'] = Tourney_Win_Results['Seed2']

tourney_lose_result['Seed2'] = Tourney_Win_Results['Seed1']

tourney_lose_result['ScoreT1'] = Tourney_Win_Results['ScoreT2']

tourney_lose_result['ScoreT2'] = Tourney_Win_Results['ScoreT1']

tourney_lose_result
Tourney_Win_Results['Seed_diff'] = Tourney_Win_Results['Seed1'] - Tourney_Win_Results['Seed2']

Tourney_Win_Results['ScoreT_diff'] = Tourney_Win_Results['ScoreT1'] - Tourney_Win_Results['ScoreT2']

#Tourney_Win_Results['Score_diff'] = Tourney_Win_Results['Score1'] - Tourney_Win_Results['Score2']

tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']

tourney_lose_result['ScoreT_diff'] = tourney_lose_result['ScoreT1'] - tourney_lose_result['ScoreT2']
Tourney_Win_Results['result'] = 1

tourney_lose_result['result'] = 0

tourney_result_Final = pd.concat((Tourney_Win_Results, tourney_lose_result)).reset_index(drop=True)



tourney_result_Final.to_csv('Tourneyvalidate.csv', index=False)
tourney_result_Final.head()
tourney_result_Final1 = tourney_result_Final[[

    'Seed1', 'Seed2', 'ScoreT1', 'ScoreT2', 'Seed_diff', 'ScoreT_diff', 'result']]
tourney_result_Final1.head()
tourney_result_Final1.loc[lambda x: (x['Seed1'].isin([14,15,16])) & (x['Seed2'].isin([1,2,3])),'result'

        ] = 0







test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')
test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))

test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))

test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))

test_df
Tourney_Seeds.head()
#Tourney_Results_Compact1=Tourney_Results_Compact.drop(['DayNum','WinningSeed','LoosingSeed','WScoreTotal','LScoreTotal'],axis=1)
#Tourney_Results_Compact1.head()
#Tourney_Seeds= pd.merge(Tourney_Results_Compact1, Tourney_Seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
#Tourney_Seeds=Tourney_Seeds.drop(['WTeamID','LTeamID'],axis=1)

#Tourney_Seeds.head()
test_df = pd.merge(test_df, Tourney_Seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Seed':'Seed1'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, Tourney_Seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Seed':'Seed2'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)
test_df.head()
test_df = pd.merge(test_df, RegularSeason_Compact_Results_Final, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Score':'ScoreT1'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, RegularSeason_Compact_Results_Final, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')

test_df.rename(columns={'Score':'ScoreT2'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df

test_df.to_csv('test_df_Test.csv', index=False)
test_df['Seed1'] = test_df['Seed1'].str.extract('(\d+)', expand=True)

test_df['Seed2'] = test_df['Seed2'].str.extract('(\d+)', expand=True)

test_df.Seed1 = pd.to_numeric(test_df.Seed1, errors='coerce')

test_df.Seed2 = pd.to_numeric(test_df.Seed2, errors='coerce')
test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']

test_df['ScoreT_diff'] = test_df['ScoreT1'] - test_df['ScoreT2']

test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)

test_df
tourney_result_Final1.head()
X = tourney_result_Final1.drop('result', axis=1)

y = tourney_result_Final1.result
# Standardization for regression models

df = pd.concat([X, test_df], axis=0, sort=False).reset_index(drop=True)

df_log = pd.DataFrame(

    preprocessing.MinMaxScaler().fit_transform(df),

    columns=df.columns,

    index=df.index

)

train_log, test_log = df_log.iloc[:len(X),:], df_log.iloc[len(X):,:].reset_index(drop=True)
logreg = LogisticRegression()

logreg.fit(train_log, y)

coeff_logreg = pd.DataFrame(train_log.columns.delete(0))

coeff_logreg.columns = ['feature']

coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])

coeff_logreg.sort_values(by='score_logreg', ascending=False)
y_logreg_train = logreg.predict(train_log)

y_logreg_pred = logreg.predict_proba(test_log)
from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(n_estimators=250,max_depth=50)

clf = RandomForestClassifier(n_estimators=200,max_depth=90,min_samples_leaf=300,min_samples_split=200,max_features=5)

#We can get accuracy of 0.7145738779626828 using {'max_depth': 30, 'max_features': 5, 'min_samples_leaf': 300, 'min_samples_split': 300, 'n_estimators': 200}

#clf.fit(X, y)

clf.fit(train_log, y)

clf_probs = clf.predict_proba(test_log)
y_pred_df_random = pd.DataFrame(clf_probs)

y_pred_1 = y_pred_df_random.iloc[:,[1]]

#y_pred_1.head()

y_pred_df_random
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
#n_folds=5

# Create the parameter grid based on the results of random search 

#param_grid = {

#    'max_depth': [30,60,90,120,150],

#    'min_samples_leaf': (50,100, 200, 300,400),

#    'min_samples_split': (25,50,100, 200, 300),

#    'n_estimators': [100,200, 300,400], 

#    'max_features': [5, 10,15]

#}

# Create a based model

#rf = RandomForestClassifier()

# Instantiate the grid search model

#grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          #cv = 3, n_jobs = -1,verbose = 1)
#grid_search.fit(train_log, y)
#print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
submission_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

submission_df['Pred'] = y_pred_1

submission_df
submission_df.to_csv('submission_New6.csv', index=False)