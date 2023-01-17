import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from catboost import Pool, cv, CatBoostClassifier

from sklearn.model_selection import TimeSeriesSplit, train_test_split

from sklearn.metrics import  classification_report, log_loss, roc_auc_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
GAME_COLS  =['GAME_DATE_EST', 'GAME_ID', 'HOME_TEAM_ID',

       'VISITOR_TEAM_ID', 'SEASON', 'HOME_TEAM_WINS']

df = pd.read_csv("/kaggle/input/nba-games/games.csv",usecols = GAME_COLS,parse_dates=["GAME_DATE_EST"],infer_datetime_format=True)

df = df.drop_duplicates().sort_values("GAME_DATE_EST").set_index(["GAME_DATE_EST"])

print(df.shape)

df.head()
df.tail(3)
df_players = pd.read_csv("/kaggle/input/nba-games/players.csv")

print(df_players.shape)

df_players.head()



### joining the players means a many to 1 join - of all players per team. Kludgy code, skip for now, especially without further features at the player level
df_teams = pd.read_csv("/kaggle/input/nba-games/teams.csv")

print(df_teams.shape)

df_teams.head()
df.HOME_TEAM_ID.nunique() ## no obvious mismatch in # teams
df_ranking = pd.read_csv("/kaggle/input/nba-games/ranking.csv",parse_dates=["STANDINGSDATE"])

df_ranking.sort_values("STANDINGSDATE",inplace=True)

print(df_ranking.shape)



## drop the less interesting or amenably columns . We could get ratio features from the record cols, but that'd require splitting first : 

df_ranking.drop(["CONFERENCE","LEAGUE_ID","HOME_RECORD","ROAD_RECORD"],axis=1,inplace=True) 



df_ranking.head()
df_ranking["SEASON_ID"].unique()
print(df.shape)



df_ranking.set_index("STANDINGSDATE",inplace=True)



df = pd.merge_asof(df, df_ranking.add_suffix("_homeTeam"),

              left_index=True,

                       right_index=True,

              left_by="HOME_TEAM_ID",

                       right_by='TEAM_ID'+"_homeTeam",

#                         suffixes="_homeTeam",  ## for some reason this gives error, so we workaround it by adding suffixes

                       allow_exact_matches=False)



df = pd.merge_asof(df, df_ranking.add_suffix("_awayTeam"),

              left_index=True,

              right_index=True,

              left_by="VISITOR_TEAM_ID",

                       right_by='TEAM_ID'+"_awayTeam",

                       allow_exact_matches=False)



df.drop(["SEASON_ID_awayTeam","TEAM_ID_awayTeam","TEAM_ID_homeTeam"],axis=1,inplace=True) ## redundant

df.rename(columns={"SEASON_ID_homeTeam":"SEASON_ID"},inplace=True)

print(df.shape)

df.head()
df.loc[df.G_homeTeam==0]
def missing_game_rankings(row,suffix="_homeTeam"):

    if ((row["G"+suffix]==0) & (row["W"+suffix]==0)): 

        row["G"+suffix]=np.nan

        row["W"+suffix]=np.nan

        row["L"+suffix]=np.nan

        row["W_PCT"+suffix]=np.nan

    return row

  

df = df.apply(lambda x: missing_game_rankings(x,suffix="_awayTeam"),axis=1)

df = df.apply(lambda x: missing_game_rankings(x,suffix="_homeTeam"),axis=1)



print(df.isna().sum())



df = df.dropna()

print("df without nans size:", df.shape[0])
## sklearn temporal split is for CV - we don't need. data is sorted, so we'll just take the last 20% of rows

## get only numeric columns - we don't need the strings here

CUTOFF_ROW = int(df.shape[0]*0.8)

# X = df.reset_index().drop(["SEASON"],axis=1)._get_numeric_data().copy() 

X = df.drop(["SEASON"],axis=1)._get_numeric_data().copy() 

X_train = X[:CUTOFF_ROW].drop(["HOME_TEAM_WINS"],axis=1)

print("X_train",X_train.shape)

X_test = X[CUTOFF_ROW:].drop(["HOME_TEAM_WINS"],axis=1)

print("X_test",X_test.shape)

y_train = X[:CUTOFF_ROW]["HOME_TEAM_WINS"]

print("y_train",len(y_train))

y_test = X[CUTOFF_ROW:]["HOME_TEAM_WINS"]
X_test
print([c for c in X_train.columns if 5<X_train[c].nunique()<8000])



categorical_cols = ['HOME_TEAM_ID', 'VISITOR_TEAM_ID']
## catBoost Pool object

train_pool = Pool(data=X_train,label = y_train,cat_features=categorical_cols,

#                   baseline= X_train["W_PCT_homeTeam"], ## not as relevant as a baseline, since we subtracted by it (rather than dividing)

#                   group_id = X_train['SEASON_ID']

                 )



test_pool = Pool(data=X_test,label = y_test,cat_features=categorical_cols,

#                   baseline= X_train["W_PCT_homeTeam"], ## not as relevant as a baseline, since we subtracted by it (rather than dividing)

#                   group_id = X_test['SEASON_ID']

                 )
model = CatBoostClassifier(verbose=False) # ,task_type="GPU") # use GPU acceleration - requires kernel to have GPU activated and limits availability



model.fit(train_pool, plot=True,silent=True)

print(model.get_best_score())
## get results on test set 

test_preds = model.predict(test_pool,prediction_type='Class')

print(classification_report(y_true=y_test,y_pred=test_preds))
test_preds_proba = model.predict(test_pool,prediction_type='Probability')[:,1]



print("Test AUC:")

print("%.4f" % roc_auc_score(y_true=y_test, y_score = test_preds_proba))



print("Test Log Loss:")

print("%.4f" % log_loss(y_true=y_test, y_pred = test_preds_proba))
params = {

          "loss_function": "Logloss",

          "verbose": False,

          "use_best_model":True, ## requires a validation dataset to be provided.

          "custom_metric":['Logloss', 'AUC',"Precision"],

         }



df_cv = cv(pool=train_pool,params=params,plot=True,type="TimeSeries",fold_count=6,metric_period=3)



display(df_cv.sample(5))



test_eval_cols = [c for c in df_cv.columns if ("test" in c) & ("mean" in c)]

display(df_cv[test_eval_cols].median())

display(df_cv.tail(1)[test_eval_cols])
feature_importances = model.get_feature_importance(train_pool)

feature_names = X_train.columns

for score, name in sorted(zip(feature_importances, feature_names), reverse=True):

    print('{}: {}'.format(name, score))
pd.concat([X_train,y_train],axis=1).to_csv("NBA_teams_train.csv")

pd.concat([X_test,y_test],axis=1).to_csv("NBA_teams_test.csv")
pd.concat([X_train,y_train],axis=1)