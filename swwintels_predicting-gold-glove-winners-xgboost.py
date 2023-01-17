import pandas as pd

from pandas import DataFrame

import numpy as np



import io

#import boto3



import xgboost as xgb

from xgboost import plot_tree



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [24, 16]

%matplotlib inline

%pylab inline
LOCAL = True

LOCAL_FILE_DIR = "../input/the-history-of-baseball/"

BUCKET = "" #Point to your AWS S3 bucket



YEARS = range(1957,2015)

POSITION_GROUPS = ["INF"] #["P", "C", "INF", "OUTF" ]

POS_IN_GROUPS ={"INF":["1B","2B","3B","SS"], "OUTF": ["OF"], "P":["P"],"C":["C"]}
def read_df_from_S3(csv_file):

    s3 = boto3.client('s3'

    #local only

                        , aws_access_key_id="",

                          aws_secret_access_key=""

    #

                     )

    obj = s3.get_object(Bucket=BUCKET, Key='2017/'+csv_file) 

    temp_df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    return temp_df
df_master = read_df_from_S3('Master.csv') if not(LOCAL) else pd.read_csv(LOCAL_FILE_DIR+'player.csv')

df_fielding = read_df_from_S3('Fielding.csv') if not(LOCAL) else pd.read_csv(LOCAL_FILE_DIR+'fielding.csv')

df_appearances = read_df_from_S3('Appearances.csv') if not(LOCAL) else pd.read_csv(LOCAL_FILE_DIR+'appearances.csv')

df_batting = read_df_from_S3('Batting.csv') if not(LOCAL) else pd.read_csv(LOCAL_FILE_DIR+'batting.csv')

df_teams = read_df_from_S3('Teams.csv') if not(LOCAL) else pd.read_csv(LOCAL_FILE_DIR+'team.csv')



df_gg = read_df_from_S3('ggwinners.csv') if not(LOCAL) else pd.read_csv('../input/goldglovewinners/ggwinners.csv')
def one_hot_encode(column, df, remove=False):



    label_encoder = LabelEncoder()

    df_recoded = label_encoder.fit_transform(df[column].fillna("NA"))

    df_recoded = df_recoded.reshape(df.shape[0],1)

    onehot_encoder = OneHotEncoder(sparse=False)

    df_recoded = onehot_encoder.fit_transform(df_recoded)



    col =[] 

    for i in range(0,df_recoded.shape[1]):

        col.append(column + "_"+ label_encoder.classes_[i])

    

    df_recoded = DataFrame(df_recoded, columns=col)

    if(remove):

        df.pop(column)

    return df_recoded
def populate_features(position_group, year):

    

    #Select only active players in the Position group

    active_players = df_appearances[df_appearances.year==year]["player_id"]

    df_master_year = df_master[df_master.player_id.isin(active_players)].reset_index(drop=True)

       

    #Filter out only position in the position group

    df_field_group = df_fielding[df_fielding.year==year].sort_values(by=['player_id','g'], ascending=[True,False]).drop_duplicates("player_id")

    df_field_group = df_field_group[df_field_group.pos.isin(POS_IN_GROUPS[position_group])]

    

    #Fielding

    df_field_group["fielding_average"] = (df_field_group.po+df_field_group.a)/(df_field_group.po+df_field_group.a+df_field_group.e)

    df_field_group["double_play_rate"] = (df_field_group.dp)/(df_field_group.po+df_field_group.a+df_field_group.e)

    df_field_group["putout_vs_assist"] = (df_field_group.po)/df_field_group.a

    df_field_group["activity"] = (df_field_group.po+df_field_group.a)/(df_field_group.inn_outs)

        

    df_master_year = pd.merge(df_master_year, df_field_group,how='inner', on=['player_id','player_id'])    

    del df_master_year['year']

    del df_master_year['league_id']

    del df_master_year['stint']

        

    #Appearances

    df_app_group=df_appearances[df_appearances.year==year].groupby(df_appearances['player_id']).sum().reset_index()

    df_master_year = pd.merge(df_master_year, df_app_group,how='left', on=['player_id','player_id'])

    del df_master_year['year']

    

    #Batting

    df_bat_group=df_batting[df_batting.year==year].groupby(df_batting['player_id']).sum().reset_index()

    df_bat_group["batting_average"] = df_bat_group.h / df_bat_group.ab

    df_bat_group["on_base_percentage"] = (df_bat_group.h + df_bat_group.bb +df_bat_group.hbp) / (df_bat_group.ab+ df_bat_group.bb+ df_bat_group.hbp+df_bat_group.sf)

    df_bat_group["1B"] = df_bat_group.h - df_bat_group["double"]- df_bat_group["triple"] - df_bat_group.hr

    df_bat_group["slugging_percentage"] = (df_bat_group["1B"] + 2*df_bat_group["double"] +3*df_bat_group["triple"]+4*df_bat_group["hr"]) / df_bat_group.ab

    df_bat_group["RBI_rate"] = df_bat_group.rbi / df_bat_group.ab

    df_master_year = pd.merge(df_master_year, df_bat_group,how='left', on=['player_id','player_id'])

    del df_master_year['year']

    

    #Team performance

    df_team_group = df_teams[df_teams.year==year][["team_id", "rank", "w", "div_win", "wc_win","lg_win","ws_win","fp","e"]]

    df_team_group["e_team"]=df_team_group.pop("e")

                                                    

    df_team_group = df_team_group = df_team_group.replace(['Y', 'N'], [1, 0])

    df_master_year = pd.merge(df_master_year, df_team_group,how='left', on=['team_id','team_id'])

        

    #One hot encode several features

    df_master_year = pd.concat([df_master_year, one_hot_encode("bats", df_master_year, True)], axis=1)

    df_master_year = pd.concat([df_master_year, one_hot_encode("throws", df_master_year, True)], axis=1)

    df_master_year = pd.concat([df_master_year, one_hot_encode("pos", df_master_year, True)], axis=1)

     

    #Other features

    df_master_year["age_in_season"] = year - df_master_year["birth_year"]

        

    #Target label creation

    winners = df_gg[(df_gg.Year==year)& (df_gg.Position_group == position_group)]['playerID']

    df_master_year["won"]=df_master_year.player_id.isin(winners)*1

    df_master_year["stat_year"] = year

        

    return df_master_year
def clean_features(df, feature_names=None):

    if(feature_names is None):

        not_features = ['player_id','birth_year','birth_month','birth_day','birth_country', 'birth_state','birth_city','death_year','death_month','death_day',

                        'debut','final_game','death_country','death_state','death_city','name_first','name_last',

                        'name_given','retro_id','bbref_id','throws_S', 'stint', 'team_id', 

                        'franchID', 'divID', 'name', 'park',

                        'league_id', 'teamIDBR','teamIDlahman45','teamIDretro']

        feature_names = [f for f in df.columns if f not in not_features]

    return df[feature_names], feature_names
df_train = pd.DataFrame()

for position_group in POSITION_GROUPS:

    frames = [populate_features(position_group, year) for year in YEARS]

    frame_order = frames[0].keys().values # can be removed later

    df_train = pd.concat(frames)

    df_train = df_train[np.concatenate((frame_order,[c for c in df_train.columns if c not in frame_order]))]

    df_features, feature_names = clean_features(df_train)
plt.hist(df_train["fielding_average"].dropna(), bins=100)

plt.xlabel('Fielding average')

plt.ylabel('Number of players')

plt.show()
Xtr = df_features[df_features.stat_year%4!=0]

Xv  = df_features[df_features.stat_year%4==0]

ytr = Xtr.pop("won")

yv  = Xv.pop("won")
df_test_full = populate_features("INF", 2015)

df_test, te_feature_names = clean_features(df_test_full, feature_names)

yreal = df_test.pop("won")

print(Xtr.shape)

print(df_test.shape)
dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

dtest = xgb.DMatrix(df_test)



watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



xgb_pars = {'min_child_weight': 1, 'eta': 0.003, 'colsample_bytree': 0.8, 'max_depth': 8,

            'subsample': 0.3, 'lambda': 1., 'booster' : 'gbtree', 'silent': 0,

            'eval_metric': ['auc','logloss'], 'objective': 'binary:logistic'}



model = xgb.train(xgb_pars, dtrain, 2500, watchlist, early_stopping_rounds=50,

                  maximize=False, verbose_eval=100)



print('Modeling score %.5f' % model.best_score)

#t1 = dt.datetime.now()

#print('Training time: %i seconds' % (t1 - t0).seconds)

feature_importance_dict = model.get_fscore()

fs = ['f%i' % i for i in range(len(feature_names))]

f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()), 'importance': list(feature_importance_dict.values())})

feature_importance = f1

#feature_importance = feature_importance.fillna(0)

feature_importance[['f', 'importance']].sort_values(by='importance', ascending=False).head(10)
ypred = pd.DataFrame(model.predict(dtest),columns=["win_chance"])

yreal = pd.DataFrame(yreal,columns=["won"])

plt.hist(ypred.values, bins=100)

plt.yscale('log', nonposy='clip')

plt.xlabel('Probability of winning')

plt.ylabel('Number of players')

plt.show()
df_t = pd.concat([df_test_full,ypred], axis=1)

df_t.to_csv('predictions.csv')
nominated={2016:

           ["dickera01","keuchda01", "verlaju01",

            "mccanja02","perezca02","perezsa02", 

            "davisch02","hosmeer01","morelmi01",

            "canoro01","kinslia01", "pedrodu01",

            "beltrad01","machama01","seageky01",

            "iglesjo01","lindofr01","simmoan01",

            "gardnbr01","gordoal01","rasmuco01",

            "bradlja02","pillake01","kiermke01",

            "bettsmo01","eatonad02","springe01",

            "arrieja01","greinza01","wainwad01",

            "lucrojo01","molinya01","poseybu01",

            "goldspa01","myerswi01","rizzoan01",

            "lemahdj01","panikjo01","segurje01",

            "arenano01","rendoan01","turneju01",

            "crawfbr01","galvifr01","russead02",

            "duvalad01","martest01","yelicch01",

            "hamilbi02","herreod01","inciaen01",

            "gonzaca01","heywaja01","markani01"

            ],

          2015:

          ["buehrma01","grayso01", "keuchda01",

            "castrja01","martiru01","perezsa02", 

            "hosmeer01","napolmi01","teixema01",

            "altuvjo01","doziebr01", "kinslia01",

            "beltrad01","longoev01","machama01",

            "bogaexa01","escobal02","gregodi01",

            "cespeyo01","gardnbr01","gordoal01",

            "kiermke01","pillake01","troutmi01",

            "calhoko01","martijd02","reddijo01",

            "arrieja01","colege01","greinza01",

            "molinya01","poseybu01","ramoswi01",

            "beltbr01","goldspa01","gonzaad01",

            "gordode01","lemahdj01","phillbr01",

            "arenano01","duffyma01","frazito01",

            "crawfbr01","hechaad01","simmoan01",

            "martest01","uptonju01","yelicch01",

            "hamilbi02","mccutan01","polloaj01",

            "grandcu01","harpebr03","heywaja01"

          ]}



df_t["nominated"] = df_t.player_id.isin(nominated[2015])
df_t.boxplot(column='win_chance', by=['nominated','won'], grid=False)

for i in [0,1]:

    for j in [0,1]:

        y = df_t.win_chance[(df_t.nominated==i)&(df_t.won==j)].dropna()

        # Add some random "jitter" to the x-axis

        x = np.random.normal(i+j+1, 0.02, size=len(y))

        plot(x, y, 'r.', alpha=0.6)
disp = ["player_id","name_first","name_last","win_chance","won","nominated"]

df_t[(df_t.won==1)].sort_values(by="win_chance", ascending=False)[disp]
df_t[(df_t.won==0)&(df_t.nominated)].sort_values(by="win_chance", ascending=False)[disp]
df_nominandum = df_t[(df_t.nominated==False) & (df_t.win_chance>df_t[df_t.nominated==True]["win_chance"].mean())]

df_nominandum.sort_values(by="win_chance", ascending=False)[disp]
df_earlier_wins= pd.merge(df_nominandum[['player_id']], df_train, on=['player_id','player_id'], how='inner')[["player_id",'name_first','name_last',"won","stat_year"]]

df_earlier_wins= df_earlier_wins.groupby(['player_id','name_first','name_last']).sum()

del df_earlier_wins["stat_year"]

df_earlier_wins["has_won_GG_earlier"] = (df_earlier_wins["won"]>0)



df_earlier_wins["number_of_wins"] = df_earlier_wins.pop("won")

df_earlier_wins.sort_values(by='number_of_wins',ascending=False)
xgb.plot_tree(model, num_trees=2)

fig = matplotlib.pyplot.gcf()

fig.set_size_inches(150, 100)

fig.savefig('tree.png')