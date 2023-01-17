from kaggle.competitions import nflrush

import pandas as pd

import numpy as np

#from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier as RFC

from tqdm import tqdm



env = nflrush.make_env()
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

%matplotlib inline
import warnings

warnings.simplefilter('ignore')
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
iter_test = env.iter_test()
new_df=df.groupby(['PlayId','Position']).count()
position_count=new_df['GameId'].unstack().fillna(0).astype(int)
#ボールを持っている人のデータのみ抽出

rusher_df=df[df['NflId']==df['NflIdRusher']]
def count_position(rusher_df):

    rusher_df=rusher_df.merge(position_count, on='PlayId')

    rusher_df=rusher_df.rename(columns={'S_x':'S','S_y':'S_position'})

    return rusher_df
def preprocess(df):

    #StadiumTypeからおかしなデータを削除

    df=df[(df['StadiumType']!='Cloudy') & (df['StadiumType']!='Bowl')]

    #StadiumTypeの文字列を屋外内で分けてリスト化

    outdoor=['Outdoor', 'Outdoors','Open','Indoor, Open Roof','Outdoor Retr Roof-Open', 'Oudoor', 'Ourdoor','Retr. Roof-Open','Outdor','Retr. Roof - Open', 'Domed, Open', 'Domed, open', 'Outside','Heinz Field']

    indoor=['Indoors', 'RetractableRoof', 'Indoor','Retr. Roof-Closed','Dome', 'Domed, closed','Indoor, Roof Closed', 'Retr. Roof Closed','Closed Dome','Dome, closed','Domed']

    #StadiumTypeがoutdoorの時に１になるようにダミー変数化

    df['stadiumtype']=(df['StadiumType'].isin(outdoor)*1)

    #天候の悪い時だけリスト化

    rain=['Light Rain', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.','Rain', 'Heavy lake effect snow','Snow', 'Cloudy, Rain','Rain shower','Rainy']

    #天気が悪くない時に１になるようにダミー変数化

    df['weather']=(~df['GameWeather'].isin(rain)*1)

    #身長をフィートからセンチに変換

    df['PlayerHeight']= df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    #ゲームの経過時間を算出

    df['gameclock']=[ pd.Timedelta(val).total_seconds() for val in df['GameClock']]

    #Orientationを整える

    df.loc[df["Season"]==2017, "Orientation"] = (df.loc[df["Season"]==2017, "Orientation"] -90)%360

    #攻撃の向きを右を正として揃える

    df.loc[df['PlayDirection']=='left','Dir'] = 180 + df['Dir'] - 360

    df.loc[df['PlayDirection']=='left','Orientation'] = 180 + df['Orientation'] - 360

    df.loc[df['PlayDirection']=='left','X'] = 120 - df['X']

    df.loc[df['PlayDirection']=='left','Y'] = 53.3 - df['Y']

    #Orientationをx,y成分に分ける

    df['sin_Ori']=(df['Orientation']*np.pi/180).map(np.sin) 

    df['cos_Ori']=(df['Orientation']*np.pi/180).map(np.cos)

    #Dirをx,y成分に分けて速度をかける

    df['sin_Dir_S']=(df['Dir']*np.pi/180).map(np.sin)*df['S']

    df['cos_Dir_S']=(df['Dir']*np.pi/180).map(np.cos)*df['S']

    

    return df
'''def add_team_yard(rusher_df):

    #チーム毎(home/away別)の獲得ヤード数の平均を見る

    team_yards_df = rusher_df.groupby(['Team','PossessionTeam']).mean()[['Yards']]

    team_yards_df = team_yards_df.rename(columns={'Yards':'team_yards'})

    #rusherのみのデータにチーム毎の平均獲得ヤード数を加える

    rusher_df = rusher_df.merge(team_yards_df,on='PossessionTeam',how="left")

    return rusher_df,team_yards_df'''
def add_team_score(rusher_df):

    # 攻撃チームの得点

    rusher_df.loc[rusher_df["Team"]=="home", "rusherTeamScore"] = rusher_df["HomeScoreBeforePlay"]

    rusher_df.loc[rusher_df["Team"]=="away", "rusherTeamScore"] = rusher_df["VisitorScoreBeforePlay"]



    # 守備チームの得点

    rusher_df.loc[rusher_df["Team"]=="home", "defenceTeamScore"] = rusher_df["VisitorScoreBeforePlay"]

    rusher_df.loc[rusher_df["Team"]=="away", "defenceTeamScore"] = rusher_df["HomeScoreBeforePlay"]



    # 得点差

    rusher_df.loc[:, "diffScore"] = rusher_df["rusherTeamScore"] - rusher_df["defenceTeamScore"]

    return rusher_df
def count_yard_to_touchdown(rusher_df):

    #タッチダウンまで何ヤードあるか

    rusher_df["yardsToTouchdown"] = 100-rusher_df['X']

    rusher_df["yardsToTouchdown"].clip(0,100,inplace=True)

    return rusher_df
def add_personal_yard(rusher_df):

    # 選手毎の平均獲得ヤード

    rusher_yards = rusher_df[["NflId", "Yards"]].groupby("NflId").mean()[["Yards"]]

    rusher_yards.dropna(inplace=True)

    rusher_yards=rusher_yards.rename(columns={'Yards':'PersonalYard'})

    rusher_df = rusher_df.merge(rusher_yards, on="NflId", how="left")

    return rusher_df,rusher_yards

def add_average_data(df,rusher_df):

    offence_position = ['WR', 'TE', 'T', 'QB', 'RB', 'G', 'C', 'FB', 'HB',  'OT', 'OG']

    df["offence"] = 0

    df.loc[df["Position"].isin(offence_position), "offence"] = 1

    # 攻撃,守備チーム平均 体重, 身長, S, A（PlayIdがキー）

    offence_av = df.loc[df["offence"]==1, ["PlayerHeight", "PlayerWeight", "S", "A", "PlayId"]].groupby("PlayId").mean()

    defence_av = df.loc[df["offence"]==0, ["PlayerHeight", "PlayerWeight", "S", "A", "PlayId"]].groupby("PlayId").mean()

    offence_av.columns = ['PlayerHeight_offence', 'PlayerWeight_offence', 'S_offence', 'A_offence']

    defence_av.columns = ['PlayerHeight_defence', 'PlayerWeight_defence', 'S_defence', 'A_defence']

    rusher_df = rusher_df.merge(offence_av, on="PlayId", how="left").merge(defence_av, on="PlayId", how="left")

    return rusher_df

    
def feature(df):

    features=pd.DataFrame(df,columns=['X', 'Y', 'S', 'A', 'Dis',

       'gameclock', 'Distance','HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'DefendersInTheBox','PlayerHeight',

       'PlayerWeight','Temperature', 'Humidity',

        'stadiumtype', 'weather', 

        'C', 'CB', 'DB','DE', 'DL', 'DT', 'FB', 'FS', 'G', 'HB', 'ILB', 'LB', 'MLB', 'NT', 'OG','OLB', 'OT', 'QB', 'RB', 'S_position', 'SAF', 'SS', 'T', 'TE', 'WR',

        "yardsToTouchdown",

       'PersonalYard',

       #'team_yards',

       #"rusherTeamScore","defenceTeamScore",

        "diffScore",

        'PlayerHeight_offence', 'PlayerWeight_offence', 'S_offence', 'A_offence',

        'PlayerHeight_defence', 'PlayerWeight_defence', 'S_defence', 'A_defence',

        'sin_Dir_S','cos_Dir_S','sin_Ori','cos_Ori'])

    return features   
rusher_df=count_position(rusher_df)
df=preprocess(df)
rusher_df=preprocess(rusher_df)
#rusher_df,team_yards_df=add_team_yard(rusher_df)
rusher_df=add_team_score(rusher_df)
rusher_df=count_yard_to_touchdown(rusher_df)
rusher_df,rusher_yards=add_personal_yard(rusher_df)
rusher_df=add_average_data(df,rusher_df)
rusher_df=rusher_df.dropna()
features=feature(rusher_df)
train_mean=features.mean(axis=0)
train_std=features.std(axis=0)
def normalize(features):

    X=(features-train_mean)/train_std

    return X
X=normalize(features)
target=pd.Series(rusher_df['Yards'])
train_X,test_X,train_y,test_y=train_test_split(X,target,test_size=0.2)
import optuna
'''

def objectives(trial):

       

        params = {

            'criterion': 'entropy', #trial.suggest_categorical('criterion', ['gini', 'entropy']),

            'n_estimators': trial.suggest_int("n_estimators", 100, 500),

            'max_depth': trial.suggest_int("max_depth", 4,10),

            'min_samples_split': trial.suggest_int("min_samples_split", 2,50),

            'min_samples_leaf': trial.suggest_int('min_samples_leaf',1,10),

            'random_state': 0,

            'verbose' : 0,

            'max_features' : trial.suggest_int('max_features', 1,56)

        }

        

        RFC_optuna=RFC(**params)

        #訓練

        RFC_optuna.fit(train_X,train_y)

        #検証

        pred_y=RFC_optuna.predict_proba(test_X)

        

        #実測値の累積確率のアレーを作成

        test_y_score=np.array([(i >= test_y)*1 for i in range(-99,100)])

        

        #予測値の累積確率のアレーを作成

        pred_prob_cdf=pd.DataFrame(pred_y,columns=[ "Yards"+str(i) for i in RFC_optuna.classes_])

        pred_prob_cdf=pd.DataFrame(pred_prob_cdf,columns=[ "Yards"+str(i) for i in range(-99,100)])

        pred_prob_cdf.fillna(0,inplace=True)

        pred_prob_cdf = pred_prob_cdf.cumsum(axis=1)

        pred_prob=np.array(pred_prob_cdf.values)

        

        #実測値と予測値の誤差で評価

        C=((pred_prob - test_y_score.T)**2).sum().sum()/(199*len(pred_y))

        

        return C

'''
'''

# optimizeの第一引数に対象のメソッドを指定、n_trialsにプログラムが試行錯誤する回数を指定

study = optuna.create_study()

study.optimize(objectives, n_jobs=-1,n_trials=100)

'''
'''

import plotly

from optuna.visualization import is_available,plot_contour,plot_intermediate_values,plot_optimization_history,plot_parallel_coordinate

#optunaの可視化ができるか確認

optuna.visualization.is_available()

'''
'''

#パラメーターの関係性を等高線図としてプロット

optuna.visualization.plot_contour(study,params=['n_estimators','max_depth','min_samples_split','min_samples_leaf','max_features'])

#全試行の中間値をプロット

optuna.visualization.plot_intermediate_values(study)

#最適化の過程をプロット

optuna.visualization.plot_optimization_history(study)

optuna.visualization.plot_parallel_coordinate(study,params=['n_estimators','max_depth','min_samples_split','min_samples_leaf','max_features'])

'''
'''

#optunaで見つけた最適なパラメーターを出力

#best_params=study.best_params

study.best_params

'''
'''

#最適なパラメーターでのモデルの精度 C=0.01245163097304255

study.best_value

'''
best_params={'criterion': 'entropy',

             'n_estimators': 421,

             'max_depth': 10,

             'min_samples_split': 3,

             'min_samples_leaf': 6,

             'max_features': 41,

             'random_state': 0}
model = RFC(**best_params)
#モデルにさらに全データを学習

rfc=model.fit(X,target)
rfc.get_params
feat_imp=pd.DataFrame(rfc.feature_importances_,index=X.columns)

feat_imp.sort_values(0,ascending=False)
train_df=rusher_df.iloc[:0,:]
for (test_df, sample_prediction_df) in tqdm(iter_test):

    new_df=test_df.groupby(['PlayId','Position']).count()

    position_count=new_df['GameId'].unstack().fillna(0).astype(int)

    rusher_df=test_df[test_df['NflId']==test_df['NflIdRusher']]

    rusher_df=preprocess(rusher_df)

    test_df=preprocess(test_df)

    rusher_df=count_position(rusher_df)

    #rusher_df=rusher_df.merge(rusher_yards,  on="NflId", how="left")

    #rusher_df = rusher_df.merge(team_yards_df,on='PossessionTeam',how="left")

    rusher_df=add_team_score(rusher_df)

    rusher_df=count_yard_to_touchdown(rusher_df)

    rusher_df=add_average_data(test_df,rusher_df)

    rusher_df=pd.concat([train_df,rusher_df],sort=False)

    test_feature=feature(rusher_df)

    test_feature=test_feature.fillna(0)

    test_X=normalize(test_feature)

    pred_prob=rfc.predict_proba(test_X)

    pred_prob_cdf=pd.DataFrame(pred_prob,columns=[ "Yards"+str(i) for i in rfc.classes_])

    pred_prob_cdf=pd.DataFrame(pred_prob_cdf, columns=[ "Yards"+str(i) for i in range(-99,100)])

    pred_prob_cdf.fillna(0,inplace=True)

    pred_prob_cdf = pred_prob_cdf.cumsum(axis=1)

    pred_prob_cdf[pred_prob_cdf>1]=1

    #pred_prob_cdf.loc[:, :"Yards-6"] = 0

    #pred_prob_cdf.loc[:, "Yards21":] = 1

    sample_prediction_df.iloc[0,:]=pred_prob_cdf.iloc[0,:]

    env.predict(sample_prediction_df)
sample_prediction_df
env.write_submission_file()
import os

print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])