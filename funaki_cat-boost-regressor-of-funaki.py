from kaggle.competitions import nflrush

import pandas as pd

import numpy as np

#from sklearn.preprocessing import StandardScaler

from tqdm import tqdm



env = nflrush.make_env()
from sklearn.model_selection import GridSearchCV,train_test_split
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
iter_test = env.iter_test()
#new_df=df.groupby(['PlayId','Position']).count()
#position_count=new_df['GameId'].unstack().fillna(0).astype(int)
rusher_df=df[df['NflId']==df['NflIdRusher']]
#def count_position(rusher_df):

    #usher_df=rusher_df.merge(position_count, on='PlayId')

    #rusher_df=rusher_df.rename(columns={'S_x':'S','S_y':'S_position'})

    #return rusher_df
def preprocess(df):

    #StadiumTypeからおかしなデータを削除

    #df=df[(df['StadiumType']!='Cloudy') & (df['StadiumType']!='Bowl')]

    #StadiumTypeの文字列を屋外内で分けてリスト化

    #outdoor=['Outdoor', 'Outdoors','Open','Indoor, Open Roof','Outdoor Retr Roof-Open', 'Oudoor', 'Ourdoor','Retr. Roof-Open','Outdor','Retr. Roof - Open', 'Domed, Open', 'Domed, open', 'Outside','Heinz Field']

    #indoor=['Indoors', 'RetractableRoof', 'Indoor','Retr. Roof-Closed','Dome', 'Domed, closed','Indoor, Roof Closed', 'Retr. Roof Closed','Closed Dome','Dome, closed','Domed']

    #StadiumTypeがoutdoorの時に１になるようにダミー変数化

    #df['stadiumtype']=(df['StadiumType'].isin(outdoor)*1)

    #天候の悪い時だけリスト化

    #rain=['Light Rain', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.','Rain', 'Heavy lake effect snow','Snow', 'Cloudy, Rain','Rain shower','Rainy']

    #天気が悪くない時に１になるようにダミー変数化

    #df['weather']=(~df['GameWeather'].isin(rain)*1)

    #身長をフィートからセンチに変換

    df['PlayerHeight']= df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    #ゲームの経過時間を算出

    df['gameclock']=[ pd.Timedelta(val).total_seconds() for val in df['GameClock']]

    return df
def add_team_yard(rusher_df):

    #チーム毎(home/away別)の獲得ヤード数の平均を見る

    team_yards_df = rusher_df.groupby(['Team','PossessionTeam']).mean()[['Yards']]

    team_yards_df = team_yards_df.rename(columns={'Yards':'team_yards'})

    #rusherのみのデータにチーム毎の平均獲得ヤード数を加える

    rusher_df = rusher_df.merge(team_yards_df,on='PossessionTeam',how="left")

    return rusher_df,team_yards_df
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

    rusher_df["yardsToTouchdown"] = rusher_df["YardLine"]

    rusher_df.loc[rusher_df["PossessionTeam"] == rusher_df["FieldPosition"], "yardsToTouchdown"] = 100-rusher_df["YardLine"]

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

    features=pd.DataFrame(df,columns=['X', 'Y', 'S', 'A', 'Dis','Dir','YardLine', 'Quarter',

       'gameclock', 'Down', 'Distance','HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'DefendersInTheBox','PlayerHeight',

       'PlayerWeight','Temperature', 'Humidity',

        #'stadiumtype', 'weather', 

        #'C', 'CB', 'DB','DE', 'DL', 'DT', 'FB', 'FS', 'G', 'HB', 'ILB', 'LB', 'MLB', 'NT', 'OG','OLB', 'OT', 'QB', 'RB', 'S_position', 'SAF', 'SS', 'T', 'TE', 'WR',

        "yardsToTouchdown",

       'PersonalYard','team_yards',

       "rusherTeamScore","defenceTeamScore","diffScore",

        'PlayerHeight_offence', 'PlayerWeight_offence', 'S_offence', 'A_offence',

        'PlayerHeight_defence', 'PlayerWeight_defence', 'S_defence', 'A_defence'])

    return features   
#rusher_df=count_position(rusher_df)
df=preprocess(df)
rusher_df=preprocess(rusher_df)
rusher_df,team_yards_df=add_team_yard(rusher_df)
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
features.shape
#import warnings

#warnings.simplefilter('ignore')
#from sklearn.model_selection import cross_val_score

#from bayes_opt import BayesianOptimization
import warnings

warnings.simplefilter('ignore')
import matplotlib.pyplot as plt

%matplotlib inline
#ベイズ最適化で出力されたパラメーターを四捨五入で丸める

#'max_depth': 48.704309793515755,

#'max_features': 5.671956856631306,

#'min_samples_split': 6.739604788678628,

#'n_estimators': 248.58715626809243

#決定係数は0.673114259554454



#def train_predict(X,target):

#    clf = RFC(n_estimators=249,

#              max_features=6,

#              max_depth=49,

#              min_samples_split=7)

#    clf.fit(X,target)

#    return clf
from catboost import CatBoostRegressor as CBR

from catboost import Pool
#検証用データの作成

#index=test_y.isin(train_y).index

#index_test_X=test_X.loc[index]

#index_test_y=test_y.loc[index]
# データセットの作成。Poolで説明変数、目的変数を指定

#train_pool = Pool(train_X, train_y)

#validate_pool = Pool(test_X, test_y)
#model=CBR()
#param_grid= {'learning_rate': [0.03, 0.1],

#             'depth': [4, 6, 10],

#             'l2_leaf_reg': [1, 3, 5, 7, 9],

#             'iterations': [350,400,450]}



#search_params=model.grid_search(param_grid,

#                                X=train_X,

#                                y=train_y,

#                                cv=3)
#search_params
#grid_searchで最適なパラメーターを代入

model=CBR(depth=10,

          l2_leaf_reg=1,

          iterations=450,

          learning_rate=0.1)
model.fit(train_X,train_y,

          #eval_set=validate_pool,    # 検証用データ

          #early_stopping_rounds=10,  # 10回以上精度が改善しなければ中止

          #use_best_model=True,       # 最も精度が高かったモデルを使用するかの設定

          #plot=True,                # 誤差の推移を描画するか否かの設定

          verbose=False)
model.score(test_X,test_y)
ft_imp=pd.DataFrame(model.get_feature_importance(),index=X.columns)

ft_imp.sort_values(0,ascending=False)
score_test=np.array([(i >= test_y)*1 for i in range(-99,100)])
pred_y=model.predict(test_X)
from scipy.stats import norm 
yard = np.arange(-99, 100) 

pred_prob = [norm.cdf(yard, loc=i, scale=target.std()) for i in pred_y] 
import matplotlib.pyplot as plt



#pred_probをdataframeにする

pred_prob2=pd.DataFrame(pred_prob)



#累積確率曲線の表示

plt.plot(yard,pred_prob2.mean())

plt.show()
#score=np.array([(i >= pred_y)*1 for i in range(-99,100)])
c=((pred_prob - score_test.T)**2).sum().sum()/(199*len(pred_prob))

c
train_df=rusher_df.iloc[:0,:]
#yard=['Yards' + str(i) for i in range(-99,100)]
for (test_df, sample_prediction_df) in tqdm(iter_test):

    #new_df=test_df.groupby(['PlayId','Position']).count()

    #position_count=new_df['GameId'].unstack().fillna(0).astype(int)

    rusher_df=test_df[test_df['NflId']==test_df['NflIdRusher']]

    rusher_df=preprocess(rusher_df)

    test_df=preprocess(test_df)

    #test_df=count_position(test_df)

    rusher_df=rusher_df.merge(rusher_yards,  on="NflId", how="left")

    rusher_df = rusher_df.merge(team_yards_df,on='PossessionTeam',how="left")

    rusher_df=add_team_score(rusher_df)

    rusher_df=count_yard_to_touchdown(rusher_df)

    rusher_df=add_average_data(test_df,rusher_df)

    rusher_df=pd.concat([train_df,test_df],sort=False)

    test_feature=feature(rusher_df)

    test_feature=test_feature.fillna(0)

    test_X=normalize(test_feature)

    pred_y=model.predict(test_X)

    pred_y=np.round(pred_y)

    pred_prob =norm.cdf(yard, loc=pred_y[0], scale=target.std()) 

    sample_prediction_df.iloc[0,:]=pred_prob

    env.predict(sample_prediction_df)
sample_prediction_df
env.write_submission_file()
import os

print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])