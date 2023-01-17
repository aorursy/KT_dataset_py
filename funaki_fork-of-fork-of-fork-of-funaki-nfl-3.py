from kaggle.competitions import nflrush

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor as RFR

from tqdm import tqdm



env = nflrush.make_env()
from sklearn.model_selection import GridSearchCV,train_test_split
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
iter_test = env.iter_test()
new_df=df.groupby(['PlayId','Position']).count()
position_count=new_df['GameId'].unstack().fillna(0).astype(int)

position_count
rusher_df=df[df['NflId']==df['NflIdRusher']]
rusher_df.columns
#チーム毎の獲得ヤード数の平均を見る

team_yards_df = rusher_df.groupby('Team').mean()[['Yards']]

team_yards_df = team_yards_df.rename(columns={'Yards':'team_yards'})
team_yards_df
#rusherのみのデータにチーム毎の平均獲得ヤード数を加える

rusher_df = rusher_df.merge(team_yards_df,on='Team',how="left")
# 攻撃チームの得点

rusher_df.loc[rusher_df["Team"]=="home", "rusherTeamScore"] = rusher_df["HomeScoreBeforePlay"]

rusher_df.loc[rusher_df["Team"]=="away", "rusherTeamScore"] = rusher_df["VisitorScoreBeforePlay"]



# 守備チームの得点

rusher_df.loc[rusher_df["Team"]=="home", "defenceTeamScore"] = rusher_df["VisitorScoreBeforePlay"]

rusher_df.loc[rusher_df["Team"]=="away", "defenceTeamScore"] = rusher_df["HomeScoreBeforePlay"]



# 得点差

rusher_df.loc[:, "diffScore"] = rusher_df["rusherTeamScore"] - rusher_df["defenceTeamScore"]
#rusherのみのデータにポジション毎の人数を加える

rusher_df=pd.merge(rusher_df,position_count, on='PlayId')
rusher_df=rusher_df.rename(columns={'S_x':'S','S_y':'S_position'})
# 選手毎の平均獲得ヤード

rusher_yards = rusher_df[["NflId", "Yards"]].groupby("NflId").mean()["Yards"]

rusher_yards.dropna(inplace=True)
rusher_id = rusher_df.merge(rusher_yards, on="NflId", how="left")
def process(df):

    df=pd.get_dummies(df,columns=['Team','PlayDirection','OffenseFormation'])

    df=df[(df['StadiumType']!='Cloudy') & (df['StadiumType']!='Bowl')]

    df=df.drop('FieldPosition', axis=1)

    outdoor=['Outdoor', 'Outdoors','Open','Indoor, Open Roof','Outdoor Retr Roof-Open', 'Oudoor', 'Ourdoor','Retr. Roof-Open','Outdor',

       'Retr. Roof - Open', 'Domed, Open', 'Domed, open', 'Outside','Heinz Field']

    indoor=['Indoors', 'RetractableRoof', 'Indoor','Retr. Roof-Closed','Dome', 'Domed, closed','Indoor, Roof Closed', 'Retr. Roof Closed','Closed Dome','Dome, closed','Domed']

    df['stadiumtype']=(df['StadiumType'].isin(outdoor)*1)

    rain=['Light Rain', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.','Rain', 'Heavy lake effect snow','Snow', 'Cloudy, Rain','Rain shower','Rainy']

    df['weather']=(~df['GameWeather'].isin(rain)*1)

    df['PlayerHeight']= df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))

    df['gameclock']=[ pd.Timedelta(val).total_seconds() for val in df['GameClock']]

    return df
rusher_df=process(rusher_df)
rusher_df=rusher_df.dropna()
def feature(df):

    features=pd.DataFrame(df,columns=['X', 'Y', 'S', 'A', 'Dis','Dir','YardLine', 'Quarter',

       'gameclock', 'Down', 'Distance','HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'DefendersInTheBox','PlayerHeight',

       'PlayerWeight','Temperature', 'Humidity', 'Team_home', 'stadiumtype', 'weather','PlayDirection_right',

       'OffenseFormation_ACE',

       'OffenseFormation_I_FORM', 'OffenseFormation_JUMBO',

       'OffenseFormation_PISTOL', 'OffenseFormation_SHOTGUN',

       'OffenseFormation_SINGLEBACK', 'OffenseFormation_WILDCAT', 'C', 'CB', 'DB',

       'DE', 'DL', 'DT', 'FB', 'FS', 'G', 'HB', 'ILB', 'LB', 'MLB', 'NT', 'OG',

       'OLB', 'OT', 'QB', 'RB', 'S_position', 'SAF', 'SS', 'T', 'TE', 'WR',"rusherTeamScore","defenceTeamScore","rusherTeamScore","diffScore",'team_yards'])

    return features

    



                      
features=feature(rusher_df)
train_mean=features.mean(axis=0)

train_mean
train_std=features.std(axis=0)

train_std
def normalize(features):

    X=(features-train_mean)/train_std

    return X
X=normalize(features)
target=pd.Series(rusher_df['Yards'])
train_X,test_X,train_y,test_y=train_test_split(X,target,test_size=0.2)
features.shape
search_params = {

    'n_estimators'      : [300],

    'max_features'      : [3,10,20,40,'auto'],

    #'random_state'      : [1],

    #'n_jobs'            : [1],

    #'min_samples_split' : [10, 20, 30],

    'max_depth'         : [3]

}

 

gsr = GridSearchCV(

    RFR(),

    search_params,

    cv = 3,

    n_jobs = -1,

    verbose=True

)

 

gsr.fit(train_X, train_y)
print(gsr.best_score_)

print(gsr.best_estimator_)

print(gsr.best_params_)
def train_predict(X,target):

    clf = RFR(bootstrap=True, criterion='mse', max_depth=3,

                      max_features=40, max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, n_estimators=300,

                      n_jobs=None, oob_score=False, random_state=None,

                      verbose=0, warm_start=False)

    clf.fit(X, target)

    return clf

    
clf=train_predict(train_X,train_y)
score_test=np.array([(i >= test_y)*1 for i in range(-99,100)])
pred_y=clf.predict(test_X)
from scipy.stats import norm 
yard = np.arange(-99, 100) 

pred_prob = [norm.cdf(yard, loc=i, scale=target.std()) for i in pred_y] 

pred_prob
#score=np.array([(i >= pred_y)*1 for i in range(-99,100)])
c=((pred_prob - score_test.T)**2).sum().sum()/(199*len(pred_prob))

c
#train_df=rusher_df.iloc[:0,:]
#yard=['Yards' + str(i) for i in range(-99,100)]
for (test_df, sample_prediction_df) in tqdm(iter_test):

    new_df=test_df.groupby(['PlayId','Position']).count()

    position_count=new_df['GameId'].unstack().fillna(0).astype(int)

    test_df=test_df.merge(position_count, on='PlayId')

    test_df=test_df.rename(columns={'S_x':'S','S_y':'S_position'})

    test_df=test_df[test_df['NflId']==test_df['NflIdRusher']]

    test_df = test_df.merge(team_yards_df,on='Team',how="left")

    # 攻撃チームの得点

    test_df.loc[test_df["Team"]=="home", "rusherTeamScore"] = test_df["HomeScoreBeforePlay"]

    test_df.loc[test_df["Team"]=="away", "rusherTeamScore"] = test_df["VisitorScoreBeforePlay"]

    # 守備チームの得点

    test_df.loc[test_df["Team"]=="home", "defenceTeamScore"] = test_df["VisitorScoreBeforePlay"]

    test_df.loc[test_df["Team"]=="away", "defenceTeamScore"] = test_df["HomeScoreBeforePlay"]

    # 得点差

    test_df.loc[:, "diffScore"] = test_df["rusherTeamScore"] - test_df["defenceTeamScore"]

    test_df=process(test_df)

    test_df=test_df.fillna(0)

    test_feature=feature(test_df)

    test_feature=test_feature.fillna(0)

    test_X=normalize(test_feature)

    pred_y=clf.predict(test_X)

    pred_y=np.round(pred_y)

    yard = np.arange(-99, 100) 

    pred_prob =norm.cdf(yard, loc=pred_y[0], scale=target.std()) 

    sample_prediction_df.iloc[0,:]=pred_prob

    env.predict(sample_prediction_df)
sample_prediction_df
env.write_submission_file()
import os

print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])