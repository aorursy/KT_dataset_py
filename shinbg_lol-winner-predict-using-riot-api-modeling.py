# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

import gc

from tqdm import tqdm_notebook # 오래 걸리는 작업 진행확인용

import warnings

warnings.filterwarnings(action='ignore')
pd.options.display.max_rows = 100
match = pd.read_pickle("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/matchpre.pkl")



win_team_stat = pd.read_csv("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/win_team_stats.csv")



lose_team_stat = pd.read_csv("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/lose_team_stats.csv")



win_team = pd.read_pickle("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/match_winner_data.pkl")



lose_team = pd.read_pickle("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/match_lose_data.pkl")



date = pd.read_csv("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/lol_version_Date.csv")
"""

팀 스탯 row 와 팀 기록 row 일치시키기

"""

gameId = win_team_stat["gameId"]

match = pd.merge(gameId,match,how="inner",on="gameId")

win_team = pd.merge(gameId,win_team,how="inner",on="gameId")

lose_team = pd.merge(gameId,lose_team,how="inner",on="gameId")



"""

duplicated() 함수는 리스트에 대한 중복검사를 지원하지 않으므로 리스트 칼럼 삭제

"""

match.drop("participants",axis=1,inplace=True)



win_team.drop("bans",axis=1,inplace=True)



lose_team.drop("bans",axis=1,inplace=True)



match = match.drop_duplicates()

win_team = win_team.drop_duplicates()

lose_team = lose_team.drop_duplicates()



win_team_stat = win_team_stat.drop_duplicates()

lose_team_stat = lose_team_stat.drop_duplicates()



"""

win_team, lose_team으로 나누어 모든 테이블 병합

"""

win_team = pd.merge(match,win_team,how="left",on="gameId")

win_team = pd.merge(win_team,win_team_stat,how="left",on="gameId")



lose_team = pd.merge(match,lose_team,how="left",on="gameId")

lose_team = pd.merge(lose_team,lose_team_stat,how="left",on="gameId")



del match

gc.collect()



"""

팀 데이터셋을 전체 데이터로 병합하기 전에 칼럼명 일치시키기

"""

win_team.columns = win_team.columns.str.replace("win_","")



lose_team.columns = lose_team.columns.str.replace("lose_","")



"""

전체 게임 정보 gamedata 테이블 생성

"""

gamedata = pd.concat([win_team,lose_team])

gamedata = gamedata.reset_index()

gamedata.drop("index",axis=1,inplace=True)

gamedata = gamedata.astype({"gameVersion":int})

gamedata = pd.merge(gamedata,date,how="inner",on="gameVersion")



del win_team

del lose_team

gc.collect()



"""

카테고리형 데이터(True,False) Label encoding / 분석에 용이하도록 가공

"""

bool_mapping = {True:1,False:0}

bool_col = gamedata.select_dtypes('bool').columns.tolist()



for col in bool_col:

    gamedata[col] = gamedata[col].map(bool_mapping)

    

win_mapping = {"Win":1,"Fail":0}

gamedata["win"] = gamedata["win"].map(win_mapping)



gamedata["date"] = pd.to_datetime(gamedata["date"])

gamedata["gameId"] = gamedata.astype({"gameId":object})
gamedata
#예측에 필요없는 칼럼 드랍(승패예측에 영향을 끼치지 못함)

gamedata.drop(["gameVersion","vilemawKills","dominionVictoryScore","date"],axis=1,inplace=True)
pd.set_option('display.float_format', '{:.5f}'.format) # 항상 float 형식으로
#승패에 영향이 가는 상관계수 분석

gamedata.corr()["win"].sort_values()
gamedata.head()
#팀의 킬카운트 총합

gamedata["team_kills"] = gamedata["kills1"] + gamedata["kills2"] + gamedata["kills3"] + gamedata["kills4"] + gamedata["kills5"]



#팀의 데스카운트 총합

gamedata["team_deaths"] = gamedata["deaths1"] + gamedata["deaths2"] + gamedata["deaths3"] + gamedata["deaths4"] + gamedata["deaths5"]



#팀이 획득한 총 골드 - 제외됨

#gamedata["team_goldEarned"] = gamedata["goldEarned1"] + gamedata["goldEarned2"] + gamedata["goldEarned3"] + gamedata["goldEarned4"] + gamedata["goldEarned5"]



#팀이 가한 총 피해량

gamedata["team_totalDamageDealtToChampions"] = gamedata["totalDamageDealtToChampions1"] + gamedata["totalDamageDealtToChampions2"] + gamedata["totalDamageDealtToChampions3"] + gamedata["totalDamageDealtToChampions4"] + gamedata["totalDamageDealtToChampions5"]



#팀이 가한 총 CC기 시간

gamedata["team_totalTimeCrowdControlDealt"] = gamedata["totalTimeCrowdControlDealt1"] + gamedata["totalTimeCrowdControlDealt2"] + gamedata["totalTimeCrowdControlDealt3"] + gamedata["totalTimeCrowdControlDealt4"] + gamedata["totalTimeCrowdControlDealt5"]



#팀의 총 시야점수

gamedata["team_visionScore"] = gamedata["visionScore1"] + gamedata["visionScore2"] + gamedata["visionScore3"] + gamedata["visionScore4"] + gamedata["visionScore5"]



#팀이 처치한 총 오브젝트 갯수 - 제외됨

#gamedata["team_Object"] = (gamedata["riftHeraldKills"] + gamedata["baronKills"] + gamedata["dragonKills"] + gamedata["inhibitorKills"] + gamedata["towerKills"])
#천상계는 전체적으로 하위 티어보다 킬, 데스의 분포 범위가 다름을 고려, 킬/데스의 비율로 킬데스를 반영.

def kdc(df):

    if df["team_deaths"]==0:

        return df["team_kills"]/(df["team_deaths"]+1)*1.2 #만약 팀의 총 데스가 0일경우 퍼펙트 게임을 적용해 가중치 1.2 적용

    return df["team_kills"]/df["team_deaths"]
#팀의 킬/데스 지표

gamedata["team_K/D"] = gamedata.apply(kdc,axis=1)
#승패에 영향이 가는 상관계수 분석

gamedata.corr()["win"].sort_values()
"""

차이 칼럼들은 다른 변수들과 다중공선성 문제가 크게 생기고 회귀결과에 미치는 영향이 너무 커서 제외함.

"""



# from sklearn.preprocessing import LabelEncoder

# gamedata["gameId"] = LabelEncoder().fit_transform(gamedata["gameId"])



# #gameId, win을 기준으로 데이터프레임화

# gamedata = gamedata.sort_values(by=["gameId","win"])



# gamedata_win = gamedata[1::2]



# gamedata_lose = gamedata[::2]



# gamedata_win.reset_index(drop=True,inplace=True)



# gamedata_lose.reset_index(drop=True,inplace=True)



# #승리팀, 진팀의 골드 차이

# gamedata_win["team_goldDiff"] = gamedata_win["team_goldEarned"] - gamedata_lose["team_goldEarned"]

# gamedata_lose["team_goldDiff"] = gamedata_lose["team_goldEarned"] - gamedata_win["team_goldEarned"]



# #승리팀, 진팀의, 딜량 차이

# gamedata_win["team_dealtDiff"] = gamedata_win["team_totalDamageDealtToChampions"] - gamedata_lose["team_totalDamageDealtToChampions"]

# gamedata_lose["team_dealtDiff"] = gamedata_lose["team_totalDamageDealtToChampions"] - gamedata_win["team_totalDamageDealtToChampions"]



# gamedata = pd.concat([gamedata_win,gamedata_lose],axis=0)



# gamedata.reset_index(drop=True,inplace=True)



# #승패에 영향이 가는 상관계수 분석

# gamedata.corr()["win"].sort_values()
#게임시간(초) -> 분

gamedata["gameMinute"] = gamedata["gameDuration"] / 60



#다시하기 경기 제외

gamedata = gamedata[gamedata["gameMinute"] > 5]
#bins = [0, ,15, 20, 25, 30, 35]

bins = [0, 20, 25, 30]
gamedata["time_bin"] = np.digitize(gamedata["gameMinute"],bins)

gamedata["time_bin"].value_counts()
game_part1 = gamedata[gamedata["time_bin"] == 1]

game_part2 = gamedata[gamedata["time_bin"] == 2]

game_part3 = gamedata[gamedata["time_bin"] == 3]

game_part4 = gamedata[gamedata["time_bin"] == 4]



game_part1.drop(["gameMinute","gameDuration"],axis=1,inplace=True)

game_part2.drop(["gameMinute","gameDuration"],axis=1,inplace=True)

game_part3.drop(["gameMinute","gameDuration"],axis=1,inplace=True)

game_part4.drop(["gameMinute","gameDuration"],axis=1,inplace=True)
display(game_part1.corr()["win"].sort_values(),



game_part2.corr()["win"].sort_values(),



game_part3.corr()["win"].sort_values(),



game_part4.corr()["win"].sort_values())
#분당 팀의 킬,데스 스코어



gamedata["team_kills_per_minute"] = gamedata["team_kills"] / gamedata["gameMinute"]



gamedata["team_deaths_per_minute"]= gamedata["team_deaths"] / gamedata["gameMinute"]



gamedata["team_K/D_per_minute"] = gamedata["team_K/D"] / gamedata["gameMinute"]



#분당 팀이 가한 총 데미지

gamedata["team_totalDamageDealt_per_minute"] = gamedata["team_totalDamageDealtToChampions"] / gamedata["gameMinute"]



#분당 팀이 가한 CC기 시간

gamedata["team_totalTimeCrowdControlDealt_per_minute"] =  gamedata["team_totalTimeCrowdControlDealt"] / gamedata["gameMinute"]



#분당 팀의 시야 점수

gamedata["team_visionScore_per_minute"] = gamedata["team_visionScore"] / gamedata["gameMinute"]



#분당 팀의 골드 획득량 - 제외됨

#gamedata["team_goldEarned_per_minute"] = gamedata["team_goldEarned"] / gamedata["gameMinute"]



#분당 팀의 총 오브젝트 스코어량 - 제외됨

#gamedata["team_object_per_minute"] = gamedata["team_Object"] / gamedata["gameMinute"]



#분당 팀의 타워 파괴량

gamedata["towerKills_per_minute"] = gamedata["towerKills"] / gamedata["gameMinute"]



#분당 팀의 억제기 파괴량

gamedata["inhibitorKills_per_minute"] = gamedata["inhibitorKills"] /gamedata["gameMinute"]



#분당 팀의 바론 처치량

gamedata["baronKills_per_minute"] = gamedata["baronKills"] / gamedata["gameMinute"]



#분당 팀의 드래곤 처치량

gamedata["dragonKills_per_minute"] = gamedata["dragonKills"] / gamedata["gameMinute"]



#분당 팀의 전령 처치량

gamedata["riftHeraldKills_per_minute"] = gamedata["riftHeraldKills"] / gamedata["gameMinute"]
#ex) 15분에 타워 10개 파괴 ->

10/15
#ex) 30분에 타워 10개 파괴 ->

10/30
#inf 값 검사용

np.where(gamedata.values >= np.finfo(np.float64).max)
#승패에 영향이 가는 상관계수 분석

gamedata.corr()["win"].sort_values()
gamedata.drop("gameId",axis=1,inplace=True)



gamedata["time_bin"] = gamedata["time_bin"].astype("object")

gamedata = pd.get_dummies(gamedata)
import seaborn as sns    

plt.figure(figsize= (20, 10))

sns.heatmap(gamedata.corr())
personal_col = ['kills1', 'kills2', 'kills3', 'kills4', 'kills5','deaths1', 'deaths2', 'deaths3', 'deaths4', 'deaths5','totalDamageDealtToChampions1', 'totalDamageDealtToChampions2',

       'totalDamageDealtToChampions3', 'totalDamageDealtToChampions4',

       'totalDamageDealtToChampions5','goldEarned1', 'goldEarned2',

       'goldEarned3', 'goldEarned4', 'goldEarned5','visionScore1',

       'visionScore2', 'visionScore3', 'visionScore4', 'visionScore5',

       'totalTimeCrowdControlDealt1', 'totalTimeCrowdControlDealt2',

       'totalTimeCrowdControlDealt3', 'totalTimeCrowdControlDealt4',

       'totalTimeCrowdControlDealt5']



gamedata.drop(personal_col,axis=1,inplace=True)
import seaborn as sns    

plt.figure(figsize= (20, 10))

sns.heatmap(gamedata.corr())
gamedata.columns
team_col = ['towerKills','inhibitorKills','baronKills','dragonKills','riftHeraldKills','team_kills', 'team_deaths','team_totalDamageDealtToChampions', 

            'team_totalTimeCrowdControlDealt','team_visionScore','team_K/D']



gamedata.drop(team_col,axis=1,inplace=True)



#오브젝트 갯수를 모두 합한 이 칼럼은 중복적인 요소가 많아 제거

#gamedata.drop("team_object_per_minute",axis=1,inplace=True)



#가설에서 직관적인 지표를 피드백해야 하고, 각 지표가 올라갈수록 골드량이 많아지는 것은 당연하기 때문에 해당 칼럼은 가설을 검정하기에 좋지 않음.

#gamedata.drop("team_goldEarned_per_minute",axis=1,inplace=True)
import seaborn as sns    

plt.figure(figsize= (20, 10))

sns.heatmap(gamedata.corr())
gamedata.drop(["gameDuration","gameMinute"],axis=1,inplace=True)
#승패에 영향이 가는 상관계수 분석

gamedata.corr()["win"].sort_values()
gamedata
#VIF 분석으로 최종 검사 , 10 이상일 경우 다중공선성 문제가 생길 수 있음.

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(

    gamedata.values, i) for i in range(gamedata.shape[1])]

vif["features"] = gamedata.columns

vif
gamedata.shape
x_data = gamedata.drop("win",axis=1)

y_data = gamedata["win"]
x_data.info()
#사용하는 Input column

use_col = list(x_data.columns)

use_col
import statsmodels.api as sm



logit = sm.Logit(y_data,x_data) #로지스틱 회귀분석 시행

result = logit.fit()

result.summary2()
#모든 변수를 하나의 모델에서 해석

for i in range(len(result.params)):

    print('다른 변수가 변하지 않을 때, {} 이 한단위 상승하면 승리할 확률이 {:.5f} 배 증가한다.\n'.format(result.params.keys()[i],np.exp(result.params.values[i])))
from sklearn.model_selection import train_test_split



#훈련셋, 테스트셋 계층 샘플링 - 편향 방지

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=7, stratify=y_data) 
#각 feature의 평균을 0, 분산을 1로 변경 - 특성들을 모두 동일한 스케일로 반영



from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

ss.fit(x_train)

x_train = ss.transform(x_train)

x_test = ss.transform(x_test)
import statsmodels.api as sm



logit = sm.Logit(y_train,x_train) #로지스틱 회귀분석 시행

result = logit.fit()

result.summary2()
#모든 변수를 하나의 모델에서 해석

for i in range(len(result.params)):

    print('다른 변수가 변하지 않을 때, {} 이 1 상승하면 승리할 확률이 {:.5f} 배 증가한다.\n'.format(use_col[i],np.exp(result.params.values[i])))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
%%time

#그리드 서치로 하이퍼파라미터 튜닝, 계층별 10차 교차 검증 수행

param_grid = {"C" : [0.001,0.01,0.1,1,10,10,100]} #규제의 강도만 조절하고, 방식은 바꾸지 않는다. 조금이라도 승패에 영향이 가는 변수를 모두 반영하기 위해 L2(릿지)norm을 적용.

grid_search = GridSearchCV(LogisticRegression(),param_grid,cv=10,return_train_score=True,n_jobs=-1)



grid_search.fit(x_train,y_train)
coef = pd.DataFrame(data={"coef":grid_search.best_estimator_.coef_.tolist()[0],"col":use_col})

coef
for i in range(len(coef)):

    print('다른 변수가 변하지 않을 때, {} 이 1 상승하면 승리할 확률이 {:.5f} 배 증가한다.\n'.format(use_col[i],np.exp(coef.iloc[i,0])))
print("최적 파라미터 : {}".format(grid_search.best_params_))



print("훈련세트 정확도(교차 검증 정확도 평균) : {}".format(grid_search.best_score_))



print(grid_search.best_estimator_)
#결과표시

np.set_printoptions(suppress=True)



pred_y_test = grid_search.predict(x_test)

proba_y_test = grid_search.predict_proba(x_test)

display(pred_y_test,proba_y_test)
print("테스트세트 정확도 : {}".format(grid_search.score(x_test,y_test)))

log_cls_score = grid_search.score(x_test,y_test)
#f1 스코어

from sklearn.metrics import f1_score

score = f1_score(y_test,pred_y_test)

print("테스트세트 f1 점수 : {}".format(score))
#실제 모델들과 비교하기 위한 무작위 랜덤모델

from sklearn.dummy import DummyClassifier

dm = DummyClassifier(strategy="stratified").fit(x_train,y_train)

dm.score(x_test,y_test)
# %%time

# from sklearn.ensemble import RandomForestClassifier

# #그리드 서치로 하이퍼파라미터 튜닝, 계층별 5차 교차 검증 수행

# param_grid = {"n_estimators" : [10,50,100,500],

#              "criterion": ["entropy","gini"]}



# rf = RandomForestClassifier(random_state=7,n_jobs=-1)

# grid_search = GridSearchCV(rf,param_grid,cv=5,return_train_score=True)



# grid_search.fit(x_train,y_train)



# print("최적 파라미터 : {}".format(grid_search.best_params_))



# print("훈련세트 정확도(교차 검증 정확도 평균) : {}".format(grid_search.best_score_))



# print(grid_search.best_estimator_)



# print("테스트세트 정확도 : {}".format(grid_search.score(x_test,y_test)))

# rf_cls_score = grid_search.score(x_test,y_test)



# result = pd.DataFrame(grid_search.cv_results_)

# result



#최적 파라미터 : {'criterion': 'entropy', 'n_estimators': 100}

#훈련세트 정확도(교차 검증 정확도 평균) : 0.9791780438283274
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=7,n_jobs=-1)

rf.fit(x_train,y_train)
print("훈련세트 정확도 : {}".format(rf.score(x_train,y_train)))



print("테스트세트 정확도 : {}".format(rf.score(x_test,y_test)))
#특성 중요도 분석

import seaborn as sns



plt.figure(figsize=(20,12))

feat_imp = {'col' : x_data.columns,

            'importances' : rf.feature_importances_}



feat_imp = pd.DataFrame(feat_imp).sort_values(by = 'importances', ascending = False)



sns.barplot(y = feat_imp['col'] ,

            x = feat_imp['importances'])
from catboost import CatBoostClassifier



cbr = CatBoostClassifier(verbose=50,task_type="GPU")

cbr.fit(x_train,y_train)
print("훈련세트 정확도 : {}".format(cbr.score(x_train,y_train)))



print("테스트세트 정확도 : {}".format(cbr.score(x_test,y_test)))
#특성 중요도 분석

import seaborn as sns



f,ax = plt.subplots(1,2,figsize=(20,12))

rf_imp = {'col' : x_data.columns,

            'rf_importances' : rf.feature_importances_}



rf_imp = pd.DataFrame(rf_imp).sort_values(by = 'rf_importances', ascending = False)



sns.barplot(y = rf_imp['col'] ,

            x = rf_imp['rf_importances'],ax=ax[0])



cbr_imp = {'col' : x_data.columns,

            'cbr_importances' : cbr.feature_importances_}



cbr_imp = pd.DataFrame(cbr_imp).sort_values(by = 'cbr_importances', ascending = False)



sns.barplot(y = cbr_imp['col'] ,

            x = cbr_imp['cbr_importances'],ax=ax[1])
# from xgboost import XGBClassifier



# xgb = XGBClassifier()



# param_grid = {'objective':['binary:logistic'],

#               'learning_rate': [0.025], #catboost에서 측정된 적절 eta

#               'max_depth': [2,3,5],

#               'min_child_weight': [1,5,10], #최소 가중치 합 : 높게하면 언더피팅

#               'colsample_bytree': [0.7],

#               'n_estimators': [10,50,100,1000],

#               'seed': [7]}



# grid_search = GridSearchCV(xgb, param_grid, n_jobs=-1, 

#                    cv=5, 

#                    verbose=2, refit=True)



# grid_search.fit(x_train, y_train)



#최적 파라미터 : {'objective':['binary:logistic'],

#               'learning_rate': [0.025], #catboost에서 측정된 적절 eta

#               'max_depth': [3],

#               'min_child_weight': [10], #최소 가중치 합 : 높게하면 언더피팅

#               'colsample_bytree': [0.7],

#               'n_estimators': [1000],

#               'seed': [7]}

#훈련세트 정확도(교차 검증 정확도 평균) : 0.9795003920690042
from xgboost import XGBClassifier

xgb = XGBClassifier(

    objective='binary:logistic',

    max_depth=3,

    n_estimators=1000,

    min_child_weight=10, 

    colsample_bytree=0.7, 

    eta=0.025,

    nthread=-1,

    seed=7)



xgb.fit(x_train, y_train, verbose=True)
print("훈련세트 정확도 : {}".format(xgb.score(x_train,y_train)))



print("테스트세트 정확도 : {}".format(xgb.score(x_test,y_test)))
xgb.feature_importances_
#특성 중요도 분석

import seaborn as sns



f,ax = plt.subplots(1,3,figsize=(20,12))

rf_imp = {'col' : x_data.columns,

            'rf_importances' : rf.feature_importances_}



rf_imp = pd.DataFrame(rf_imp).sort_values(by = 'rf_importances', ascending = False)



sns.barplot(y = rf_imp['col'] ,

            x = rf_imp['rf_importances'],ax=ax[0])



cbr_imp = {'col' : x_data.columns,

            'cbr_importances' : cbr.feature_importances_}



cbr_imp = pd.DataFrame(cbr_imp).sort_values(by = 'cbr_importances', ascending = False)



sns.barplot(y = cbr_imp['col'] ,

            x = cbr_imp['cbr_importances'],ax=ax[1])



xgb_imp = {'col' : x_data.columns,

            'xgb_importances' : xgb.feature_importances_}



xgb_imp = pd.DataFrame(xgb_imp).sort_values(by = 'xgb_importances', ascending = False)



sns.barplot(y = xgb_imp['col'] ,

            x = xgb_imp['xgb_importances'],ax=ax[2])
cbr.save_model("LOL_predict_cbr.cbm")
xgb.save_model("LOL_predict_xgb.bst")
cbr = CatBoostClassifier()

cbr.load_model("LOL_predict_cbr.cbm")
xgb = XGBClassifier({'nthread': 4})

xgb.load_model('LOL_predict_xgb.bst')