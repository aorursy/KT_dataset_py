import numpy as np

import pandas as pd



import math

import scipy

from scipy import stats

from scipy.spatial.distance import euclidean

from scipy.special import expit



from tqdm import tqdm



import random

import warnings

warnings.filterwarnings("ignore")



from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', parse_dates=['TimeHandoff','TimeSnap'], infer_datetime_format=True, low_memory=False)
df_train.columns
meta_df = pd.DataFrame({})



meta_df = meta_df.append([['GameId','比赛ID','分类','比赛','无','','测试与训练中的GameId都是不同的']])

meta_df = meta_df.append([['PlayId','回合ID','分类','比赛','无','','每个待预测数据都是一个唯一的回合Id']])

meta_df = meta_df.append([['Team','球队','分类','球队','中','','不同的球队有不同的进攻防守能力']])

meta_df = meta_df.append([['X','球员位置x','数值','球员动态','高','','球员位置决定了战术执行顺利与否']])

meta_df = meta_df.append([['Y','球员位置y','数值','球员动态','高','','球员位置决定了战术执行顺利与否']])

meta_df = meta_df.append([['S','球员速度','数值','球员动态','高','','最直接的说rusher的速度与码数的关系是很直接的']])

meta_df = meta_df.append([['A','球员加速度','数值','球员动态','高','','最直接的说rusher的加速度与码数的关系是很直接的']])

meta_df = meta_df.append([['Dis','','数值','球员动态','中','','']])

meta_df = meta_df.append([['Orientation','球员面向角度','数值','球员动态','低','','表现球员的观察方向，或者在更高级的维度会更有用']])

meta_df = meta_df.append([['Dir','球员移动角度','数值','球员动态','中','','移动方向感觉直接的作用不如间接的大']])

meta_df = meta_df.append([['NflId','球员Id','分类','球员静态','中','','根据具体球员能力不同，最终达成的码数不同']])

meta_df = meta_df.append([['DisplayName','球衣名字','分类','球员静态','无','','基本没用，更多是在可视化部分起作用']])

meta_df = meta_df.append([['JerseyNumber','球员号码','分类','球员静态','无','','一般决定了位置，但是位置有单独的字段，所以也没啥用']])

meta_df = meta_df.append([['Season','赛季','分类','比赛','无','','赛季看起来范围太大，应该没什么用']])

meta_df = meta_df.append([['YardLine','码线','分类','比赛','低','','看过比赛后我觉得码线有影响但是不大，是不是在终场前且分差很接近时会有呢']])

meta_df = meta_df.append([['Quarter','第几节','分类','比赛','低','','不认为第几节会有太大关系']])

meta_df = meta_df.append([['GameClock','比赛时间','时间','比赛','低','','同样不认为时间关系会很大']])

meta_df = meta_df.append([['PossessionTeam','进攻方','分类','比赛','中','','球队进攻防守能力有关']])

meta_df = meta_df.append([['Down','1~4 down','分类','比赛','中','','影响战术，通常如果是1选择保守，2,3会进攻性强一些，4则弃踢多']])

meta_df = meta_df.append([['Distance','需要多少码可以继续进攻','数值','比赛','中','','与Donw的关系类似']])

meta_df = meta_df.append([['FieldPosition','目前比赛在哪个队半场进行','分类','比赛','低','','这个信息在码线上也有体现']])

meta_df = meta_df.append([['HomeScoreBeforePlay','主队赛前积分','数值','球队','低','','这主要是体现球队实力']])

meta_df = meta_df.append([['VisitorScoreBeforePlay','客队赛前积分','数值','球队','低','','这主要是体现球队实力，影响应该是总体的']])

meta_df = meta_df.append([['NflIdRusher','持球人Id','分类','比赛','中','','持球人的影响肯定是单个特征中最大的']])

meta_df = meta_df.append([['OffenseFormation','进攻队形','分类','比赛','中','','不同的进攻方式通常目的是不同的，对应的码数也不同']])

meta_df = meta_df.append([['OffensePersonnel','进攻人员组成','分类','比赛','中','','这个主要是配合队形使用，可以认为是队形的更细分']])

meta_df = meta_df.append([['DefendersInTheBox','防守方在混战区的人数','数值','比赛','高','','这个人数跟战术有关，感觉有关系，其他kernel看关系还挺大']])

meta_df = meta_df.append([['DefensePersonnel','防守人员组成','分类','比赛','中','','防守人员是针对进攻人员来设置的']])

meta_df = meta_df.append([['PlayDirection','回合进行的方向','分类','比赛','无','','比赛方向，左还是右，关系不大']])

meta_df = meta_df.append([['TimeHandoff','传球时间','时间','比赛','低','','可能跟战术有关，或者进展是否顺序，一般来说越快越好']])

meta_df = meta_df.append([['TimeSnap','发球时间','时间','比赛','无','','感觉不到有什么关系，与handoff求时间差吧']])

meta_df = meta_df.append([['PlayerHeight','球员身高','数值','球员静态','低','','太明显感觉不到，但是不能说没有']])

meta_df = meta_df.append([['PlayerWeight','球员体重','数值','球员静态','低','','太明显感觉不到，但是不能说没有*2']])

meta_df = meta_df.append([['PlayerBirthDate','球员生日','时间','球员静态','无','','直接用没用，但是可以转为年龄，但是关系应该也不太大']])

meta_df = meta_df.append([['PlayerCollegeName','球员大学','分类','球员静态','低','','关系不大，虽然说名校可能实力更大，但是不尽然']])

meta_df = meta_df.append([['Position','球员职责','分类','球员静态','低','','根据持球人的Position或者有不错的效果']])

meta_df = meta_df.append([['HomeTeamAbbr','主队名','分类','球队','低','','聚合统计球队进攻防守能力']])

meta_df = meta_df.append([['VisitorTeamAbbr','客队名','分类','球队','低','','聚合统计球队进攻防守能力']])

meta_df = meta_df.append([['Week','第几周','分类','比赛','无','','目前是第几周，或者会考虑疲劳，但是缩小到每个回合，关系不大']])

meta_df = meta_df.append([['Stadium','球场','分类','环境','无','','微乎其微']])

meta_df = meta_df.append([['Location','球场所在位置','分类','环境','低','','可能有气候问题，比如NBA的掘金所在的高原地区']])

meta_df = meta_df.append([['StadiumType','球场类型','分类','环境','无','','微乎其微']])

meta_df = meta_df.append([['Turf','草皮','分类','环境','无','','微乎其微']])

meta_df = meta_df.append([['GameWeather','比赛天气','分类','环境','无','','微乎其微']])

meta_df = meta_df.append([['Temperature','温度','数值','环境','无','','微乎其微']])

meta_df = meta_df.append([['Humidity','湿度','数值','环境','无','','微乎其微']])

meta_df = meta_df.append([['WindSpeed','风速','数值','环境','无','','微乎其微']])

meta_df = meta_df.append([['WindDirection','风向','数值','环境','无','','微乎其微']])

meta_df = meta_df.append([['Yards','所获得的码数','数值','比赛','目标','','该次回合进攻方获得的码数，理论上多为整数，少数为负数或零']])



meta_df.columns = ['name','desc','type','segment','expectation','conclusion','comment']

meta_df.sort_values(by='expectation')
df_train.Yards.describe()
sns.distplot(df_train.Yards);
print("Skewness: %f" % df_train.Yards.skew())

print("Kurtosis: %f" % df_train.Yards.kurt())
df_train_rusher = df_train[df_train.NflId==df_train.NflIdRusher].copy()



df_train_rusher['ToLeft'] = df_train_rusher.PlayDirection.apply(lambda play_direction:play_direction=='left')

df_train_rusher['X_std'] = df_train_rusher[['ToLeft','X']].apply(lambda row:120-row.X-10 if row.ToLeft else row.X-10, axis=1)

df_train_rusher['Y_std'] = df_train_rusher[['ToLeft','Y']].apply(lambda row:160/3-row.Y if row.ToLeft else row.Y, axis=1)
var = 'X_std'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

data.plot.scatter(x=var, y='Yards');
meta_df.loc[meta_df.name=='X','conclusion'] = '可以生成一个规则特征，用于对最终结果的限制，另外作为空间位置信息有待挖掘'
var = 'Y_std'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

data.plot.scatter(x=var, y='Yards');
meta_df.loc[meta_df.name=='Y','conclusion'] = '可以生成一个距离左右边界距离的特征，作为空间位置有待更深入的挖掘'
var = 'S'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

data.plot.scatter(x=var, y='Yards');
meta_df.loc[meta_df.name=='S','conclusion'] = '没有明显线性关系，大码数集中在中间部分，所以速度适中是个选项'
var = 'A'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

data.plot.scatter(x=var, y='Yards');
meta_df.loc[meta_df.name=='A','conclusion'] = '没有明显线性关系，大码数集中在左边，也就是说球员速度更平均时'
var = 'DefendersInTheBox'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

f, ax = plt.subplots(figsize=(12, 6))

fig = sns.boxplot(x=var, y="Yards", data=data)
meta_df.loc[meta_df.name=='DefendersInTheBox','conclusion'] = '不看超大码数时，人数越多，码数越小'
var = 'PossessionTeam'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

f, ax = plt.subplots(figsize=(20, 6))

fig = sns.boxplot(x=var, y="Yards", data=data)
meta_df.loc[meta_df.name=='PossessionTeam','conclusion'] = '球队层面对码数的影响不大，各只球队基本持平'
var = 'NflIdRusher'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

f, ax = plt.subplots(figsize=(30, 6))

fig = sns.boxplot(x=var, y="Yards", data=data)
meta_df.loc[meta_df.name=='NflIdRusher','conclusion'] = '球员层面区别很大，但是相对来说球员层面的数据量更小，所以是否具有代表性有待研究'
var = 'Down'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="Yards", data=data)
meta_df.loc[meta_df.name=='Down','conclusion'] = 'Down越大，代表所剩机会越少，一般码数都会越小'
var = 'OffenseFormation'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

f, ax = plt.subplots(figsize=(12, 6))

fig = sns.boxplot(x=var, y="Yards", data=data)
meta_df.loc[meta_df.name=='OffenseFormation','conclusion'] = '数据分布主体是类似的，大码数方向根据队形不同有所差异'
meta_df
#correlation matrix

corrmat = df_train_rusher.corr()

f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10

cols = corrmat.nlargest(k, 'Yards')['Yards'].index

cm = np.corrcoef(df_train_rusher[cols].values.T)

plt.subplots(figsize=(8, 8))

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['Yards', 'A', 'S', 'Distance', 'YardLine', 'Season', 'NflIdRusher']

sns.pairplot(df_train_rusher[cols], size = 2.5)

plt.show();
total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data.Total>0]
df_train = df_train.drop(['WindSpeed','WindDirection','Temperature','GameWeather','Humidity','StadiumType'], axis=1)
df_train.FieldPosition = df_train.FieldPosition.fillna('middle')
# OffenseFormation, Dir, Orientation, DefendersInTheBox

# df_train[]
df_train.isnull().sum().max()
from sklearn.preprocessing import StandardScaler

#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['Yards'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#histogram and normal probability plot

sns.distplot(df_train['Yards'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(df_train['Yards'], plot=plt)
df_train_rusher = df_train[df_train.NflId==df_train.NflIdRusher].copy()
plt.subplots(figsize=(16, 8))

tmp = df_train_rusher[["Y", "X", "Yards"]].copy()

tmp.X = pd.cut(tmp.X, 12*5)

tmp.Y = pd.cut(tmp.Y, 6*5)

sns.heatmap(tmp.groupby(['Y','X']).count().reset_index().pivot("Y", "X", "Yards"))
plt.subplots(figsize=(16, 8))

sns.heatmap(tmp.groupby(['Y','X']).mean().reset_index().pivot("Y", "X", "Yards"), center=0, vmin=-5, vmax=10)
meta_df[meta_df.conclusion.str.len()>0]
df_train_rusher.PlayerHeight = df_train_rusher.PlayerHeight.apply(lambda height:int(height[0])*12+int(height[2:])).astype('int')

df_train_rusher['Age'] = df_train_rusher.PlayerBirthDate.apply(lambda bd:2019-int(bd[-4:]))
plt.subplots(figsize=(20, 5))



plt.subplot(1,3,1)

var = 'PlayerHeight'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

#data.plot.scatter(x=var, y='Yards')

#axs[0][0].plot.scatter(data[var],data['Yards'])

plt.scatter(data[var],data['Yards'])



plt.subplot(1,3,2)

var = 'PlayerWeight'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

plt.scatter(data[var],data['Yards'])



plt.subplot(1,3,3)

var = 'Age'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

plt.scatter(data[var],data['Yards'])
df_train_rusher['Y_dis'] = np.abs(np.abs(df_train_rusher.Y - df_train_rusher.Y.mean()) - df_train_rusher.Y.mean())
var = 'Y_dis'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

data.plot.scatter(x=var, y='Yards');
df_train_rusher['Dir_orientation'] = np.abs(df_train_rusher.Dir - df_train_rusher.Orientation)

df_train_rusher['Dir_orientation'] = df_train_rusher['Dir_orientation'].apply(lambda do:360-do if do > 180 else do)
var = 'Dir_orientation'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

data.plot.scatter(x=var, y='Yards');
df_train_rusher['TimeFromSnapToHandoff'] = (df_train_rusher.TimeHandoff - df_train_rusher.TimeSnap).apply(

    lambda x:x.total_seconds()).astype('int8')
var = 'TimeFromSnapToHandoff'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

data.plot.scatter(x=var, y='Yards');
df_train_rusher['GameDuration'] = (df_train_rusher.GameClock.apply(

    lambda gc:15*60-int(gc[:2])*60-int(gc[3:5]))) + (df_train_rusher.Quarter-1)*15*60
var = 'GameDuration'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

data.plot.scatter(x=var, y='Yards');
var = 'Quarter'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="Yards", data=data)
df_train_rusher['DistanceTouchDown'] = df_train_rusher[['YardLine','FieldPosition','PossessionTeam']].apply(

    lambda yfp:100-yfp['YardLine'] if(yfp['PossessionTeam']==yfp['FieldPosition']) else yfp['YardLine'], axis=1)
var = 'DistanceTouchDown'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

data.plot.scatter(x=var, y='Yards');
# Create the DL-LB combos

# Clean up and convert to DL-LB combo

df_train_rusher['DL_LB'] = df_train_rusher['DefensePersonnel'].str[:10].str.replace(' DL, ','-').str.replace(' LB','')

top_5_dl_lb_combos = df_train_rusher.groupby('DL_LB').count()['GameId'].sort_values().tail(10).index.tolist()
var = 'DL_LB'

data = pd.concat([df_train_rusher.loc[df_train_rusher['DL_LB'].isin(top_5_dl_lb_combos)].Yards, 

                  df_train_rusher.loc[df_train_rusher['DL_LB'].isin(top_5_dl_lb_combos)][var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="Yards", data=data)

fig.set_ylim(-10,20)
var = 'Position'

data = pd.concat([df_train_rusher.Yards, df_train_rusher[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="Yards", data=data)
possessionteam_map = {

    'BLT':'BAL',

    'CLV':'CLE',

    'ARZ':'ARI',

    'HST':'HOU'

}

df_train.PossessionTeam = df_train.PossessionTeam.apply(lambda pt:possessionteam_map[pt] if pt in possessionteam_map.keys() else pt)
df_train['TeamBelongAbbr'] = df_train.apply(

    lambda row:row['HomeTeamAbbr'] if row['Team']=='home' else row['VisitorTeamAbbr'],axis=1)



df_train['Offense'] = df_train.apply(lambda row:row['PossessionTeam']==row['TeamBelongAbbr'],axis=1)
df_aggregation = pd.DataFrame(columns={

    'GameId':[],'PlayId':[],'Teammate_dis':[],'Enemy_dis':[],'Teamate_enemy_dis':[],'Nearest_is_teammate':[],'Yards':[]})



for k,group in df_train.groupby(['GameId','PlayId']):

    rusher = group[group.NflId==group.NflIdRusher].iloc[0]

    offenses = group[group.NflId!=group.NflIdRusher][group.Offense]

    defenses = group[~group.Offense]

    def get_nearest(target, df):

        df['Tmp_dis'] = df[['X','Y']].apply(

            lambda xy:np.linalg.norm(np.array([xy.X,xy.Y])-np.array([rusher.X,rusher.Y])), 

            axis=1)

        return df.sort_values(by='Tmp_dis', ascending=False).iloc[0]

    nearest_offense = get_nearest(rusher, offenses)

    nearest_defense = get_nearest(rusher, defenses)

    Teamate_enemy_dis = np.linalg.norm(np.array([nearest_offense.X,nearest_offense.Y])-np.array([nearest_defense.X,nearest_defense.Y]))

    

    df_aggregation = df_aggregation.append(

        {'GameId':k[0],'PlayId':k[1],

         'Teammate_dis':nearest_offense.Tmp_dis,

         'Enemy_dis':nearest_defense.Tmp_dis,

         'Teamate_enemy_dis':Teamate_enemy_dis,

         'Nearest_is_teammate': 1 if nearest_offense.Tmp_dis < nearest_defense.Tmp_dis else 0,

        'Yards':rusher.Yards}, ignore_index=True)

    

df_aggregation.info()
plt.subplots(figsize=(20, 5))



plt.subplot(1,3,1)

var = 'Teammate_dis'

data = pd.concat([df_aggregation.Yards, df_aggregation[var]], axis=1)

plt.scatter(data[var],data['Yards'])



plt.subplot(1,3,2)

var = 'Enemy_dis'

data = pd.concat([df_aggregation.Yards, df_aggregation[var]], axis=1)

plt.scatter(data[var],data['Yards'])



plt.subplot(1,3,3)

var = 'Teamate_enemy_dis'

data = pd.concat([df_aggregation.Yards, df_aggregation[var]], axis=1)

plt.scatter(data[var],data['Yards'])
var = 'Nearest_is_teammate'

data = pd.concat([df_aggregation.Yards, df_aggregation[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="Yards", data=data)
def standardize_dataset(train):

    train['ToLeft'] = train.PlayDirection == "left"

    train['IsBallCarrier'] = train.NflId == train.NflIdRusher

    train['TeamOnOffense'] = "home"

    train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"

    train['IsOnOffense'] = train.Team == train.TeamOnOffense # Is player on offense?

    train['YardLine_std'] = 100 - train.YardLine

    train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

            'YardLine_std'

             ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  

              'YardLine']

    train['X_std'] = train.X

    train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 

    train['Y_std'] = train.Y

    train.loc[train.ToLeft, 'Y_std'] = 53.3 - train.loc[train.ToLeft, 'Y'] 

    train['Orientation_std'] = train.Orientation

    train.loc[train.ToLeft, 'Orientation_std'] = np.mod(180 + train.loc[train.ToLeft, 'Orientation_std'], 360)

    train['Dir_std'] = train.Dir

    train.loc[train.ToLeft, 'Dir_std'] = np.mod(180 + train.loc[train.ToLeft, 'Dir_std'], 360)

    train.loc[train['Season'] == 2017, 'Orientation'] = np.mod(90 + train.loc[train['Season'] == 2017, 'Orientation'], 360)    

    

    return train



df_train2 = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

dominance_df = standardize_dataset(df_train2)

dominance_df['Rusher'] = dominance_df['NflIdRusher'] == dominance_df['NflId']
def radius_calc(dist_to_ball):

    ''' I know this function is a bit awkward but there is not the exact formula in the paper,

    so I try to find something polynomial resembling

    Please consider this function as a parameter rather than fixed

    I'm sure experts in NFL could find a way better curve for this'''

    return 4 + 6 * (dist_to_ball >= 15) + (dist_to_ball ** 3) / 560 * (dist_to_ball < 15)



class Controller:

    '''This class is a wrapper for the two functions written above'''

    def __init__(self, play):

        self.play = play

        self.vec_influence = np.vectorize(self.compute_influence)

        self.vec_control = np.vectorize(self.pitch_control) 

        

    def compute_influence(self, x_point, y_point, player_id):

        '''Compute the influence of a certain player over a coordinate (x, y) of the pitch

        '''

        point = np.array([x_point, y_point])

        player_row = self.play.loc[player_id]

        theta = math.radians(player_row[56])

        speed = player_row[5]

        player_coords = player_row[54:56].values

        ball_coords = self.play[self.play['IsBallCarrier']].iloc[:, 54:56].values



        dist_to_ball = euclidean(player_coords, ball_coords)



        S_ratio = (speed / 13) ** 2         # we set max_speed to 13 m/s

        RADIUS = radius_calc(dist_to_ball)  # updated



        S_matrix = np.matrix([[RADIUS * (1 + S_ratio), 0], [0, RADIUS * (1 - S_ratio)]])

        R_matrix = np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        COV_matrix = np.dot(np.dot(np.dot(R_matrix, S_matrix), S_matrix), np.linalg.inv(R_matrix))



        norm_fact = (1 / 2 * np.pi) * (1 / np.sqrt(np.linalg.det(COV_matrix)))    

        mu_play = player_coords + speed * np.array([np.cos(theta), np.sin(theta)]) / 2



        intermed_scalar_player = np.dot(np.dot((player_coords - mu_play),

                                        np.linalg.inv(COV_matrix)),

                                 np.transpose((player_coords - mu_play)))

        player_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_player[0, 0])



        intermed_scalar_point = np.dot(np.dot((point - mu_play), 

                                        np.linalg.inv(COV_matrix)), 

                                 np.transpose((point - mu_play)))

        point_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_point[0, 0])



        return point_influence / player_influence

    

    

    def pitch_control(self, x_point, y_point):

        '''Compute the pitch control over a coordinate (x, y)'''



        offense_ids = self.play[self.play['IsOnOffense']].index

        offense_control = self.vec_influence(x_point, y_point, offense_ids)

        offense_score = np.sum(offense_control)



        defense_ids = self.play[~self.play['IsOnOffense']].index

        defense_control = self.vec_influence(x_point, y_point, defense_ids)

        defense_score = np.sum(defense_control)



        return expit(offense_score - defense_score)

    

    def display_control(self, grid_size=(50, 30), figsize=(12, 8)):

        front, behind = 30, 10

        left, right = 30, 30



        if self.play['IsOnOffense'].iloc[0]==True:

            colorm = ['purple'] * 11 + ['yellow'] * 11

        else:

            colorm = ['yellow'] * 11 + ['purple'] * 11

#         colorm = ['purple'] * 11 + ['yellow'] * 11

        colorm[np.where(self.play.Rusher.values)[0][0]] = 'black'

        player_coords = self.play[self.play['Rusher']][['X_std', 'Y_std']].values[0]



        X, Y = np.meshgrid(np.linspace(player_coords[0] - behind, 

                                       player_coords[0] + front, 

                                       grid_size[0]), 

                           np.linspace(player_coords[1] - left, 

                                       player_coords[1] + right, 

                                       grid_size[1]))



        # infl is an array of shape num_points with values in [0,1] accounting for the pitch control

        infl = self.vec_control(X, Y)



        plt.figure(figsize=figsize)

        plt.contourf(X, Y, infl, 12, cmap='bwr')

        plt.scatter(self.play['X_std'].values, self.play['Y_std'].values, c=colorm)

        plt.title('Yards gained = {}, play_id = {}'.format(self.play['Yards'].values[0], 

                                                           self.play['PlayId'].unique()[0]))

        plt.show()
_play_id1 = random.choice(dominance_df[~dominance_df.ToLeft].PlayId.tolist())

my_play = dominance_df[dominance_df.PlayId==_play_id1].copy()

control = Controller(my_play)

coords = my_play.iloc[1, 54:56].values         # let's compute the influence at the location of the first player

_pitch_control = control.vec_control(*coords)

print(_pitch_control)

control.display_control()
_play_id2 = random.choice(dominance_df[~dominance_df.ToLeft].PlayId.tolist())

my_play2 = dominance_df[dominance_df.PlayId==_play_id2].copy()

control2 = Controller(my_play2)

control2.display_control()
_controls = []

_yards = []

for _play_id in dominance_df.PlayId.unique().tolist()[:10000]:

    _my_play = dominance_df[dominance_df.PlayId==_play_id].copy()

    _control = Controller(_my_play)

    _rusher = _my_play.query('Rusher == True').iloc[0]

    coords = (_rusher.X_std,_rusher.Y_std)

    _pitch_control = _control.vec_control(*coords)

    _controls.append(_pitch_control)

    _yards.append(_rusher.Yards)

    

plt.scatter(_controls, _yards)