import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import math as m

from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")
all_Data = pd.read_csv('/kaggle/input/data.csv')
all_Data[all_Data['shot_made_flag'].isnull()].shape
all_Data.dtypes
#分解train,test 

train_Data = all_Data.loc[all_Data['shot_made_flag'].notnull(),]

test_Data = all_Data.loc[all_Data['shot_made_flag'].isnull(),]
#看下train中投进球的总体情况

sns.countplot(train_Data['shot_made_flag'])

"""看到1,0差距不大不用进行balance"""
print(train_Data['shot_made_flag'].value_counts())

print("命中率为:",11465*100/(14232+11465))

"""经过查询科比生涯命中率为44.7% 所以数据还是比较全面的"""
#action_type

train_Data['action_type'].isnull().sum()
action_Type = train_Data['action_type'].value_counts()

action_Type

"""科比最多的就是跳投，但是其中分类太多将其中部分进行合并"""
action_Type_More = action_Type[action_Type>100]

action_Type_Less = action_Type[action_Type<=100]
plt.figure(figsize=(20,8))

sns.barplot(x=action_Type_More.index, y = action_Type_More)
plt.figure(figsize=(14,5))

sns.barplot('action_type', 'shot_made_flag', data = train_Data)

plt.xticks(rotation = 'vertical')

plt.show()

"""投篮的方式对命中率有很大的影响"""
df = all_Data.copy()

df['action_type'] = df.action_type.apply(lambda x: x.replace('-', ''))

df['action_type'] = df.action_type.apply(lambda x: x.replace('Follow Up', 'followup'))

df['action_type'] = df.action_type.apply(lambda x: x.replace('Finger Roll','fingerroll'))

df['action_type'] = df.action_type.apply(lambda x: x.replace('Alley Oop','alleyoop'))



cv = CountVectorizer(max_features=50, stop_words=['shot'])



shot_features = cv.fit_transform(df['action_type']).toarray()

shot_features = pd.DataFrame(shot_features, columns=cv.get_feature_names())

"""根据每个关键字对投篮类型进行分类"""
df = pd.concat([df, shot_features], axis = 1)
#球场投射图

plt.figure(figsize=(12,12))

plt.subplot(121)

sns.scatterplot(x = df.loc[df['shot_made_flag']==1, 'loc_x'] ,y = df.loc[df['shot_made_flag']==1, 'loc_y'],

                alpha = 0.15,

                color = 'g')

plt.title('shots made distribution')

plt.ylim(-100,900)



plt.subplot(122)

sns.scatterplot(x = df.loc[df['shot_made_flag']==0, 'loc_x'] ,y = df.loc[df['shot_made_flag']==0, 'loc_y'],

                alpha = 0.08,

                color = 'r')

plt.title('shots missed distribution')

plt.ylim(-100,900)

"""投篮投失的区域与投进的差不多，也看的出来科比更喜欢正面，左右侧45度以及2边底线进攻"""
#minutes_remaining

sns.countplot(train_Data['minutes_remaining'])

"""从分布来看科比更喜欢在最后时刻进行进攻，明显的曼巴精神,总体比较均匀"""
#minutes_remaing的命中率

minutes_made = train_Data.groupby('minutes_remaining')['shot_made_flag'].mean()

plt.plot(minutes_made.index, minutes_made.values)

plt.scatter(minutes_made.index, minutes_made.values)

plt.ylabel('made rate')

plt.xlabel('minutes remain')

"""到最后命中率也不错，没有很大的区别，最后一分钟进球确实难度很高，反而当一节刚开始时命中率不是很高"""
#seconds_remaining的分布

plt.figure(figsize=(14,7))

sns.countplot(train_Data['seconds_remaining'])

"总体来看非常均匀，和minutes_remaining一样，最后几秒投的很多"
#seconds_minutes的命中率

seconds_made = train_Data.groupby('seconds_remaining')['shot_made_flag'].mean()

plt.plot(seconds_made.index, seconds_made.values)

plt.scatter(seconds_made.index, seconds_made.values)

plt.ylabel('made rate')

plt.xlabel('seconds remain')

"""基本和minutes_remaining情况差不多"""
#seconds和minutes合并

df['time_remaining'] = df['seconds_remaining'] + df['minutes_remaining'] * 60
print(df['time_remaining'].describe())

sns.distplot(df['time_remaining'])
#shot_distance

plt.figure(figsize=(10,8))

shot_distance = train_Data.groupby('shot_distance')['shot_made_flag'].mean()

sns.lineplot(shot_distance.index, shot_distance)

"""看得出来基本是越远命中率越低"""
#shot_distance distribution

plt.figure(figsize=(15,6))

sns.countplot(train_Data['shot_distance'])

"""除了0以内的扣篮上篮外，科比更喜欢远投"""
#shot_type vs distance

fig, axis = plt.subplots(2,1, figsize=(16,6))

sns.countplot(train_Data[train_Data['shot_type']=='2PT Field Goal']['shot_distance'], color = 'g', ax=axis[0])

sns.countplot(train_Data[train_Data['shot_type']=='3PT Field Goal']['shot_distance'], color = 'r', ax=axis[1])

"""看到23英尺左右应该就是3分线但是还有一些脏数据需要删除"""
#删除不合理数据

s = df[df['shot_made_flag'].notnull()]



df.drop(s[(s['shot_type']=='2PT Field Goal') & (s['shot_distance']>23)].index, inplace = True)

df.drop(s[(s['shot_type']=='3PT Field Goal') & (s['shot_distance'].isin([0,9]))].index, inplace = True)
#shot_distance vs time_remaining

train1 = train_Data.copy()

train1['time_remaining'] = train1['seconds_remaining'] + train1['minutes_remaining'] * 60

sns.scatterplot('time_remaining', 'shot_distance', hue = 'shot_made_flag', data = train1)

"""可以看到科比投篮的距离跟时间基本就是没有关系的，除了少部分数据以及等于0的时候"""
#shot_type

sns.countplot(train_Data['shot_type'])

"""还是2分球为主"""
#shot_zone_area

plt.figure(figsize=(13,5))

sns.barplot(train_Data['shot_zone_area'], train_Data['shot_made_flag'])

"""中路投篮还是命中率最高的"""
plt.figure(figsize=(13,5))

sns.countplot(train_Data['shot_zone_area'])

"""中路投篮投的也多，更喜欢右侧进攻"""
#shot_zone_basic shot rate

plt.figure(figsize=(13,5))

sns.barplot(train_Data['shot_zone_basic'], train_Data['shot_made_flag'])

"""很正常的禁区以及油漆区命中率最高"""
#shot_zone_basic

plt.figure(figsize=(13,5))

sns.countplot(train_Data['shot_zone_basic'])

"""和shot_zone_area结果差不多"""
#shot_zone_range rate

plt.figure(figsize=(13,5))

sns.barplot(train_Data['shot_zone_range'], train_Data['shot_made_flag'])
#shot_zone_range 

plt.figure(figsize=(13,5))

sns.countplot(train_Data['shot_zone_range'])
#定义一个颜色函数来

def color_generator(num_colors):

    colors = []

    for i in range(num_colors):

        colors.append((np.random.rand(), np.random.rand(), np.random.rand()))

    return colors
colors = color_generator(100)

def plot_zone_wise(zone_name):

    c_mean = train_Data.groupby(zone_name)['shot_made_flag'].mean()

    plt.figure(figsize=(10,10))

    for i, area in enumerate(train_Data[zone_name].unique()):

        plt.subplot(121)

        c = train_Data.loc[(train_Data[zone_name]==area)]

        plt.scatter(x = c['loc_x'], y = c['loc_y'], alpha = 0.6, color = colors[i])

        plt.text(c['loc_x'].mean(), c['loc_y'].quantile(0.8), '%0.3f'%(c_mean[area]), size = 13, 

                bbox = dict(facecolor='red', alpha=0.5))

        plt.ylim(-100, 900)

    plt.legend(train_Data[zone_name].unique())

    plt.title(zone_name)

    plt.show()

        
#直观图

plot_zone_wise('shot_zone_area')
#shot_zone_basic

plot_zone_wise('shot_zone_basic')
#shot_zone_range

plot_zone_wise('shot_zone_range')
#period

sns.countplot(train_Data['period'])

"""科比更喜欢第三节接管比赛"""
c = train_Data.groupby('period')['shot_made_flag'].mean()

plt.plot(c.index, c.values)

plt.scatter(c.index, c.values)

"""这个数据有点失望，看来科比其实最后一节命中率很低,第二个加时命中率比较高很可能是数据量太小的关系"""
#三分球两分球命中率 vs period

x1 = train_Data[train_Data['shot_type'] == '2PT Field Goal']

x2 = train_Data[train_Data['shot_type'] == '3PT Field Goal']

c1 = x1.groupby('period')['shot_made_flag'].mean()

c2 = x2.groupby('period')['shot_made_flag'].mean()

plt.plot(c1.index, c1.values,'g')

plt.scatter(c1.index, c1.values)

plt.plot(c2.index, c2.values,'r')

plt.scatter(c2.index, c2.values)

"""不管三分球还是两分球命中率都是越来越低，而且加时的时候三分球命中率变化幅度很大，应该是数据太少的关系"""
sns.countplot(train_Data['period'], hue = train_Data['shot_type'])

"""估计想追分命，三分球随节数越投越多，而2分球是第一节第三节明显更多"""
#区分主客场

team = []

for i in train_Data['matchup'].value_counts().index:

    if i.find('vs')!=-1:

        team.append(i)
df.loc[df['matchup'].isin(team)==True, 'matchup'] = 1

df.loc[df['matchup'] != 1, 'matchup'] = 0
#主客场区别

s = df[df['shot_made_flag'].notnull()]

sns.barplot(s['matchup'], s['shot_made_flag'])

"""主客场的命中率差不多，主场略高一点，但是应该关系不大"""
#lon, lat

sns.scatterplot(x = 'lat', y= 'lon', data = train_Data, alpha = 0.5, hue = 'shot_made_flag')

"""和loc_x,loc_y一样是描绘投篮区域的，直接删除"""
#play-off

c = train_Data.groupby('playoffs')['shot_made_flag'].mean()

sns.barplot(c.index, c.values)

"""是否为季后赛似乎也没有区别"""
#season

plt.figure(figsize=(10,5))

c = train_Data['season'].sort_values().values

sns.countplot(train_Data['season'].sort_values())

plt.xticks(rotation = 'vertical')

"""除了受伤以及新秀赛季科比的投篮次数还是比较平均的，其中05,06,07为巅峰"""
#season rate

c = train_Data.groupby('season')['shot_made_flag'].mean()

plt.figure(figsize=(10,5))

plt.plot(c.index, c.values)

plt.scatter(c.index, c.values)

plt.xticks(rotation = 'vertical')

"""命中率实际上没有很大的区别，中间赛季因为受伤以及花边新闻导致命中率下降，也说明season是一个比较重要的变量"""
#2pts and 3ts through time

plt.figure(figsize=(10,5))

sns.lineplot(train_Data['season'].sort_values(), train_Data['shot_made_flag'], hue = train_Data['shot_type'],

            markers=True)

plt.xticks(rotation = 'vertical')

"""两分球有稳步上升的趋势，而三分球命中率波动较大"""
#opponent

c = train_Data.groupby('opponent')['shot_made_flag'].mean()

c = c.sort_values()

plt.figure(figsize=(10,5))

plt.plot(c.index, c.values)

plt.scatter(c.index, c.values)

plt.xticks(rotation = 'vertical')

"""尼克斯，国王还有VAN三个最惨，篮网因为才换新的，刚好科比已经暮年，所以命中率最低"""
#gamedate

df['game_month'] = df['game_date'].apply(lambda x: x.split('-')[1])

df['game_day'] = df['game_date'].apply(lambda x: x.split('-')[2])

"""将game_date区分为月份和天数，看看是否会有区别"""
#game_day

c = df[df['shot_made_flag'].notnull()]

plt.figure(figsize=(10,5))

sns.countplot(c['game_day'])

"""game_day区别不大"""
#game_day rate

plt.figure(figsize=(10,5))

sns.lineplot(c['game_day'], c['shot_made_flag'])

"""居然区别还不小，但是总体在一条直线"""
#game_month

plt.figure(figsize=(10,5))

sns.countplot(c['game_month'])

"""因为有常规赛，季后赛和休赛期所以场次有区别"""
#game_month rate

plt.figure(figsize=(10,5))

sns.lineplot(c['game_month'], c['shot_made_flag'])

"""似乎game_month会有一定影响"""
c = df[df['shot_made_flag'].notnull()]

c = c[c['time_remaining']<60]

sns.lineplot(c['time_remaining'], c['shot_made_flag'])
df['angle'] = df.apply(lambda row: 90 if row['loc_y']==0 else m.degrees(m.atan(row['loc_x']/abs(row['loc_y']))),axis=1)
df['angle_bin'] = pd.cut(df.angle, 7, labels=range(7))

df['angle_bin'] = df.angle_bin.astype(int)
train_Data['angle'] = train_Data.apply(lambda row: 90 if row['loc_y']==0 else m.degrees(m.atan(row['loc_x']/abs(row['loc_y']))),axis=1)

train_Data['angle_bin'] = pd.cut(train_Data.angle, 7, labels=range(7))

train_Data['angle_bin'] = train_Data.angle_bin.astype(int)
plot_zone_wise('angle_bin')
df['distance_bin'] = pd.cut(df['shot_distance'], bins=10, labels=range(10))
#需要删除的列

drop = ['game_id', 'action_type', 'game_event_id', 'lat', 'lon', 'playoffs', 'team_id', 'team_name' ,

       'game_date', 'shot_id', 'shot_distance']

predictors = df.drop(drop, axis = 1)
#randomforest筛选变量

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

train_data = predictors[predictors['shot_made_flag'].notnull()]

test_data = predictors[predictors['shot_made_flag'].isnull()]
train_label = train_data['shot_made_flag']

test_label = test_data['shot_made_flag']

train_data = train_data.drop('shot_made_flag', axis = 1)

test_data = test_data.drop('shot_made_flag', axis = 1)
for column in train_data:

    if train_data[column].dtype == type(object):

        le = preprocessing.LabelEncoder()

        train_data[column] = le.fit_transform(train_data[column])



le = preprocessing.LabelEncoder()

train_data['distance_bin'] = le.fit_transform(train_data['distance_bin'])
for column in test_data:

    if test_data[column].dtype == type(object):

        le = preprocessing.LabelEncoder()

        test_data[column] = le.fit_transform(test_data[column])

le = preprocessing.LabelEncoder()

test_data['distance_bin'] = le.fit_transform(test_data['distance_bin'])
rf = RandomForestClassifier(oob_score=True)

param_grid = {'criterion': ['gini','entropy'],

               'max_depth':[4,6,8,10],

               'n_estimators':[50,100,200,300]

             }

tune_model = model_selection.GridSearchCV(rf, param_grid=param_grid, cv = 5, scoring='accuracy')

tune_model.fit(train_data, train_label)
tune_model.best_params_
rf1 = RandomForestClassifier(criterion='gini', max_depth=10, n_estimators=300)

rf1.fit(train_data,train_label)
d = rf1.predict_proba(test_data)[:,1]

sub['shot_made_flag'] = d

sub.to_csv('sub4.csv', index=False)

xg = XGBClassifier()

xg.fit(train_data, train_label)

dd = xg.predict_proba(test_data)[:, 1]

sub['shot_made_flag'] = dd

sub.to_csv('sub5.csv', index=False)
importance = rf1.feature_importances_

indices = np.argsort(importance)[::-1]

features = train_data.columns

for f in range(train_data.shape[1]):

    print(("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importance[indices[f]])))
predictor = ['driving', 'combined_shot_type', 'running', 'dunk', 'time_remaining', 'loc_y', 'angle' ,'bank',

            'turnaround', 'season', 'seconds_remaining', 'loc_x', 'pullup', 'game_day', 'shot_zone_basic', 'fadeaway',

            'opponent', 'distance_bin', 'minutes_remaining', 'shot_zone_range', 'game_month','layup','period',

            'slam', 'matchup','shot_zone_area']
train_data_selected = train_data.loc[:, predictor]

test_data_selected = test_data.loc[:, predictor]
from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.1)

xgb.fit(train_data_selected, train_label)
xgb = XGBClassifier(learning_rate=0.1)

param_test1 = {'max_depth' : range(5,11,2),

              'min_child_wate': range(1,6,2)}

tune_model1 = model_selection.GridSearchCV(xgb, param_grid=param_test1, cv = 5, scoring='roc_auc')

tune_model1.fit(train_data_selected, train_label)
tune_model1.best_params_
xgb1 = XGBClassifier(max_depth=5, min_child_weight=1)

param_test2 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

tune_mode2 = model_selection.GridSearchCV(xgb1, param_grid=param_test2, cv = 5, scoring='roc_auc')

tune_mode2.fit(train_data_selected, train_label)
tune_mode2.best_params_
xgb2 = XGBClassifier(max_depth=5, min_child_weight=1, gamma = 0.2)

param_test3 = {

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}

tune_mode3 = model_selection.GridSearchCV(xgb2, param_grid=param_test3, cv = 5, scoring='roc_auc')

tune_mode3.fit(train_data_selected, train_label)
tune_mode3.best_params_
xgb3 = XGBClassifier(max_depth=7, learning_rate=0.01, n_estimators=400,

                     min_child_weight=1, gamma = 0.2, colsample_bytree = 0.7, subsample = 0.6)

xgb3.fit(train_data_selected, train_label)
xgb3.predict_proba(test_data_selected)[:,1]