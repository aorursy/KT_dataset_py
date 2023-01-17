import tensorflow as tf

import keras

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_color_codes("muted")

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import random
RANDOM_SEED_VAL = 5

random.seed(RANDOM_SEED_VAL)

np.random.seed(RANDOM_SEED_VAL)

tf.set_random_seed(RANDOM_SEED_VAL)
data = pd.read_csv('/kaggle/input/kobe-bryant-shot-selection/data.csv')



train_data = data.loc[data.shot_made_flag.isnull() == False].copy()

test_data = data.loc[data.shot_made_flag.isnull() == True].copy()
print('Total Data Shape: {}'.format(data.shape))

print('Train Data Shape: {}'.format(train_data.shape))

print('Test Data Shape: {}'.format(test_data.shape))
# No missing value in dataset!



for colname in data.columns:

    print('Missing value count of {} : {}'.format(colname, data[colname].isnull().sum()))
train_data.loc[:10,['action_type', 'combined_shot_type']]
# Jump Shot, Layup, Dunk 순으로 많이 시도하였다

display(pd.crosstab(train_data.combined_shot_type,train_data.shot_made_flag, margins=True))



sns.countplot('combined_shot_type', hue='shot_made_flag', data=train_data)

plt.show()
# Dunk 의 경우 시도 대비 샷 성공률이 92%로 가장 높았다.

# shot type 에 따라 성공률 값이 다양하다 (i.e. shot type 으로 성공 유무를 설명 가능할 것 같다)



display(pd.crosstab(train_data.combined_shot_type,train_data.shot_made_flag).apply(lambda r: r/r.sum(), axis=1))



temp = train_data[['combined_shot_type','shot_made_flag']].groupby('combined_shot_type').mean()

temp = temp.sort_values(by=('shot_made_flag'),ascending=False)    

sns.barplot(y=temp.index.values, x='shot_made_flag', data=temp, color="b")

plt.title('Success rates by combined_shot_type')

plt.show()
# 각 combined_shot_type 내의 카테고리 별로 성공률 분포가 궁금해서 그려보았다.

# 같은 combined_shot_type 내에서도 action_type 별로 성공률 분포가 다양하다. (유용한 피쳐일 수 있다)



for ctype in ['Dunk', 'Bank Shot', 'Layup', 'Hook Shot', 'Jump Shot', 'Tip Shot']:

    df = train_data[train_data['combined_shot_type'] == ctype][['action_type','shot_made_flag']].groupby('action_type').mean()

    df = df.sort_values(by=('shot_made_flag'),ascending=False)    

    

    sns.barplot(y=df.index.values, x='shot_made_flag', data=df, color="b")

    plt.title('{} - Success rates by action_type'.format(ctype))

    plt.xlim(right=1.0)

    plt.xlabel('Success rate')

    plt.show()
# 전체 action_type 별 성공률



temp = train_data[['action_type','shot_made_flag']].groupby('action_type').mean()

temp = temp.sort_values(by=('shot_made_flag'),ascending=False)    

    

sns.barplot(y=temp.index.values, x='shot_made_flag', data=temp, color="b")

plt.title('Success rates by action_type')

plt.xlim(right=1.0)

plt.xlabel('Success rate')

plt.gcf().set_size_inches(20,10)

plt.show()
# shot_type 이라는 애들은 무엇인가?



train_data.shot_type.value_counts()
# shot_type = '3PT Field Goal' 인 경우 전부 combined_shot_type = 'Jump Shot' 이다.

# 단, combined_shot_type = 'Jump Shot' 라고 해서 모두 shot_type = '3PT Field Goal' 인 것은 아니다.



pd.crosstab(train_data.shot_type, train_data.combined_shot_type)
# 쉬운 샷일수록 ('2PT') 성공 황률이 더 높다

f, ax = plt.subplots(1, 2, figsize=(10, 5))



sns.barplot('shot_type', y='shot_made_flag', data=train_data, estimator=np.mean, ax=ax[0])

ax[0].set_title('Success rates by shot_type')

ax[0].set_ylabel('Success rate')



sns.barplot('shot_type', y='shot_made_flag', data=train_data.loc[train_data.combined_shot_type=='Jump Shot'], estimator=np.mean, ax=ax[1])

ax[1].set_title('Success rates by shot_type within Jump Shot')

ax[1].set_ylim(top=0.5)

ax[1].set_ylabel('Success rate')



plt.show()
# game_date 가 string 값이므로, datetime 으로 변환한다.

# 단 나중을 위해 원래 string 값은 game_date_str 라는 변수에 새로 할당한다.



train_data['game_date_str'] = train_data['game_date'].copy()

train_data['game_date'] = pd.to_datetime(train_data['game_date'], format='%Y-%m-%d', exact=True)

train_data['game_date_tmp'] = train_data['game_date'].values.astype(np.int64)

train_data['game_date_year'] = train_data['game_date_str'].str.slice(0,4)

train_data['game_date_year'] = pd.to_numeric(train_data['game_date_year'])
# 하나의 날짜에 하나의 게임만을 뛰었다.

train_data[['game_date', 'game_id']].groupby('game_date')['game_id'].nunique().describe()
# 한 게임은 하나의 날짜에만 진행되었다.

# game_date 와 game_id 는 1대1 관계이다. 둘 중 하나만 모델에 넣어도 될 것 같다.



train_data[['game_date', 'game_id']].groupby('game_id').nunique()['game_date'].describe()
# game_date, game_id, game_event_id 는 shot_made_flag 와 상관계수가 높지 않다 - 비례 관계 같은건 없을것 같다.

# game_id 는 경기 일자순이 아닌 것 같다.



train_data[['shot_made_flag', 'game_date_tmp', 'game_id', 'game_event_id']].corr()
# 아래를 타이핑 한 뒤에야 season 이 무슨 변수인지 파악했다 (시작년도4자리-끝년도2자리).



season_mapper = {

    '1996-97':0,

    '1997-98':1,

    '1998-99':2,

    '1999-00':3,

    '2000-01':4,

    '2001-02':5,

    '2002-03':6,

    '2003-04':7,

    '2004-05':8,

    '2005-06':9,

    '2006-07':10,

    '2007-08':11,

    '2008-09':12,

    '2009-10':13,

    '2010-11':14,

    '2011-12':15,

    '2012-13':16,

    '2013-14':17,

    '2014-15':18,

    '2015-16':19

}



train_data['season_scale'] = train_data['season'].replace(season_mapper)

# game_date_year 나 season_scale 이나 거의 똑같은 정보다.

# shot_made_flag 와의 상관계수도 낮다.



train_data[['shot_made_flag', 'season_scale', 'game_date_year']].corr()
# season 및 year 별 성공률: 마지막 seasons / years 에 성공률이 비교적 저조하다.



f, ax = plt.subplots(2, 1, figsize=(20,10))



temp = train_data[['season', 'shot_made_flag']].groupby('season').mean()

temp = temp.sort_values(by=('season'),ascending=True)

sns.barplot(x=temp.index.values, y='shot_made_flag', data=temp, color="b", order=list(season_mapper.keys()), ax=ax[0])

ax[0].set_title('Success rates across Seasons')

ax[0].set_ylabel('Success rate')



temp = train_data[['game_date_year', 'shot_made_flag']].groupby('game_date_year').mean()

temp = temp.sort_values(by=('game_date_year'),ascending=True)

sns.barplot(x=temp.index.values, y='shot_made_flag', data=temp, color="b", ax=ax[1])

ax[1].set_title('Success rates across Years')

ax[1].set_ylabel('Success rate')



plt.show()
# LA 의 경위도가 (34.052235, -118.243683) 라는데, 이 기준으로 기록된 데이터일지도 모르겠다.



f, ax = plt.subplots(1, 2, figsize=(15,5))



sns.distplot(train_data.lat, ax=ax[0])

ax[0].set_title('lat distribution')



sns.distplot(train_data.lon, ax=ax[1])

ax[1].set_title('lon distribution')



plt.show()
train_data.lat.describe()
# (lon, lat) 과 shot_made_flag 의 산포도를 보면 패턴이 보이지 않는다.

# 데이터가 주로 반원 안에, 그것도 상단 중심쪽에 몰린걸 보면 경기장의 위치가 아니라 코트 안에서의 슈팅 시도 위치를 측정한 정보인 것 같다.



f, ax = plt.subplots(1, 1, figsize=(25,20))

sns.scatterplot(x="lon", y="lat", data=train_data.loc[train_data.lat > 33.75], hue='shot_made_flag', ax=ax)

plt.show()
train_data[['loc_x', 'loc_y', 'shot_zone_area', 'shot_distance', 'shot_zone_range', 'shot_zone_basic']].head()
shot_zone_range_mapper = { 'Less Than 8 ft.':0, '8-16 ft.': 1, '16-24 ft.': 2, '24+ ft.': 3, 'Back Court Shot':4 }

train_data['shot_zone_range_scale'] = train_data['shot_zone_range'].replace(shot_zone_range_mapper)
train_data[['shot_zone_range_scale', 'shot_zone_range']].head()
# loc_y 가 290 ~ 300 넘어가면 성공률이 현저하게 줄어든다. 그 미만 거리에선 특별한 패턴이 눈에 띄진 않는다.

# loc_x 기준으로 좌우 간의 성공률 차이는 별로 없는 듯 하다.



f, ax = plt.subplots(1, 2, figsize=(20,10))

sns.scatterplot(x="loc_x", y="loc_y", data=train_data, hue='shot_made_flag', ax=ax[0])

sns.scatterplot(x="loc_x", y="loc_y", data=train_data.loc[train_data.loc_y < 300], hue='shot_made_flag', ax=ax[1])

plt.show()
# shot_zone_area, shot_zone_range, shot_zone_basic 모두 코트 내의 위치를 다른 모양으로 구분한다.

# (loc_x, loc_y) 를 피쳐로 사용하는 것 보다 이들 조합을 사용하는 것이 더 의미있을지도 모른다.



f, ax = plt.subplots(1, 3, figsize=(15,10))

sns.scatterplot(x="loc_x", y="loc_y", data=train_data, hue='shot_zone_area', ax=ax[0])

ax[0].set_title('shot_zone_area across location')

sns.scatterplot(x="loc_x", y="loc_y", data=train_data, hue='shot_zone_range', ax=ax[1])

ax[1].set_title('shot_zone_range across location')

sns.scatterplot(x="loc_x", y="loc_y", data=train_data, hue='shot_zone_basic', ax=ax[2])

ax[2].set_title('shot_zone_basic across location')

plt.show()
# center 가 슈팅 시도도 제일 많고 성공 확률도 제일 높다.

# right, right-center 가 left, left-center 보다 슈팅 시도 및 득점 횟수가 조금 더 높은것 처럼 보인다.

# Back Court 는 꼴지.



display(pd.crosstab(train_data.shot_zone_area, train_data.shot_made_flag))



f, ax = plt.subplots(2, 1, figsize=(15,10))



sns.countplot('shot_zone_area', hue='shot_made_flag', data=train_data, ax=ax[0])



temp = train_data[['shot_zone_area', 'shot_made_flag']].groupby('shot_zone_area').mean().sort_values(by=('shot_made_flag'), ascending=False)

sns.barplot(temp.index.values, y='shot_made_flag', data=temp, ax=ax[1], color='b')

ax[1].set_ylabel('Success rate')



plt.show()
# test_data 에서의 Back Courte(BC) 빈번률 또한 매우 낮다. 

# 이를 모두 실패했다고 예측해버려도 무방할 것 같다. (BC 를 제외한 나머지 값들에 모델이 최적화되도록 트레이닝 해보자)



print(test_data.shape)

print(test_data.loc[test_data.shot_zone_area == 'Back Court(BC)'].shape)
# 전반적으로 거리가 멀어질수록 성공률은 내려간다.



display(pd.crosstab(train_data.shot_zone_range, train_data.shot_made_flag))



f, ax = plt.subplots(2, 1, figsize=(15,10))



sns.countplot('shot_zone_range', hue='shot_made_flag', data=train_data, ax=ax[0], order=['Less Than 8 ft.', '8-16 ft.', '16-24 ft.', '24+ ft.', 'Back Court Shot'])



temp = train_data[['shot_zone_range', 'shot_made_flag']].groupby('shot_zone_range').mean().sort_values(by=('shot_made_flag'), ascending=False)

sns.barplot(temp.index.values, y='shot_made_flag', data=temp, ax=ax[1], color='b')

ax[1].set_ylabel('Success rate')



plt.show()
# Restricted Area 가 유독 성공률이 높다. shot_distance 가 낮을수록 성공률이 높으니 당연할지도.



display(pd.crosstab(train_data.shot_zone_basic, train_data.shot_made_flag))



f, ax = plt.subplots(2, 1, figsize=(15,10))



sns.countplot('shot_zone_basic', hue='shot_made_flag', data=train_data, ax=ax[0])



temp = train_data[['shot_zone_basic', 'shot_made_flag']].groupby('shot_zone_basic').mean().sort_values(by=('shot_made_flag'), ascending=False)

sns.barplot(temp.index.values, y='shot_made_flag', data=temp, ax=ax[1], color='b')

ax[1].set_ylabel('Success rate')



plt.show()
train_data[['team_id', 'team_name', 'matchup']].head()
# 모든 인스턴스의 team_id, team_name 이 동일하다 (LAL)



print('-- train data teams--')

print(train_data.team_id.value_counts())

print(train_data.team_name.value_counts())

print('-- test data teams--')

print(test_data.team_id.value_counts())

print(test_data.team_name.value_counts())
# matchup 값의 포맷이 두 가지가 있다. 맨 마지막 3글자만 추출하면 상대팀 이니셜이 된다.



train_data['matchup'] = train_data['matchup'].str.slice(-3)
train_data.matchup.value_counts()
f, ax = plt.subplots(1, 1, figsize=(20,5))

ax_avg = ax.twinx()

ax_avg.set_ylim(bottom=0)

ax_avg.set_ylabel('success rate')

sns.countplot('matchup', hue='shot_made_flag', data=train_data, ax=ax)

ax_avg = plt.plot(train_data[['shot_made_flag', 'matchup']].groupby('matchup').mean(), color='red')

plt.show()
# opponent 가 matchup 마지막 3글자랑 완전히 똑같은 정보인가?

# 아래를 보면 완전히 똑같은 정보는 아니다. 그럼 이건 오타 내지는 다른 이름인건가?



temp = train_data[train_data.matchup != train_data.opponent][['matchup','opponent']]

pd.crosstab(temp.matchup, temp.opponent)
# opponent 와 matchup 은 동일한 정보라 할 수 있다.



temp = train_data[train_data.matchup=='CHH']['opponent'].value_counts()

print('-- matchup CHH opponent values --')

print(temp)

print()



temp = train_data[train_data.matchup=='NOK']['opponent'].value_counts()

print('-- matchup NOK opponent values --')

print(temp)

print()



temp = train_data[train_data.matchup=='PHO']['opponent'].value_counts()

print('-- matchup PHO opponent values --')

print(temp)

print()



temp = train_data[train_data.matchup=='SAN']['opponent'].value_counts()

print('-- matchup SAN opponent values --')

print(temp)

print()



temp = train_data[train_data.matchup=='UTH']['opponent'].value_counts()

print('-- matchup UTH opponent values --')

print(temp)
pd.crosstab(train_data.playoffs, train_data.shot_made_flag, margins=True)
pd.crosstab(train_data.playoffs,train_data.shot_made_flag).apply(lambda r: r/r.sum(), axis=1)
f, ax = plt.subplots(1, 2, figsize=(20,5))



sns.countplot('playoffs', hue='shot_made_flag', data=train_data, ax=ax[0])

ax[0].set_title('counts')



sns.barplot('playoffs', y='shot_made_flag', data=train_data, estimator=np.mean, ax=ax[1])

ax[1].set_title('success rates')



plt.show()
train_data[['minutes_remaining', 'seconds_remaining', 'period']].head()
train_data['seconds_remaining_total'] = train_data['minutes_remaining'] * 60 + train_data['seconds_remaining']
# minutes_remaining 이나 seconds_remaining 에서 이상값은 보이지 않는다. 0 값이 특히 많은 것 같다.



f, ax = plt.subplots(2, 1, figsize=(20, 10))

sns.countplot('minutes_remaining', data=train_data, ax=ax[0], color='b')

sns.countplot('seconds_remaining', data=train_data, ax=ax[1], color='b')

plt.show()
f, ax = plt.subplots(1, 1, figsize=(13, 5))

sns.distplot(train_data.seconds_remaining_total, ax=ax)

plt.show()
# period 가 증가할수록 seconds_remaining_total 가 감소하는 추세로 보이지만, 항상 떨어지는것은 아니다. (즉, 둘은 동일한 정보가 아니다)



sns.boxplot( x=train_data.period, y=train_data.seconds_remaining_total )

plt.show()
# seconds_remaining 또는 minutes_remaining 값이 0 일때 성공률이 제일 낮긴 하지만, 대체적으로 uniform 하다.



f, ax = plt.subplots(2, 1, figsize=(20, 15))



ax0_avg = ax[0].twinx()

ax0_avg.set_ylim(bottom=0)

ax0_avg.set_ylabel('success rate')

sns.countplot(hue='shot_made_flag', x='minutes_remaining', ax=ax[0], data=train_data)

ax0_avg = plt.plot(train_data[['shot_made_flag', 'minutes_remaining']].groupby('minutes_remaining').mean(), color='red')



ax1_avg = ax[1].twinx()

ax1_avg.set_ylim(bottom=0)

ax1_avg.set_ylabel('success rate')

sns.countplot(hue='shot_made_flag', x='seconds_remaining', ax=ax[1], data=train_data)

ax1_avg = plt.plot(train_data[['shot_made_flag', 'seconds_remaining']].groupby('seconds_remaining').mean(), color='red')



plt.show()
# period 에 따라서 성공률은 크게 차이나지 않는다.



train_data[['shot_made_flag', 'period']].groupby('period').mean()
# seconds_remaining_total 과 성공률은 상관계수가 높은 편은 아니다. 

# "경기 중 남은 시간"은 성공률을 예측하는데 그닥 중요한 피쳐가 아닐수도 있겠다.



train_data[['shot_made_flag', 'seconds_remaining_total', 'period', 'minutes_remaining', 'seconds_remaining']].corr()