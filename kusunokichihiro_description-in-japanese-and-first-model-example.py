import pandas as pd



df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)



# PlayId, GameId 内で共通の値であるカラムを調べる

# 明らかに選手の特徴であるカラムは除く

feats_player = ['Team', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 

                'NflId', 'DisplayName', 'JerseyNumber', 'PlayerHeight', 'PlayerWeight', 

                'PlayerBirthDate', 'PlayerCollegeName', 'Position']



for id_ in ['PlayId', 'GameId']:

    df_ = df.drop(columns=feats_player)

    groups = df_.groupby(id_)

    s = groups.agg(lambda s: len(s.unique())).apply(max) # Id内でユニークな値の個数の最大値

    print('---------- ' + id_ + ' 内共通 ----------')

    for index, value in s.iteritems():

        if value == 1:

            print(index)
import numpy as np

df_play = df.drop_duplicates(subset='PlayId')

plays_down1 = df_play[df_play['Down'] == 1]

plays_down2 = df_play[df_play['Down'] == 2]

plays_down3 = df_play[df_play['Down'] == 3]

plays_down4 = df_play[df_play['Down'] == 4]

print(plays_down1.shape)

print(plays_down2.shape)

print(plays_down3.shape)

print(plays_down4.shape)



%matplotlib inline

import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 12, 4

rcParams['font.size'] = 16

plt.hist(plays_down1['Yards'], range=(-100, 100), bins=199, density=True, cumulative=True, alpha=0.3, histtype='stepfilled', color='r', label='down1')

plt.hist(plays_down2['Yards'], range=(-100, 100), bins=199, density=True, cumulative=True, alpha=0.3, histtype='stepfilled', color='b', label='down2')

plt.hist(plays_down3['Yards'], range=(-100, 100), bins=199, density=True, cumulative=True, alpha=0.3, histtype='stepfilled', color='g', label='down3')

plt.hist(plays_down4['Yards'], range=(-100, 100), bins=199, density=True, cumulative=True, alpha=0.3, histtype='stepfilled', color='y', label='down4')

plt.xlim(-20, 30)

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

plt.show()
print(plays_down3[plays_down3['Distance']==1].shape)

print(plays_down3[plays_down3['Distance']>1].shape)

print(plays_down4[plays_down4['Distance']==1].shape)

print(plays_down4[plays_down4['Distance']>1].shape)



plt.hist(plays_down3[plays_down3['Distance']==1]['Yards'], range=(-100, 100), bins=199, density=True, cumulative=True, alpha=0.3, histtype='stepfilled', color='r', label='down3_1')

plt.hist(plays_down3[plays_down3['Distance']>1]['Yards'], range=(-100, 100), bins=199, density=True, cumulative=True, alpha=0.3, histtype='stepfilled', color='b', label='down3_not1')

plt.xlim(-20, 30)

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

plt.show()



plt.hist(plays_down4[plays_down4['Distance']==1]['Yards'], range=(-100, 100), bins=199, density=True, cumulative=True, alpha=0.3, histtype='stepfilled', color='r', label='down4_1')

plt.hist(plays_down4[plays_down4['Distance']>1]['Yards'], range=(-100, 100), bins=199, density=True, cumulative=True, alpha=0.3, histtype='stepfilled', color='b', label='down4_not1')

plt.xlim(-20, 30)

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

plt.show()
df_play_2017 = df_play[df_play['Season'] == 2017]

df_play_2018 = df_play[df_play['Season'] == 2018]



cd = np.histogram(df_play_2017['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_play_2017)



yard_to_index = {}

for i, y in enumerate(range(-99, 100)):

    yard_to_index[y] = i



# 理論上ありえない確率を削っておく

# Ex. 自陣35なら、ゲインは 65 以上にならないし、-35 以下にならない

# Ex. 敵陣35なら、ゲインは 35 以上にならないし、-65 以下にならない

def postprocess(cd, target_df):

    yardline = target_df['YardLine'].iloc[0]

    own = (target_df['FieldPosition'].iloc[0] == target_df['PossessionTeam'].iloc[0])

    if own:

        cd[:yard_to_index[- yardline]] = 0.0

        cd[yard_to_index[100 - yardline]:] = 1.0

    else:

        cd[:yard_to_index[- 100 + yardline]] = 0.0

        cd[yard_to_index[yardline]:] = 1.0



##### モデル0： 訓練データの経験分布をそのままあてるだけの予測 #####

def make_my_predictions_0(target_df, prediction_df):

    cd_ = np.copy(cd)

    postprocess(cd_, target_df)

    prediction_df.iloc[0,:] = cd_



df_ = df_play_2017[df_play_2017['Down'] == 1]

cd1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play_2017[df_play_2017['Down'] == 2]

cd2 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play_2017[(df_play_2017['Down'] == 3) & (df_play_2017['Distance'] == 1)]

cd3_1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play_2017[(df_play_2017['Down'] == 3) & (df_play_2017['Distance'] != 1)]

cd3_not1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play_2017[(df_play_2017['Down'] == 4) & (df_play_2017['Distance'] == 1)]

cd4_1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play_2017[(df_play_2017['Down'] == 4) & (df_play_2017['Distance'] != 1)]

cd4_not1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)



##### モデル1： 訓練データの経験分布をダウン別、残りヤード別にあてる予測 #####

def make_my_predictions_1(target_df, prediction_df):

    if target_df['Down'].iloc[0] == 1:

        cd_ = np.copy(cd1)

    elif target_df['Down'].iloc[0] == 2:

        cd_ = np.copy(cd2)

    elif (target_df['Down'].iloc[0] == 3) and (target_df['Distance'].iloc[0] == 1):

        cd_ = np.copy(cd3_1)

    elif target_df['Down'].iloc[0] == 3:

        cd_ = np.copy(cd3_not1)

    elif (target_df['Down'].iloc[0] == 4) and (target_df['Distance'].iloc[0] == 1):

        cd_ = np.copy(cd4_1)

    else:

        cd_ = np.copy(cd4_not1)

    postprocess(cd_, target_df)

    prediction_df.iloc[0,:] = cd_
from IPython.core.display import display, HTML



# 予測結果のひな型

prediction_df = pd.DataFrame()

for y in range(-99, 100):

    prediction_df['Yards' + str(y)] = [0.0]



loss = []



for i in range(len(df_play_2018)):

    target_df_ = df_play_2018.iloc[[i],:]

    prediction_df_ = prediction_df.copy()

    make_my_predictions_0(target_df_, prediction_df_) # モデル0

    # display(HTML(prediction_df_.to_html(index=False))) # 表示してみる

    # break

    

    actual_ = np.array([0.0] * 199)

    actual_[yard_to_index[target_df_['Yards'].iloc[0]]:] = 1.0

    loss_ = (prediction_df_.iloc[0,:] - actual_) * (prediction_df_.iloc[0,:] - actual_)

    loss.append(loss_.mean())



print(np.array(loss).mean())



loss = []



for i in range(len(df_play_2018)):

    target_df_ = df_play_2018.iloc[[i],:]

    prediction_df_ = prediction_df.copy()

    make_my_predictions_1(target_df_, prediction_df_) # モデル1

    # display(HTML(prediction_df_.to_html(index=False))) # 表示してみる

    # break

    

    actual_ = np.array([0.0] * 199)

    actual_[yard_to_index[target_df_['Yards'].iloc[0]]:] = 1.0

    loss_ = (prediction_df_.iloc[0,:] - actual_) * (prediction_df_.iloc[0,:] - actual_)

    loss.append(loss_.mean())



print(np.array(loss).mean())
### 1. データを読み込む。



# import pandas as pd

# import numpy as np

# df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

# df_play = df.drop_duplicates(subset='PlayId')



### 2. 第1ダウン、第2ダウン、第3ダウン（残り1ヤードか否か）、第4ダウン（残り1ヤードか否か）の別に分布をつくっておく。



df_ = df_play[df_play['Down'] == 1]

cd1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play[df_play['Down'] == 2]

cd2 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play[(df_play['Down'] == 3) & (df_play['Distance'] == 1)]

cd3_1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play[(df_play['Down'] == 3) & (df_play['Distance'] != 1)]

cd3_not1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play[(df_play['Down'] == 4) & (df_play['Distance'] == 1)]

cd4_1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)

df_ = df_play[(df_play['Down'] == 4) & (df_play['Distance'] != 1)]

cd4_not1 = np.histogram(df_['Yards'], bins=range(-100, 100, 1))[0].cumsum() / len(df_)



### 3. ダウンと残りヤードだけみてそのまま上の分布をあてる関数を実装する（postprocess はありえない確率の削除； 上の方のセル参照）。 



def make_my_predictions_1(target_df, prediction_df):

    if target_df['Down'].iloc[0] == 1:

        cd_ = np.copy(cd1)

    elif target_df['Down'].iloc[0] == 2:

        cd_ = np.copy(cd2)

    elif (target_df['Down'].iloc[0] == 3) and (target_df['Distance'].iloc[0] == 1):

        cd_ = np.copy(cd3_1)

    elif target_df['Down'].iloc[0] == 3:

        cd_ = np.copy(cd3_not1)

    elif (target_df['Down'].iloc[0] == 4) and (target_df['Distance'].iloc[0] == 1):

        cd_ = np.copy(cd4_1)

    else:

        cd_ = np.copy(cd4_not1)

    postprocess(cd_, target_df) # See above

    prediction_df.iloc[0,:] = cd_



### 4. テストデータに適用する。

    

from kaggle.competitions import nflrush

env = nflrush.make_env()



for (test_df, prediction_df) in env.iter_test():

    make_my_predictions_1(test_df, prediction_df)

    env.predict(prediction_df)



env.write_submission_file()