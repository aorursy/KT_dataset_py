# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# グラフ表示

import matplotlib.pyplot as plt

import seaborn as sns

# 動画表示

import matplotlib.animation as animation

from IPython.display import Image



# matplotlib日本語表示 だめ

# font = {"family": "TakaoGothic"}

# plt.rc('font', **font)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
# データ取り込み

df = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')



df.head(3)
# 不要な列を削除

df = df.drop(['Block', 'IUCR', 'Description', 'Location Description', 'Domestic', 'Beat', 'District', 'Ward', 'FBI Code', 'Updated On', 'Latitude', 'Longitude', 'Location'], axis=1)

df.head(3)
# 空白を含まない列名に変更

df = df.rename(columns={'Primary Type': 'Type','X Coordinate': 'Coord_X', 'Y Coordinate': 'Coord_Y'})
# データクレンジング

# NaNデータ削除

df = df.dropna(axis = 0)

# 2012～2016年のデータのみ抽出（最後に2017/1月のデータが少しあり）

df = df[df['Year'] < 2017]

# 座標の無いデータを削除

df = df[df['Coord_X'] > 0]
# 24時間列

df['Hour'] =  [x[11:13] for x in df['Date']]

df['AMPM'] =  [x[20:] for x in df['Date']]

df = df.reset_index(drop=True) #これをしないとindex out of boundsエラーになる

idx_h = df.columns.get_loc('Hour')

# 'Hour'の値を24時間制数値に変換

for i, row in df.iterrows():

    h = int(row['Hour'])

    # PM12時は12時なので12のまま

    if row['AMPM'] == 'PM' and h < 12:

        h += 12

    # AM12時は午前0時

    elif row['AMPM'] == 'AM' and h == 12:

        h = 0

    df.iat[i, idx_h] = h
# 犯罪タイプ

types = df['Type'].unique()

dic_type = {

    'BATTERY' : '殴打',

    'PUBLIC PEACE VIOLATION' : '迷惑行為',

    'THEFT' : '盗難',

    'WEAPONS VIOLATION' : '武器の侵害',

    'ROBBERY' : '強盗',

    'MOTOR VEHICLE THEFT' : '車の盗難',

    'ASSAULT' : '暴行',

    'OTHER OFFENSE' : 'その他の違反',

    'DECEPTIVE PRACTICE' : '巧妙な練習',

    'CRIMINAL DAMAGE' : '犯罪による損害',

    'CRIMINAL TRESPASS' : '刑事裁判',

    'BURGLARY' : '強盗',

    'STALKING' : 'ストーキング',

    'CRIM SEXUAL ASSAULT' : '性的暴行',

    'NARCOTICS' : '麻薬',

    'SEX OFFENSE' : '性犯罪',

    'HOMICIDE' : '殺人',

    'OFFENSE INVOLVING CHILDREN' : '子供を含む犯罪',

    'INTERFERENCE WITH PUBLIC OFFICER' : '公務員との干渉',

    'PROSTITUTION' : '売春',

    'KIDNAPPING' : '誘拐',

    'GAMBLING' : 'ギャンブル',

    'INTIMIDATION' : '入院',

    'ARSON' : '放火',

    'LIQUOR LAW VIOLATION' : '飲酒法違反',

    'OBSCENITY' : '不信',

    'NON-CRIMINAL' : '非犯罪者',

    'PUBLIC INDECENCY' : '公的機関',

    'HUMAN TRAFFICKING' : '人身売買',

    'CONCEALED CARRY LICENSE VIOLATION' : '連続キャリーライセンス違反',

    'NON - CRIMINAL' : 'ノンクリミナル',

    'OTHER NARCOTIC VIOLATION' : '他の麻酔違反',

    'NON-CRIMINAL (SUBJECT SPECIFIED)' : 'ノンクリミナル（対象者指定）'}



for t in types:

    print(t + " : " + dic_type[t])
## ヒートマップ

# 座標範囲の取得

x_min = df['Coord_X'].min()

x_max = df['Coord_X'].max()

width = x_max - x_min

y_min = df['Coord_Y'].min()

y_max = df['Coord_Y'].max()

height = y_max - y_min



# X軸をdiv分割した区分したi番目のY軸方向の区切り列を返す

def get_segment_y(arg_df, i, div, y_seg, lbl): # i=0～

    x_step = width/div

    temp = arg_df[arg_df['Coord_X'] >= x_min + i*x_step]

    temp = temp[temp['Coord_X'] < x_min + (i+1)*x_step]

    return pd.cut(temp['Coord_Y'], y_seg, labels=lbl).value_counts()



# ヒートマップ用データDFの作成

def make_heatmap(crime, hour):

    df_map = pd.DataFrame() #ヒートマップ用DF

    div = 50

    lbl = ["{0}".format(i) for i in range(0, div, 1)]

    y_seg = [v for v in range(int(y_min), int(y_max), int(height/div))]

    

    # 時刻指定無し

    if hour is None:

        df_tmp = df[df['Type'] == crime]

    else:

        df_tmp = df[(df['Type'] == crime) & (df['Hour'] == hour)]



    for i in range(0, div):

        df_map[i] = get_segment_y(df_tmp, i, div, y_seg, lbl)

    df_map = df_map.sort_index(ascending=True).fillna(0) # seabornを使うときはFalseにする

    df_map = np.sqrt(df_map)

    return df_map

    

# ヒートマップ作成

def heatmap(crime):

    if crime != '':

        # 時間アニメーション表示

        if True:

            fig, ax = plt.subplots()

            ims = []

            # 24時間ループ

            for h in range(24):

                title = crime + ' : ' + str(h) + 'H'

                df_map = make_heatmap(crime, h)

                title = ax.text(0.5, 1.01, title,

                         ha='center', va='bottom',

                         transform=ax.transAxes, fontsize='large')

                #im = sns.heatmap(df_map)

                im = plt.pcolor(df_map, cmap=plt.cm.hot)

                ims.append([im]+[title])

            anim = animation.ArtistAnimation(fig, ims)

            anim.save(crime + '.gif', writer='imagemagick', fps=1)

        # 全時間合計

        else:

            df_map = make_heatmap(crime, None)

            plt.title(crime)

            plt.pcolor(df_map, cmap=plt.cm.hot)

            #sns.heatmap(df_map)

    else:

        # 犯罪名が指定されていない場合は全てのグラフをアニメーション表示

        fig, ax = plt.subplots()

        ims = []

        for t in types:

            title = t + ' : ' + dic_type[t]

            print(title)

            df_map = make_heatmap(t, None)

            title = ax.text(0.5, 1.01, t,

                     ha='center', va='bottom',

                     transform=ax.transAxes, fontsize='large')

            #im = sns.heatmap(df_map)

            im = plt.pcolor(df_map, cmap=plt.cm.hot)

            ims.append([im]+[title])

        anim = animation.ArtistAnimation(fig, ims)

        anim.save('all_crimes.gif', writer='imagemagick', fps=1)

# 全ての発生地点

plt.scatter(df['Coord_X'], df['Coord_Y'], s=1)
# BATTERY 暴行

heatmap('BATTERY')

Image("../working/BATTERY.gif")
# THEFT 窃盗

heatmap('THEFT')

Image("../working/THEFT.gif")
# BURGLARY 押し込み強盗

heatmap('BURGLARY')

Image("../working/BURGLARY.gif")
# ROBBERY 強盗

heatmap('ROBBERY')

Image("../working/ROBBERY.gif")
# SEX OFFENSE 性犯罪

heatmap('SEX OFFENSE')

Image("../working/SEX OFFENSE.gif")
# CRIMINAL DAMAGE 器物損

heatmap('CRIMINAL DAMAGE')

Image("../working/CRIMINAL DAMAGE.gif")
# WEAPONS VIOLATION 銃刀法違反

heatmap('WEAPONS VIOLATION')

Image("../working/BURGLARY.gif")
# 全ての罪種をアニメーション表示

heatmap('')

Image("../working/all_crimes.gif")