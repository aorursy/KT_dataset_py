# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
from collections import Counter
import missingno as msno

import warnings
warnings.filterwarnings('ignore')
import plotly
sns.set_style('darkgrid')
df=pd.read_csv('../input/fifa19/data.csv')
df.columns
#とりあえずいらなそうなカラムを消す
df = df.drop(['Unnamed: 0','Photo', 'Flag', 'Club Logo'], axis=1, inplace=False)
#欠損値の数を確認
#順に並べるのはsortとsort_values
df.isnull().sum()[df.isnull().sum()>0].sort_values()
df.isnull().sum()[df.isnull().sum()==48].sort_values()
#48個のデータが欠損しているカラムが多い。これらは全て同じ選手なのではないかと予想できる
missing_height = df[df['Height'].isnull()].index.tolist()
df.drop(df.index[missing_height],inplace =True)
#欠損値をビジュアライズする
msno.bar(df,color='red')
#欠損値の割合が大きいカラムを消す
df.drop(['Loaned From','Release Clause','Joined'],axis=1,inplace=True)
msno.bar(df,color='red')
df.isnull().sum()[df.isnull().sum()>0].sort_values()
df = df.drop(['Jersey Number'], axis=1)
#'Value'と'Wage'を整形する
df_value = df['Value']
def value_to_int(df_value):
    try:
        value = float(df_value[1:-1]) #数字の文字列を浮動小数点に変換: float()
        suffix = df_value[-1:]

        if suffix == 'M':
            value = value * 1000000
        elif suffix == 'K':
            value = value * 1000
    except ValueError:
        value = 0
    return value

df['Value'] = df['Value'].apply(value_to_int)
df['Wage'] = df['Wage'].apply(value_to_int)
#要素（スカラー値）に対する関数 Seriesの各要素に適用: map(), apply() / DataFrameの各要素に適用: applymap()
#行・列（一次元配列）に対する関数 DataFrameの各行・各列に適用: apply()
df.columns
df.loc[:,'ID':'Weight']
df['Body Type'].unique()
df = df.drop(['Real Face','Contract Valid Until'], axis=1)
df.loc[:,'ID':'Weight']
df.loc[:,'LS':'RB']
#positionがGKの人ってもしかしてLS~RBまで欠けてる？
GK_df = df[df['Position'] == 'GK']
GK_df_LR = GK_df.loc[:,'LS':'RB']
GK_df_LR
#LS~RBまでの抜けのほとんどはGKのであったことがわかるね
GK_df_LR.isnull().sum()
df.loc[:,'Crossing':'GKReflexes']
#'LS'~'RB'までの数値の変動は'Crossing'~'GKReflexes'が決定付けているのでは？
player_features = (
    'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes'
)


from math import pi
idx = 1
plt.figure(figsize=(30,45))
for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():
    top_features = dict(features.nlargest(5))
    
    categories=top_features.keys()
    N = len(categories)

    values = list(top_features.values())
    values += values[:1]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(6, 5, idx, polar=True)

    plt.xticks(angles[:-1], categories, color='grey', size=12)
    ax.set_rlabel_position(0)
    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=10)
    plt.ylim(0,100)
    
    plt.subplots_adjust(hspace = 0.5)
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    plt.title(position_name, size=15, y=1.1)
    idx += 1
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)
import plotly.graph_objs as go
rating = pd.DataFrame(df.groupby(['Nationality'])['Overall'].sum().reset_index())
count = pd.DataFrame(rating.groupby('Nationality')['Overall'].sum().reset_index())

trace = [go.Choropleth(
            colorscale = 'YlOrRd',
            locationmode = 'country names',
            locations = count['Nationality'],
            text = count['Nationality'],
            z = count['Overall'],
)]

layout = go.Layout(title = 'Country vs Ratings')

fig = go.Figure(data = trace, layout = layout)
py.iplot(fig)
