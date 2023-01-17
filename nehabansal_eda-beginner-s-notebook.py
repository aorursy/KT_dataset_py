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

from ipywidgets import interact,widgets
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
import seaborn as sns
%matplotlib inline
data = pd.read_csv("/kaggle/input/fifa19/data.csv")
data.drop(columns=['Unnamed: 0'],inplace=True)
data.head(3)
data.shape,data.drop_duplicates().shape,data.Name.drop_duplicates().shape,data.ID.drop_duplicates().shape
data.groupby(['Name','Nationality'],as_index=False).ID.count().query("ID >1")
@interact
def univariate(col = [ 'Age', 'Nationality','Club','Preferred Foot', 'International Reputation', 'Weak Foot',
       'Skill Moves', 'Work Rate', 'Body Type', 'Real Face', 'Position','Wage']):
    plt.figure(figsize=(25,5))
    if len(data[col].unique())>=15:
        data[col].value_counts(normalize=True).head(15).plot(kind='bar')
    else :
        data[col].value_counts(normalize=True).plot(kind='bar')
    plt.title(col)
data.columns
print(data.groupby(['Name'],as_index=False).Overall.mean().sort_values(['Overall'],ascending=False).head(10))
print("***************************************************************")
print(data.groupby(['Name'],as_index=False).Potential.mean().sort_values(['Potential'],ascending=False).head(10))
print(data.groupby(['Club'],as_index=False).Overall.mean().sort_values(['Overall'],ascending=False).head(10))
print("***************************************************************")
print(data.groupby(['Club'],as_index=False).Potential.mean().sort_values(['Potential'],ascending=False).head(10))
data['wage_numeric'] = data['Wage'].apply(lambda x: int(x.replace("K","").replace("€","")))
print(data.groupby(['Name'],as_index=False).wage_numeric.mean().sort_values(['wage_numeric'],ascending=False).head(10))
print(data.groupby(['Club'],as_index=False).wage_numeric.mean().sort_values(['wage_numeric'],ascending=False).head(10))
data['value_numeric'] = data['Value'].apply(lambda x: float(x.replace("K","").replace("M","").replace("€","")))
print(data.groupby(['Name'],as_index=False).value_numeric.mean().sort_values(['value_numeric'],ascending=False).head(10))
print(data.groupby(['Club'],as_index=False).value_numeric.mean().sort_values(['value_numeric'],ascending=False).head(10))
skill_cols = [ 'Crossing',
                   'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                   'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
                   'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
                   'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
                   'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
                   'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
                   'GKKicking', 'GKPositioning', 'GKReflexes','Overall','value_numeric','wage_numeric']
data_corr = data[skill_cols]
data_corr = data_corr.corr()
plt.figure(figsize=(8,7))
sns.heatmap(data_corr,xticklabels=data_corr.columns,yticklabels=data_corr.columns)
@interact
def skill_dependecy(col = ['Crossing',
                   'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
                   'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
                   'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
                   'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
                   'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
                   'Marking', 'StandingTackle', 'SlidingTackle']):
    data_corr[data_corr[col]!=1].sort_values([col],ascending=False).head(10)[[col]].plot(kind='bar')
    plt.title(col)
def get_most_similar_player(player,cnt):
    return  cosine_simi_df[cosine_simi_df.index!=player][player].sort_values(ascending=False).head(cnt)
data_sub = data[skill_cols+['Name']]
data_sub = data_sub[[col for col in data_sub.columns if col not in ['Overall','value_numeric','wage_numeric']]]

cosine_simi_df = pd.DataFrame(pairwise.cosine_similarity(np.array(data_sub.drop(columns=['Name']).fillna(0))),columns = data_sub.Name)
cosine_simi_df.index = data_sub.Name

@interact
def player_compare(player1=data.Name.unique() , player2 = data.Name.unique()):
    if player1!=player2:
        player_df = data[data.Name.isin([player1,player2])][skill_cols].T
        player_df.columns = [player1,player2]
        player_df.plot(kind='bar',figsize=(15,5))
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        get_most_similar_player(player1,10).plot()
        plt.subplot(1,2,2)
        get_most_similar_player(player2,10).plot()



