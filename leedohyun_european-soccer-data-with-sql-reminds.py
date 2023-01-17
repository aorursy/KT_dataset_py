# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
path = '../input/'
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables
player = pd.read_sql("SELECT * FROM Player",conn)
player_att = pd.read_sql("SELECT * FROM Player_Attributes",conn)
player_att.dropna(inplace=True)
player_att = player_att.drop_duplicates(subset='player_api_id')
player_att.overall_rating.describe()
plt.figure(figsize=(12,8))
plt.title("Overall Rating Distributions",fontdict={'fontsize':15})
plt.xlabel("Overall")
plt.ylabel("Ratio")
sns.distplot(list(player_att.overall_rating),bins=100)
player_att.columns
player_att.dropna(inplace=True)
player_att.head()
model = KMeans(n_clusters=5)
model.fit(player_att[['overall_rating','potential']])
predict = pd.DataFrame(model.predict(player_att[['overall_rating','potential']]))
predict.columns=['predict']
plt.figure(figsize=(8,7))
plt.scatter(x=player_att.overall_rating,y=player_att.potential,c=predict.predict)
plt.title("Clustering Using Overall Ratings and Potential Scores",fontdict={'fontsize':15})
plt.ylabel("Potential")
plt.xlabel("Overall_Rating")
player_att.columns
plt.figure(figsize=(10,8))
corr = player_att.corr()
sns.heatmap(corr)
forward_features = ['finishing','volleys']
mid_features = ['short_passing','vision']
defender_features = ['standing_tackle','sliding_tackle']
gk_features = ['gk_diving','gk_handling','gk_kicking','gk_positioning']

"""
def position_column(position):
    feature = str(position)+'_features'
    player_att[str(position)] = player_att[x for x in feature]
    """
print(map(sum,player_att[forward_features]))
player_att['forward'] = (player_att[forward_features].iloc[:,0] + player_att[forward_features].iloc[:,1])/2
player_att['mid'] = (player_att[mid_features].iloc[:,0] + player_att[mid_features].iloc[:,1])/2
player_att['defender'] = (player_att[defender_features].iloc[:,0] + player_att[defender_features].iloc[:,1])/2
player_att['gk'] = (player_att[gk_features].iloc[:,0] + player_att[gk_features].iloc[:,1] +
                   player_att[gk_features].iloc[:,2]+player_att[gk_features].iloc[:,3])/4
player_att
def cluster_position(pos1,pos2):
    model = KMeans(n_clusters=4)
    model.fit(player_att[['forward','mid','defender','gk']])
    predict = pd.DataFrame(model.predict(player_att[['forward','mid','defender','gk']]))
    predict.columns=['predict']
    plt.figure(figsize=(8,7))
    plt.scatter(x=player_att[str(pos1)],y=player_att[str(pos2)],c=predict.predict,marker='o')
    plt.title("Clustering Using "+pos1+" Stats "+pos2+" Stats",fontdict={'fontsize':15})
    plt.ylabel(pos2)
    plt.xlabel(pos1)
cluster_position('forward','mid')
cluster_position('mid','defender')
cluster_position('forward','defender')
cluster_position('gk','forward')
def find_player(name):
    id = player[player.player_name==name]['player_api_id']
    return(player_att[player_att.player_api_id==int(id)])
find_player("Lionel Messi")
find_player("Aaron Doran")
country = pd.read_sql("""SELECT * FROM Country""",conn)
country
country