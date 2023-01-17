# import libraries
import urllib.request, json 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline

import seaborn as sns
sns.set()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
#load data
df = pd.read_csv("../input/deliveries.csv")
df.shape
#utility functions to be used later

#1. Build a dictionary of Matches player by each batsman
played = {}
def BuildPlayedDict(x):
    #print(x.shape, x.shape[0], x.shape[1])
    for p in x.batsman.unique():
        if p in played:
            played[p] += 1
        else:
            played[p] = 1

df.groupby('match_id').apply(BuildPlayedDict)

#2. utility function to build some aggregate stats
def trybuild(lookuplist, buildlist):
    alist = []
    for i in buildlist.index:
        try:
            #print(i)
            alist.append(lookuplist[i])
            #print(alist)
        except KeyError:
            #print('except')
            alist.append(0)
    return alist
#Build the Summarized dataset 'BatmanStats' to do further analysis
BatsmanStats = df.groupby('batsman').aggregate({'ball': 'count', 'batsman_runs': 'sum'})
BatsmanStats.rename(columns={'ball': 'balls', 'batsman_runs': 'runs'}, inplace=True)
BatsmanStats['strike_rate'] = BatsmanStats['runs']/BatsmanStats['balls'] * 100
BatsmanStats['matches_played'] = [played[p] for p in BatsmanStats.index]
BatsmanStats['average']= BatsmanStats['runs']/BatsmanStats['matches_played']
    
for r in df.batsman_runs.unique():
    lookuplist = df[df.batsman_runs == r].groupby('batsman')['batsman'].count()
    BatsmanStats[str(r) + 's'] = trybuild(lookuplist, BatsmanStats)

#Filter Top batsmen in the league (palyed atleast 15 games, with an average of atleast 15, 
# strike rate of atleast 110 ordered by #srike rate)
bs = BatsmanStats
tb = bs[(bs.average > 15) & (bs.matches_played > 15) & (bs.strike_rate > 110)].sort_values(['average'], ascending = False)[:100]

#We get 80 such batsmen in our top batsmen dataset
len(tb)
# 'Dimension 1 for our analysis - Boundary Percentage')
tb['boundary_pct'] =  ((tb['4s'] * 4 ) + (tb['6s']  * 6))/tb['runs']

#'Dimension 2 for our analysis - DotBall (0s) Percentage')
tb['dotball_pct'] =  tb['0s']/tb['balls']

#Dimension 3 - Hit or Miss ratio
dfTop = df[df.batsman.isin(tb.index)]
hitormiss = {}
def computehitormiss(x):
    for p in x.batsman.unique():
        runs = x.batsman_runs.sum()
        avg = tb.loc[p]['average']
        if((runs >= 2 * avg) | (runs <= 0.33 * avg)):
            if p in hitormiss:
                hitormiss[p] += 1
            else:
                hitormiss[p] = 1

dfTop.groupby(['batsman','match_id']).apply(computehitormiss)
tb['hitmiss_pct'] = [hitormiss[p]/tb.loc[p]['matches_played'] for p in tb.index]
# slice the dataset to select only those columns relevant to our analysis
tb_knn = tb.loc[:,('boundary_pct','dotball_pct','hitmiss_pct','average','strike_rate')]
tb_knn.head()
# scale the features 
from sklearn import preprocessing
x = tb_knn.loc[:,('boundary_pct','dotball_pct','hitmiss_pct','average','strike_rate')].values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

#append scaled columns
tb_knn['bpct_scaled'] = x_scaled[:,0]
tb_knn['db_scaled'] =x_scaled[:,1]
tb_knn['hm_scaled'] =x_scaled[:,2]
tb_knn['avg_scaled'] =x_scaled[:,3]
tb_knn['sr_scaled'] =x_scaled[:,4]

#build feature matrix
X_tb = tb_knn.drop(['boundary_pct','dotball_pct','hitmiss_pct','average','strike_rate'], axis=1)

from sklearn.mixture import GaussianMixture as GMM     

model = GMM(n_components=2, covariance_type='full') 
model.fit(X_tb)                    
tb_clusters = model.predict(X_tb)    
tb_knn['clusters'] =  tb_clusters

print('Cluster - 0')
print(tb_knn[tb_knn.clusters == 0].head(15).index)

print('\n')

print('Cluster - 1')
print(tb_knn[tb_knn.clusters == 1].head(15).index)

#Renaming clusters
tb_knn.loc[tb_knn.clusters == 0,'clusters'] = 'Power Hitters'
tb_knn.loc[tb_knn.clusters == 1,'clusters'] = 'Hard Runners'


#Renaming columns for better readability
tb_knn.rename(columns={'boundary_pct':'boundary_percentage','dotball_pct':'dotball_percentage',
                      'hitmiss_pct':'hitormiss_percentage'}, inplace = True)

tb_knn.rename(columns={'bpct_scaled':'BoundaryPct_scaled',
                      'db_scaled':'DotBallPct_scaled',
                      'hm_scaled': 'HitorMissPct_scaled',
                      'avg_scaled': 'Average_scaled',
                      'sr_scaled': 'StrikeRate_scaled'},inplace=True)
sns.pairplot(tb_knn.drop(['Average_scaled','StrikeRate_scaled','BoundaryPct_scaled','HitorMissPct_scaled','DotBallPct_scaled'], axis = 1),hue = "clusters", size=2.5) #, markers=["o", "s"])
print("Batsmen with High Boundary Percentage.")
tb_knn.loc[:,('boundary_percentage','clusters')].sort_values('boundary_percentage', ascending = False).head()
print("Batsmen with low Dotball Percentage")
tb_knn.loc[:,('dotball_percentage','clusters')].sort_values('dotball_percentage', ascending = True).head()
print('Top HitorMiss ratio')
tb_knn.loc[:,('hitormiss_percentage','clusters')].sort_values('hitormiss_percentage', ascending = False).head()
print('Top batsmen by Average')
tb_knn.loc[:,('average','clusters')].sort_values('average',ascending = False).head()
print('Top Batsmen by Strike Rate')
tb_knn.loc[:,('strike_rate','clusters')].sort_values('strike_rate', ascending = False).head()
print('Bottom Batsmen by Strike Rate')
tb_knn.loc[:,('strike_rate','clusters')].sort_values('strike_rate').head()
plt.figure()
tb_knn.drop(['average','strike_rate','boundary_percentage','hitormiss_percentage','dotball_percentage'], axis = 1).boxplot(by="clusters", figsize=(15, 10))
plt.show()
from pandas.plotting import andrews_curves
plt.figure(figsize=(15,10))
andrews_curves(tb_knn.drop(['boundary_percentage','dotball_percentage','hitormiss_percentage','average','strike_rate'], axis=1), "clusters")
plt.title('Andrews Curves Plot', fontsize=20, fontweight='bold')
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(20, 15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(tb_knn.loc[:,'BoundaryPct_scaled'], tb_knn.loc[:,'StrikeRate_scaled'], tb_knn.loc[:,'HitorMissPct_scaled'], c=tb_clusters,
           cmap=plt.cm.Set1, edgecolor='k', s = tb_knn.loc[:,'DotBallPct_scaled']*50)

ax.set_xlabel('Boundary Percent')
ax.set_ylabel('Average')
ax.set_zlabel('Strike Rate')

plt.show()