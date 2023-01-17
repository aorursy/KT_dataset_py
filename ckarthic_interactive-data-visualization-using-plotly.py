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

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as pyoffline
from plotly import tools

pyoffline.init_notebook_mode(connected=True)
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
#sns.pairplot(tb_knn.drop(['Average_scaled','StrikeRate_scaled','BoundaryPct_scaled','HitorMissPct_scaled','DotBallPct_scaled'], axis = 1),hue = "clusters", size=2.5) #, markers=["o", "s"])
fig = ff.create_scatterplotmatrix(
    tb_knn.drop(['Average_scaled','StrikeRate_scaled','BoundaryPct_scaled','HitorMissPct_scaled','DotBallPct_scaled'], axis = 1), 
                index='clusters', size=4, height=1000, width=1000
                ,diag = 'histogram', title = 'IPL PowerHitters Vs HardRunners'#, colormap = 'Reds'
                , text = tb_knn.index
                )
pyoffline.iplot(fig)
df_boundary_pct =  tb_knn.loc[:,('boundary_percentage','clusters')].sort_values('boundary_percentage', ascending = False)#.head()
data = [{ 
        'y' : df_boundary_pct[df_boundary_pct['clusters'] == c].boundary_percentage,
        'x' : df_boundary_pct[df_boundary_pct['clusters'] == c].index,
        'name' : c,
        'mode' : 'markers',
        }
        for c in ['Power Hitters','Hard Runners']]
layout = go.Layout({'xaxis' : {'title' : 'Batsmen'},
                    'yaxis': {'title' : 'Boundary Percentage'},
                    'title' : 'Batsmen Boundary Percentage'
                   })

fig = go.Figure(data = data, layout = layout)
pyoffline.iplot(fig)
df_dotball_pct = tb_knn.loc[:,('dotball_percentage','clusters')].sort_values('dotball_percentage', ascending = True)#.head()
data = [{
        'y' : df_dotball_pct[df_dotball_pct['clusters'] == c].dotball_percentage,
        'x' : df_dotball_pct[df_dotball_pct['clusters'] == c].index,
        'name' : c,'mode' : 'markers',
        } for c in ['Power Hitters','Hard Runners']]
layout = go.Layout({'xaxis' : {'title' : 'Batsmen'},
                    'yaxis': {'title' : 'Dotball Percentage'},
                    'title' : 'Batsmen Dotball Percentage'
                   })
fig = go.Figure(data = data, layout = layout)
pyoffline.iplot(fig)
df_hitmiss = tb_knn.loc[:,('hitormiss_percentage','clusters')].sort_values('hitormiss_percentage', ascending = False)
data = [{'y' : df_hitmiss[df_hitmiss['clusters'] == c].hitormiss_percentage,
        'x' : df_hitmiss[df_hitmiss['clusters'] == c].index,
        'name' : c,'mode' : 'markers',
        } for c in ['Power Hitters','Hard Runners']]
layout = go.Layout({'xaxis' : {'title' : 'Batsmen'},
                    'yaxis': {'title' : 'HitorMiss Percentage'},
                    'title' : 'Batsmen HitorMiss Ratio'
                   })
fig = go.Figure(data = data, layout = layout)
pyoffline.iplot(fig)
df_avg = tb_knn.loc[:,('average','clusters')].sort_values('average',ascending = False)
data = [{'y' : df_avg[df_avg['clusters'] == c].average,
        'x' : df_avg[df_avg['clusters'] == c].index,
        'name' : c,'mode' : 'markers',
        } for c in ['Power Hitters','Hard Runners']]
layout = go.Layout({'xaxis' : {'title' : 'Batsmen'},
                    'yaxis': {'title' : 'Average Percentage'},
                    'title' : 'Batsmen Average'
                   })
fig = go.Figure(data = data, layout = layout)
pyoffline.iplot(fig)
df_strikerate = tb_knn.loc[:,('strike_rate','clusters')].sort_values('strike_rate', ascending = False)
data = [{'y' : df_strikerate[df_strikerate['clusters'] == c].strike_rate,
        'x' : df_strikerate[df_strikerate['clusters'] == c].index,
        'name' : c,'mode' : 'markers',
        } for c in ['Power Hitters','Hard Runners']]

data.append({'y' : df_strikerate.loc['TM Dilshan'].strike_rate,
        'x' : 'TM Dilshan',
        'name' : 'TM Dilshan','mode' : 'markers', 'marker' : {'size': '10'},
        })
# data[0]['y'] = data[0]['y'][:-1] # Remove Dilshan in the first trace
layout = go.Layout({'xaxis' : {'title' : 'Batsmen'},
                    'yaxis': {'title' : 'Strike Rate'},
                    'title' : 'Batsmen Strike Rate'
                   })
fig = go.Figure(data = data, layout = layout)
pyoffline.iplot(fig)
avg_ph = tb_knn[tb_knn['clusters'] == 'Power Hitters']['average']
avg_hr = tb_knn[tb_knn['clusters'] == 'Hard Runners']['average']

sr_ph = tb_knn[tb_knn['clusters'] == 'Power Hitters']['strike_rate']
sr_hr = tb_knn[tb_knn['clusters'] == 'Hard Runners']['strike_rate']

bp_ph = tb_knn[tb_knn['clusters'] == 'Power Hitters']['boundary_percentage']
bp_hr = tb_knn[tb_knn['clusters'] == 'Hard Runners']['boundary_percentage']

dp_ph = tb_knn[tb_knn['clusters'] == 'Power Hitters']['dotball_percentage']
dp_hr = tb_knn[tb_knn['clusters'] == 'Hard Runners']['dotball_percentage']

hm_ph = tb_knn[tb_knn['clusters'] == 'Power Hitters']['hitormiss_percentage']
hm_hr = tb_knn[tb_knn['clusters'] == 'Hard Runners']['hitormiss_percentage']

trace_avg_ph = go.Box(    y=avg_ph,    name = 'Power Hitters',    text = avg_ph.index  )
trace_avg_hr = go.Box(    y=avg_hr,    name = 'Hard Runners',    text = avg_hr.index   )
trace_sr_ph = go.Box(    y=sr_ph,    name = 'Power Hitters',    text = sr_ph.index  )
trace_sr_hr = go.Box(    y=sr_hr,    name = 'Hard Runners',    text = sr_hr.index   )

trace_bp_ph = go.Box(    y=bp_ph,    name = 'Power Hitters',    text = bp_ph.index  )
trace_bp_hr = go.Box(    y=bp_hr,    name = 'Hard Runners',    text = bp_hr.index   )
trace_dp_ph = go.Box(    y=dp_ph,    name = 'Power Hitters',    text = dp_ph.index  )
trace_dp_hr = go.Box(    y=dp_hr,    name = 'Hard Runners',    text = dp_hr.index   )

trace_hm_ph = go.Box(    y=hm_ph,    name = 'Power Hitters',    text = hm_ph.index  )
trace_hm_hr = go.Box(    y=hm_hr,    name = 'Hard Runners',    text = hm_hr.index   )

fig = tools.make_subplots(rows=3, cols=2)

fig.append_trace(trace_avg_ph, 1, 1)
fig.append_trace(trace_avg_hr, 1, 1)

fig.append_trace(trace_sr_ph, 1, 2)
fig.append_trace(trace_sr_hr, 1, 2)

fig.append_trace(trace_bp_ph, 2, 1)
fig.append_trace(trace_bp_hr, 2, 1)

fig.append_trace(trace_dp_ph, 2, 2)
fig.append_trace(trace_dp_hr, 2, 2)

fig.append_trace(trace_dp_ph, 3, 1)
fig.append_trace(trace_dp_hr, 3, 1)


fig['layout'].update(title='Power Hitters Vs Hard Runners', height = 1200)
fig['layout']['yaxis1']['title'] = 'Batsmen Average'
fig['layout']['yaxis2']['title'] = 'Batsmen Strike Rate'
fig['layout']['yaxis3']['title'] = 'Batsmen Boundary Percentage'
fig['layout']['yaxis4']['title'] = 'Batsmen Dotball Percentage'
fig['layout']['yaxis5']['title'] = 'Batsmen HitorMiss Ratio'
pyoffline.iplot(fig)

