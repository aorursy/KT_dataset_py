import pandas as pd
import numpy as np
import os, re, sys
from scipy.special import logit, expit
import matplotlib.pyplot as plt
%matplotlib inline

teams = pd.read_csv('../input/womens-machine-learning-competition-2018/WTeams.csv')
id2team = dict(teams[['TeamID','TeamName']].values)

seeds = pd.read_csv('../input/womens-machine-learning-competition-2018/WNCAATourneySeeds.csv')

def do_plot_heatmap(probs, team_labels, filename):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    # try different colormaps: http://matplotlib.org/users/colormaps.html
    heatmap = ax.pcolormesh(probs, vmin=0, vmax=1, cmap=plt.cm.seismic)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.invert_yaxis()
    ax.tick_params(direction='out')
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    plt.xticks(rotation=90)
    
    nteams = len(team_labels)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(nteams)+0.5, minor=False)
    ax.set_yticks(np.arange(nteams)+0.5, minor=False)
    ax.set_xticklabels(team_labels, fontsize=8)
    ax.set_yticklabels(team_labels, fontsize=8)
    plt.savefig(filename, bbox_inches='tight')
    
def get_sub_year(df):
    return int(df.id.min()[:4])

def filter_sub_year(df, year):
    return df.loc[df.id.str.find(str(year))==0]

def show_heatmap(df):
    year = get_sub_year(df)
    # team ids conveniently ordered by region, seed
    team_ids = seeds.loc[seeds.Season==year].TeamID.values
    # maps global team ids to 0..67 based on target years tournament region & seed
    team2ind = {t:i for i,t in enumerate(team_ids)}
    # labels ordered by seed
    team_labels = list(map(id2team.get, team_ids))
    nteams = len(team_labels)

    parts = df.id.str.split('_')   # year, t1, t2
    t1 = parts.str[1].astype(int).map(team2ind)
    t2 = parts.str[2].astype(int).map(team2ind)
    # diagonal is notionally when a team plays itself - naturally 50/50?
    m = np.ones((nteams, nteams)) * 0.5
    # t1 is always the lower team id
    m[t1, t2] = df.pred
    m[t2, t1] = 1 - df.pred
    
    do_plot_heatmap(m, team_labels, f'plot_{year}_{df.pred.mean():.5f}.png')

def load_sub(name):
    df = pd.read_csv(name)
    df.columns = df.columns.str.lower()
    return df
from itertools import combinations

# https://www.kaggle.com/c/march-machine-learning-mania-2014/discussion/6776
# Win % = 0.50 + 0.03 * (weak seed minus strong seed) 
def seed_benchmark_prediction(s1, s2):
    return 0.5 + (s2 - s1) * 0.03

# team -> seed; seed x seed => prob
def seed_benchmark_prediction_df(year):
    yseeds = seeds.loc[seeds.Season==year].set_index('TeamID').sort_index()
    t2s = yseeds.Seed.str[1:3].astype(int)
    def gen():
        for t1, t2 in combinations(yseeds.index, 2):
            yield f'{year}_{t1}_{t2}', seed_benchmark_prediction(t2s[t1], t2s[t2])
    return pd.DataFrame.from_records(gen(), columns=['id','pred'])
show_heatmap(seed_benchmark_prediction_df(2017))
# quick copy/paste to show tournament round structure instead
def show_tournament_structure(year):
    seeds = pd.read_csv('../input/womens-machine-learning-competition-2018/WNCAATourneySeeds.csv')
    seeds = seeds.loc[seeds.Season==year].copy()
    team2seed = seeds.set_index('TeamID').Seed

    sub = pd.read_csv('../input/womens-machine-learning-competition-2018/WSampleSubmissionStage1.csv')
    sub = sub.loc[sub.ID.str.startswith(str(year))]
    
    slots = pd.read_csv('../input/womens-machine-learning-competition-2018/WNCAATourneySlots.csv')
    d = slots.set_index('StrongSeed').Slot.to_dict()
    d.update(slots.set_index('WeakSeed').Slot.to_dict())
    
    def tournament_path(seed):
        key = seed
        l = [seed]
        while key in d:
            key = d[key]
            l.append(key)
        return l
    
    seeds['path'] = seeds.Seed.map(tournament_path).apply(set)
    # e.g. W01 : {R2W1, R4W1, R5WX, W01, R1W1, R3W1, R6CH}
    seed2path = seeds.set_index('Seed').path

    # team ids conveniently ordered by region, seed
    team_ids = seeds.loc[seeds.Season==year].TeamID.values
    # maps global team ids to 0..67 based on target years tournament region & seed
    team2ind = {t:i for i,t in enumerate(team_ids)}
    # labels ordered by seed
    team_labels = list(map(id2team.get, team_ids))
    nteams = len(team_labels)
    
    parts = sub.ID.str.split('_')   # year, t1, t2
    sub['t1'] = parts.str[1].astype(int)
    sub['t2'] = parts.str[2].astype(int)
    sub['s1'] = sub.t1.map(team2seed)
    sub['s2'] = sub.t2.map(team2seed)
    sub['i1'] = sub.t1.map(team2ind)
    sub['i2'] = sub.t2.map(team2ind)
    
    # this is round 1..6
    mc = sub.apply(lambda row: len(seed2path[row.s1] & seed2path[row.s2]), axis=1)
    mc = (mc-1) / 5. # map to 0..1

    m = np.zeros((nteams,nteams))
    m[sub.i1, sub.i2] = mc.values
    m[sub.i2, sub.i1] = mc.values
    do_plot_heatmap(m, team_labels, f'tournament_map_{year}.png')
show_tournament_structure(2017)
sub = load_sub('../input/collaborative-filterings/CF.csv')
sub.head()
show_heatmap(filter_sub_year(sub, 2017))
show_heatmap(filter_sub_year(sub, 2016))
show_heatmap(filter_sub_year(sub, 2015))
show_heatmap(filter_sub_year(sub, 2014))
def subtract_seed_benchmark(sub):
    bm = seed_benchmark_prediction_df(get_sub_year(sub)).set_index('id')
    sub = sub.set_index('id')
    sub.pred -= bm.pred
    sub.pred = (sub.pred + 1) / 2  # remap [-1..1] to [0..1]
    return sub.reset_index()

def cmp_seed_benchmark(sub):
    year = get_sub_year(sub)
    sub = sub.set_index('id')
    sub['bm'] = seed_benchmark_prediction_df(year).set_index('id').pred
    sub['diff'] = sub.pred - sub.bm
    return sub.sort_values('diff')
show_heatmap(subtract_seed_benchmark(filter_sub_year(sub, 2015)))