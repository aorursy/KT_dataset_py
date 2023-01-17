# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pylab as pl # linear algebra + plots
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.colors as colors
import matplotlib.cm as cmx
from collections import defaultdict

T = pd.read_csv('../input/santader-2018-public-leaderboard/santander-value-prediction-challenge-publicleaderboard.csv')
T['date'] = pd.to_datetime(T.SubmissionDate)
MergerDate = '2018-08-13'
pl.figure(figsize=(20,5))
pl.plot(T.date, T.Score, '.')
pl.plot([pd.to_datetime(MergerDate)+pd.DateOffset(4)]*2, [0.4, 1.6], 'k')
pl.ylim([.4, 1.6])
T_merger = T[T.date <= pd.to_datetime(MergerDate)].groupby('TeamName').agg({'Score':'min'})
T_merger['Rank'] = pl.argsort(pl.argsort(T_merger.Score)) + 1
top500_merger = sorted(list(T_merger[T_merger.Rank <= 500].index), key=lambda x:T_merger.loc[x, 'Rank'])

T_End = T.groupby('TeamName').agg({'Score':'min'})
T_End['Rank'] = pl.argsort(pl.argsort(T_End.Score)) + 1
top500_end = sorted(list(T_End[T_End.Rank <= 500].index), key=lambda x:T_End.loc[x, 'Rank'])

print(len(set(top500_merger).intersection(set(top500_end))))
SteadyTeams = []
for n in range(10, 500, 10):
    SteadyTeams.append(len(set(top500_merger[:n]).intersection(set(top500_end[:n]))) / n * 100)
pl.plot(range(10, 500, 10), SteadyTeams)
pl.xlabel('top n teams')
pl.ylabel('percent of steady teams');
def plotCI(Arr, ci=[5, 95], color='b', alpha=.4):
    X = pl.arange(Arr.shape[1])
    Y0 = list(map(lambda col: pl.percentile(col, ci[0]), Arr.T))
    Y1 = list(map(lambda col: pl.percentile(col, ci[1]), Arr.T))
    M = Arr.mean(0)
    pl.fill_between(X, Y0, Y1, color=color, alpha=alpha)
    pl.plot(X, M, color=color, lw=2)


pl.figure(figsize=(7,10))
for top500, color in zip([top500_merger, top500_end], ['b', 'r']):
    Ranks = defaultdict(lambda :[])
    for day in range(0,9):
        T_temp = T[T.date <= pd.to_datetime(MergerDate) + pd.DateOffset(day)]
        T_temp = T_temp.groupby('TeamName').agg({'Score':'min'})
        T_temp['Rank'] = pl.argsort(pl.argsort(T_temp.Score)) + 1
        for team in top500:
            Ranks[team].append(T_temp.loc[team, 'Rank'] if team in T_temp.index else pl.nan)
    RankArr = pl.array(list(Ranks.values()))
    # replace nans with 500, they are only a few
    RankArr[pl.where(pl.isnan(RankArr))] = 500
    plotCI(RankArr, color=color)
pl.xlabel('Days after merger deadline')
pl.ylabel('Ranks')
Ylim = pl.gca().get_ylim()
pl.plot([4, 4], Ylim, 'k')
pl.plot([2, 2], Ylim, 'k:')
pl.plot([6, 6], Ylim, 'k:')

pl.legend(['top at end', 'top at merger', 'kernel release', 'release of extra groups'], loc=1);
top500 = top500_merger
Ranks = defaultdict(lambda :[])
for day in range(0,9):
    T_temp = T[T.date <= pd.to_datetime(MergerDate) + pd.DateOffset(day)]
    T_temp = T_temp.groupby('TeamName').agg({'Score':'min'})
    T_temp['Rank'] = pl.argsort(pl.argsort(T_temp.Score)) + 1
    for team in top500:
        Ranks[team].append(T_temp.loc[team, 'Rank'] if team in T_temp.index else pl.nan)

pl.figure(figsize=(10, 12))
cNorm = colors.Normalize(vmin=0, vmax=500)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='plasma')
for i, team in enumerate(top500):
    pl.plot(Ranks[team], color=scalarMap.to_rgba(i), alpha=.6)
pl.plot([4, 4], pl.gca().get_ylim(), 'k')
cb = pl.colorbar(pl.scatter(0,0,s=.1,c=0,vmin=0,vmax=500, cmap='plasma'))
cb.ax.invert_yaxis()
pl.xlabel('Days after merger deadline')
pl.ylabel('Ranks')
pl.gca().invert_yaxis()
pl.tight_layout()
ExpectedRank = {}
for team in Ranks:
    p = pl.polyfit(pl.arange(5), Ranks[team][:5], 1)
    r = pl.polyval(p, pl.arange(5, 9)).round()
    r[r<0] = 0
    ExpectedRank[team] = r

FinalRank = pl.array([Ranks[x][-1] for x in Ranks])
ExpectedRankFinal = pl.array([ExpectedRank[x][-1] for x in ExpectedRank])

pl.plot(FinalRank, ExpectedRankFinal, '.')
pl.xlabel('Final Rank')
pl.ylabel('Expected Rank')

pl.figure()
pl.hist(ExpectedRankFinal - FinalRank, 20)
pl.xlabel('jumps in the leaderboard compared to expected ranking')
pl.ylabel('#')

from scipy.stats import skew
print('skew of the distribution:', skew(ExpectedRankFinal - FinalRank))
print('number of people with ranking worse than expected:', sum((FinalRank - ExpectedRankFinal) > 0))
print('number of people with ranking better than expected:', sum((FinalRank - ExpectedRankFinal) < 0))
def getRanksOnDate(Date, offset):
    T_temp = T[T.date <= pd.to_datetime(Date) + pd.DateOffset(offset)]
    T_temp = T_temp.groupby('TeamName').agg({'Score':'min'})
    T_temp['Rank'] = pl.argsort(pl.argsort(T_temp.Score)) + 1
    return T_temp


def plotRankingChanges(kernelDate, top=500, interval=(-5,10), kernelName=''):
    T_temp = getRanksOnDate(kernelDate, interval[0])
    topTeams = sorted(list(T_temp[T_temp.Rank <= top].index), key=lambda x:T_temp.loc[x, 'Rank'])
    Ranks = defaultdict(lambda :[])
    for day in range(interval[0], interval[1]):
        T_temp = getRanksOnDate(kernelDate, day)
        for team in topTeams:
            Ranks[team].append(T_temp.loc[team, 'Rank'] if team in T_temp.index else pl.nan)
    
    cNorm = colors.Normalize(vmin=0, vmax=top)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='plasma')
    pl.figure(figsize=(15, 12))
    for i, team in enumerate(topTeams):
        pl.plot(range(interval[0], interval[1]), Ranks[team], color=scalarMap.to_rgba(i), alpha=.5)
    pl.plot([0, 0], pl.gca().get_ylim(), 'k')
    cb = pl.colorbar(pl.scatter(0,0,s=.1,c=0,vmin=0,vmax=top, cmap='plasma'))
    cb.ax.invert_yaxis()
    pl.xlabel('Days after kernel release')
    pl.ylabel('Ranks')
    pl.title(kernelName)
    pl.gca().invert_yaxis()
    pl.tight_layout()
    return Ranks

################################
Ranks1 = plotRankingChanges('2018-07-03', top=300, interval=(-10,20), kernelName='Pipeline Kernel, xgb + fe [LB1.39]')
Ranks2 = plotRankingChanges('2018-07-13', top=300, interval=(-10,20), kernelName='Santander_46_features')
Ranks3 = plotRankingChanges('2018-07-18', top=300, interval=(-10,20), kernelName='Leak (a collection of kernels)')
Ylim = pl.gca().get_ylim(); p=[]
p.append(pl.plot([0, 0], Ylim, 'k')[0])
p.append(pl.plot([1, 1], Ylim, 'k--')[0])
p.append(pl.plot([2, 2], Ylim, 'k:')[0])
pl.legend(p, ['Giba\'s Property', 'Breaking LB - Fresh start', 'Baseline with Lag Select Fake Rows Dropped'], loc=3)

Ranks4 = plotRankingChanges('2018-07-19', top=500, interval=(-5,35), kernelName='Breaking LB - Fresh start')
Ylim = pl.gca().get_ylim(); p=[]
p.append(pl.plot([0, 0], Ylim, 'k')[0])
p.append(pl.plot([2, 2], Ylim, 'k--')[0])
p.append(pl.plot([6, 6], Ylim, 'k-.')[0])
p.append(pl.plot([29, 29], Ylim, 'k:')[0])
pl.legend(p, ['Mohsin\'s kernel','Best Ensemble [67]','Best Ensemble [63] + Love is the Answer II','Jiazhen to Armamut via gurchetan1000 - 0.56'], loc=3)
RanksArr = pl.array(list(Ranks4.values()))
RankDiff = pl.diff(RanksArr, 1)
pl.plot(range(-4,35), pl.std(RankDiff, 0))
pl.plot([0,0], [2, 200], 'k')
pl.plot([29,29], [2, 200], 'k')
pl.plot([-5,35], [100, 100], 'k:')
pl.ylabel('variation (std) in Rank changes')
pl.xlabel('Days after Mohsin\'s kernel')