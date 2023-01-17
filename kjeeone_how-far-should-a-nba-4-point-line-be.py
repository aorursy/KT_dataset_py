# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 

import seaborn as sns
df = pd.read_csv('/kaggle/input/nba-shot-logs/shot_logs.csv')
make_pct_pps_dist = pd.pivot_table(df, index='SHOT_DIST', values = ['FGM','PTS'] ).reset_index()

make_pct_dist = pd.pivot_table(df, index='SHOT_DIST', values = ['FGM'] ).reset_index()

make_pct_dist_cnt = pd.pivot_table(df, index='SHOT_DIST', values = ['FGM'], aggfunc='count' ).reset_index()
make_pct_dist_cnt.columns = ['SHOT_DIST','SHOT_COUNT']
make_pct_dist_cnt.plot.line(x='SHOT_DIST')
threes = make_pct_dist_cnt[make_pct_dist_cnt['SHOT_DIST'] > 23.75]

total_threes = threes.SHOT_COUNT.sum()

threes['shot_pct'] = threes.SHOT_COUNT / total_threes
threes.head()
from sklearn.linear_model import LogisticRegression
make_pct_dist.plot.line(x='SHOT_DIST')

make_pct_dist_cnt.plot.line(x='SHOT_DIST')
three_range = df[df.SHOT_DIST > 22]

log_reg = LogisticRegression(solver='lbfgs')

log_reg.fit(three_range.SHOT_DIST.values.reshape(-1, 1),three_range.FGM.values.reshape(-1, 1))

print('Score: ',log_reg.score(three_range.SHOT_DIST.values.reshape(-1, 1),three_range.FGM.values.reshape(-1, 1)))
predicted_fgm_pct = log_reg.predict_proba(np.array(24).reshape(1,-1))

actual_fgm_pct = float(make_pct_dist[make_pct_dist['SHOT_DIST'] == 24].FGM)

print('predicted: ', predicted_fgm_pct[0][1], ' actual: ',actual_fgm_pct)
from sklearn.metrics import mean_absolute_error, mean_squared_error

from math import sqrt



makepcts = make_pct_dist[(make_pct_dist.SHOT_DIST >= 22) & (make_pct_dist.SHOT_DIST <= 27.3)]

ypred = log_reg.predict_proba(makepcts.SHOT_DIST.values.reshape(-1,1))

ypred[:,1]

print('RMSE: ',sqrt(mean_squared_error(makepcts.FGM,ypred[:,1])))

print('MAE: ',mean_absolute_error(makepcts.FGM,ypred[:,1]))
log_reg_curve = [log_reg.predict_proba(np.array(i/10).reshape(1,-1))[0][1] for i in range(220,940)]

plt.plot([i/10 for i in range(220,940)], log_reg_curve)

plt.axvline(x=22, label = 'corner 3', color = 'green')

plt.axvline(x=23.75, label = '3 point arc', color = 'orange')

plt.axvline(x=25.2, label = "Ken's 3", color = 'red')

plt.axvline(x=47, label = "Half Court", color = 'black')

plt.legend()

log_reg_curve = [log_reg.predict_proba(np.array(i/10).reshape(-1,1))[0][1] for i in range(220,275)]

actual_fgm_pct = [float(make_pct_dist[make_pct_dist['SHOT_DIST'] == i/10].FGM) for i in range(220,275)]

plt.plot([i/10 for i in range(220,275)], log_reg_curve)

plt.plot([i/10 for i in range(220,275)], actual_fgm_pct)

ken_threes = threes.copy()

ken_threes['SHOT_DIST'] = ken_threes.SHOT_DIST.apply(lambda x: x+1.45)

ken_threes['shot_pct'] = ken_threes.SHOT_COUNT / ken_threes.SHOT_COUNT.sum()
ken_threes['proj_make_pct'] = ken_threes.SHOT_DIST.apply(lambda x: log_reg.predict_proba(np.array(x).reshape(-1,1))[0][1])
ken_threes['pps'] = ken_threes.shot_pct*ken_threes.proj_make_pct*3
ken_threes.pps.sum()
def pps_fpoint(three_distribution, size, pps_benchmark):

    new_distribution = three_distribution

    plot_data = []

    closest = ''

    diff = 1000

    for i in range(1,size*10):

        new_distribution.SHOT_DIST = new_distribution.SHOT_DIST.apply(lambda x: x + (i/10))

        new_distribution['proj_make_pct'] = new_distribution.SHOT_DIST.apply(lambda x: log_reg.predict_proba(np.array(x).reshape(-1,1))[0][1])

        new_distribution['pps'] = new_distribution.shot_pct*new_distribution.proj_make_pct*4

        points_per_shot = new_distribution.pps.sum()

        plot_data.append((25.2+i/10,points_per_shot))

        if abs(points_per_shot - pps_benchmark) < diff:

            closest = str(25.2+i/10)

            diff = abs(points_per_shot - pps_benchmark)

    return (closest,diff,points_per_shot), plot_data

        

        

        
closest, four_pt_dist = pps_fpoint(ken_threes,30,.92)
closest
four_pt_dist