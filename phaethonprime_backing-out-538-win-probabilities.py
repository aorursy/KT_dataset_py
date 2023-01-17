import seaborn as sns

import numpy as np

from scipy import stats, optimize

import pandas as pd
df = pd.read_csv("../input/fivethirtyeight_ncaa_forecasts (2).csv")

df.head()
matchups = [[str(x+1), str(16-x)] for x in range(8)]

df = df[df.gender == 'mens']



pre = df[df.playin_flag==1]

data = []

for region in pre.team_region.unique():

    for seed in range(2, 17):

        res = pre[(pre.team_region == region) & (pre.team_seed.isin([str(seed)+'a', str(seed)+'b']))]

        if res.shape[0] > 1:

            data.append([])

            for _, row in res.iterrows():

                data[-1].extend([row.team_rating, row.rd1_win])



post = df[df.playin_flag == 0]

for region in post.team_region.unique():

    for matchup in matchups:

        res = post[(post.team_region == region) & (post.team_seed.isin(matchup))]

        if res.shape[0] > 1:

            data.append([])

            for _, row in res.iterrows():

                data[-1].extend([row.team_rating, row.rd2_win])
match = pd.DataFrame(data, columns=['Team1_Rating',"Team1_Prob", "Team2_Rating", "Team2_Prob"])
match['delta'] = match.Team1_Rating - match.Team2_Rating

match['win_extra'] = match.Team1_Prob - 0.5
sns.regplot('delta', 'win_extra', data=match, order=2);
def matcher(std, diff, prob):

    p = stats.norm.cdf(0, diff, std)

    return np.abs(p-prob)



stds = []

for _, row in match.iterrows():

    x0 = 1

    res = optimize.minimize(matcher, x0=x0, args=(row.delta, row.Team1_Prob))

    while res.status != 0 or res.x == x0:

        x0 *= 5

        res = optimize.minimize(matcher, x0=x0, args=(row.delta, row.Team1_Prob))

        if x0 > 1000:

            break

    stds.append(res.x)
stds