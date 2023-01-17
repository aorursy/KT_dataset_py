from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats, optimize

import pandas as pd

from itertools import combinations
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
poly = np.polyfit(match.delta, match.win_extra, 2)

poly
data = []

for region in df.team_region.unique():

    for matchup in matchups:

        res = post[(post.team_region == region) & (post.team_seed.isin(matchup))]

        if res.shape[0] > 1:

            data.append([])

            for _, row in res.iterrows():

                data[-1].extend([row.team_name, row.team_rating, int(row.team_seed)])

        else:

            # grab the 'a' and 'b' teams. 

            seeds = matchup + [x+'a' for x in matchup] + [x+'b' for x in matchup]

            res = df[(df.team_region == region) & (df.team_seed.isin(seeds))]

            for t1, t2 in combinations(res.team_name.tolist(), 2):

                res2 = res[res.team_name.isin([t1, t2])]

                data.append([])

                for _, row in res2.iterrows():

                    seed = row.team_seed if len(row.team_seed) < 3 else row.team_seed[:-1]

                    data[-1].extend([row.team_name, row.team_rating, int(seed)])
rd2 = pd.DataFrame(data, columns=['Team1', 'Rank1', 'Seed1', 'Team2', 'Rank2', 'Seed2'])
def upset(row):

    top_rank = max(row.Rank1, row.Rank2)

    top_num = '1' if top_rank == row.Rank1 else '2'

    low_num = '1' if top_num == '2' else '2'

    seed_delta = row['Seed'+top_num] - row['Seed'+low_num]

    rank_delta = row['Rank'+top_num] - row['Rank'+low_num]

    prob = np.polyval([-0.00116991,  0.0461334 ,  0.01831479], np.abs(rank_delta))

    return prob * np.sign(seed_delta), top_num



def matchup_str(x, direc='l'):

    if direc == 'l':

        top_num = '2' if x.Seed1 > x.Seed2 else '1'

    else:

        top_num = '1' if x.Seed1 > x.Seed2 else '2'

    low_num = '1' if top_num == '2' else '2'

    return "{} {}".format(x['Seed'+top_num],x['Team'+top_num])
rd2.shape
rd2['upset_data'] = rd2.apply(upset, axis=1)

# this is so hacky but it does the job without a for-loop

# there's also some legacy code here that did something for a plot I'm not making any more

rd2['upset'] = rd2.upset_data.apply(lambda x: x[0])

rd2['matchup_left'] = rd2.apply(matchup_str, axis=1, args=['l'])

rd2['matchup_right'] = rd2.apply(matchup_str, axis=1, args=['r'])

rd2['matchup'] = rd2.apply(lambda x: x.matchup_left + " v " + x.matchup_right, axis=1)



# only look if the seeds are more than 2 apart

rd2 = rd2[(np.abs(rd2.Seed1-rd2.Seed2) >= 2) & (rd2.upset > -.2)]



rd2.sort_values('upset', inplace=True, ascending=False)
sns.set(style="white", context="talk")



f, ax = plt.subplots(figsize=(6, 10))

sns.barplot(x="upset", y="matchup", data=rd2, label="Win Probability", palette="RdBu_r")

ax.set_xlabel("Win Probability Above 50/50 (Postive = upset)")

ax.plot([0, 0], [-1, rd2.shape[0]], '-k'); ax.set_ylabel("");
new_matchups = [[str(a) for a in x] for x in combinations(range(1,17), 2)]



data = []

for region in df.team_region.unique():

    for matchup in new_matchups:

        res = post[(post.team_region == region) & (post.team_seed.isin(matchup))]

        if res.shape[0] > 1:

            data.append([])

            for _, row in res.iterrows():

                data[-1].extend([row.team_name, row.team_rating, int(row.team_seed)])

        else:

            # grab the 'a' and 'b' teams. 

            seeds = matchup + [x+'a' for x in matchup] + [x+'b' for x in matchup]

            res = df[(df.team_region == region) & (df.team_seed.isin(seeds))]

            for t1, t2 in combinations(res.team_name.tolist(), 2):

                res2 = res[res.team_name.isin([t1, t2])]

                data.append([])

                for _, row in res2.iterrows():

                    seed = row.team_seed if len(row.team_seed) < 3 else row.team_seed[:-1]

                    data[-1].extend([row.team_name, row.team_rating, int(seed)])
rdall = pd.DataFrame(data, columns=['Team1', 'Rank1', 'Seed1', 'Team2', 'Rank2', 'Seed2'])



rdall['upset_data'] = rdall.apply(upset, axis=1)

# this is so hacky but it does the job without a for-loop

# there's also some legacy code here that did something for a plot I'm not making any more

rdall['upset'] = rdall.upset_data.apply(lambda x: x[0])

rdall['matchup_left'] = rdall.apply(matchup_str, axis=1, args=['l'])

rdall['matchup_right'] = rdall.apply(matchup_str, axis=1, args=['r'])

rdall['matchup'] = rdall.apply(lambda x: x.matchup_left + " v " + x.matchup_right, axis=1)



# only look if the seeds are within 5 or probability is close

rdall = rdall[(np.abs(rdall.Seed1-rdall.Seed2) >= 2) & (rdall.upset >= -0.05)]



rdall.sort_values('upset', inplace=True, ascending=False)
f, ax = plt.subplots(figsize=(6, 15))

sns.barplot(x="upset", y="matchup", data=rdall, label="Win Probability", palette="RdBu_r")

ax.set_xlabel("Win Probability Above 50/50 (Postive = upset)")

ax.plot([0, 0], [-1, rdall.shape[0]], '-k'); ax.set_ylabel("");