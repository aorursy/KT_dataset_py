% matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
csv_files =[

    '../input/twosigmapubliclb/two-sigma-using-news-to-pre-public-leaderboard/two-sigma-financial-news-publicleaderboard.csv',

    '../input/lb20190215/publicleaderboarddata20190215/two-sigma-financial-news-publicleaderboard.csv',

    '../input/lb20190301/publicleaderboarddata20190301/two-sigma-financial-news-publicleaderboard.csv',

    '../input/lb20190313/publicleaderboarddata20190313/two-sigma-financial-news-publicleaderboard.csv',

    '../input/lb20190401/publicleaderboarddata20190401/two-sigma-financial-news-publicleaderboard.csv',

    '../input/lb20190418/publicleaderboarddata20190418/two-sigma-financial-news-publicleaderboard.csv'

]



dates = ['Public', '20190215', '20190301', '20190313', '20190401', '20190418']



csvs = []

for num, fn in enumerate(csv_files):

    csvs += [pd.read_csv(fn)]

    csvs[-1] = csvs[-1].groupby('TeamName').Score.max().reset_index()

    csvs[-1]['stage'] = num

    scores = csvs[-1].Score.unique()

    scores_rank = np.argsort(np.argsort(scores))

    csvs[-1]['score_rank'] = csvs[-1].Score.map(dict(zip(scores, scores_rank))) 

df = pd.concat(csvs, ignore_index=True)

teams = df.TeamName.unique()
linthreshy = df[df.stage==df.stage.max()].Score.std()



median = int(np.round(df[df.stage==df.stage.max()].Score.median()))

print('Median:', median)

plt.figure(figsize=(24,12))

for team in teams:

    team_scores = df[df.TeamName==team].sort_values('stage')[['Score', 'stage']].values

    plt.plot(team_scores[:, 1], team_scores[:, 0]-median, alpha=0.2);

plt.yscale('symlog', linthreshy=linthreshy)

plt.yticks(np.arange(int(df.Score.min())-int(median), int(df.Score.max())-int(median)), np.arange(int(df.Score.min()), int(df.Score.max())))

plt.xticks(np.arange(len(dates)), dates);

plt.ylabel('Score');


plt.figure(figsize=(24,200))

for team in teams:

    team_scores = df[df.TeamName==team].sort_values('stage')[['Score', 'stage']].values

    plt.plot(team_scores[:, 1], team_scores[:, 0]-median, alpha=0.5);

    for pos in team_scores:

        plt.text(pos[1], pos[0]-median, f'{team.upper()} [{pos[0]:.4f}]')

plt.yscale('symlog', linthreshy=linthreshy)

plt.yticks(np.arange(int(df.Score.min())-int(median), int(df.Score.max())-int(median)), np.arange(int(df.Score.min()), int(df.Score.max())))

plt.xticks(np.arange(len(dates)), dates);

plt.ylabel('Score');
plt.figure(figsize=(24,200))

for team in teams:

    team_scores = df[df.TeamName==team].sort_values('stage')[['score_rank', 'stage', 'Score']].values

    plt.plot(team_scores[:, 1], team_scores[:, 0]-350, alpha = 0.2);

    for pos in team_scores:

        plt.text(pos[1], pos[0]-350, f'{team.upper()} [{pos[2]:.4f}]')

plt.yscale('symlog', linthreshy=350, basey=10)

plt.yticks([-350, -250, -150, 50, 150, 250, 350, 650, 1150, 1650], np.array([0, 100, 200, 400, 500, 600, 700, 1000, 1500, 2000]))

plt.xticks(np.arange(len(dates)), dates);

plt.ylabel('Rank');