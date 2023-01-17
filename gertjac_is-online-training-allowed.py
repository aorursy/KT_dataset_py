import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline



df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

df_play = df[df.NflId==df.NflIdRusher].copy()



df_play['YardsFromOwnGoal'] = np.where(df_play.FieldPosition == df_play.PossessionTeam,

                                       df_play.YardLine, 50 + (50-df_play.YardLine))

df_play[['prev_game', 'prev_play', 'prev_team', 'prev_yfog', 'prev_yards']] = df_play[

        ['GameId', 'PlayId', 'Team', 'YardsFromOwnGoal', 'Yards']].shift(1)



filt = (df_play.GameId==df_play.prev_game) & (df_play.Team==df_play.prev_team) & (df_play.PlayId-df_play.prev_play<30)

df_play.loc[filt,'est_prev_yards'] = df_play[filt]['YardsFromOwnGoal'] - df_play[filt]['prev_yfog']



plt.figure(figsize=(8,8))

plt.title('deduced yards for %d of %d plays' % (sum(filt), len(filt)))

plt.scatter(*zip(*df_play[['est_prev_yards', 'prev_yards']].dropna().values), alpha=0.1)

plt.xlabel('deduced yards')

plt.ylabel('actual yards')

plt.show()