import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/free_throws.csv")
df['minute'] = df.time.apply(lambda x: int(x[:len(x)-3]))

df['sec'] = df.time.apply(lambda x: int(x[len(x)-2:]))

df['abs_min'] = 12 - df['minute']+12*(df.period -1)

df['abs_time'] = 60*(df.abs_min-1) + 60 - df['sec']
def group_values(df,minute):

    made = len(df[(df.abs_min == minute) & (df.shot_made == 1)])

    total = len(df[df.abs_min == minute])

    return np.true_divide(made,total)



minutes = range(int(max(df.abs_min)))



per_min = []

for minu in minutes:

    per_min.append(group_values(df,minu))
plt.plot(minutes,per_min)

plt.title('Scoring % over time - Always worse at the beginning')

plt.xlim([1,48])

plt.ylim([0.65,0.85])

plt.plot([12,12],[0,1], '--', linewidth = 1, color = 'r')

plt.plot([24,24],[0,1], '--', linewidth = 1, color = 'r')

plt.plot([36,36],[0,1], '--', linewidth = 1, color = 'r')

plt.plot([48,48],[0,1], '--', linewidth = 1, color = 'r')

plt.xlabel('Minute')

plt.ylabel('Free Throws %')
minutes_df = pd.DataFrame()

minutes_df['minutes'] = range(int(max(df.abs_min)))

minutes_df['shots'] = minutes_df.minutes.apply(lambda x: len(df[df.abs_min == x]))

minutes_df['players_num'] = minutes_df.minutes.apply(lambda x: len(np.unique(df.player[df.abs_min == x])))
plt.plot(minutes_df['shots'])

plt.title('Number of Shots over time - mostly in the end of the querter')

plt.ylabel('# of Shots')

plt.xlabel('Minute')

plt.xlim([1,48])

plt.plot([12,12],[0,40000], '--', linewidth = 1, color = 'r')

plt.plot([24,24],[0,40000], '--', linewidth = 1, color = 'r')

plt.plot([36,36],[0,40000], '--', linewidth = 1, color = 'r')

plt.plot([48,48],[0,40000], '--', linewidth = 1, color = 'r')
players_df = pd.DataFrame()

players_df['name'] = np.unique(df.player)

players_df['pct'] = players_df.name.apply(lambda x: np.true_divide(len(df[(df.shot_made == 1) & (df.player ==x)]), len(df[df.player == x])))

df['player_pct'] = df.player.apply(lambda x: players_df.pct[players_df.name == x].values[0])

minutes_df['avg_pct'] = minutes_df.minutes.apply(lambda x: np.mean(df.player_pct[df.abs_min == x]))
plt.plot(minutes_df.avg_pct)

plt.xlim([2,48])

plt.ylim([0.65,0.85])

plt.plot([12,12],[0,1], '--', linewidth = 1, color = 'r')

plt.plot([24,24],[0,1], '--', linewidth = 1, color = 'r')

plt.plot([36,36],[0,1], '--', linewidth = 1, color = 'r')

plt.plot([48,48],[0,1], '--', linewidth = 1, color = 'r')

plt.xlabel('Minute')

plt.ylabel('Average FT% of players shooting')

plt.title('In the money time - the better shooters go to the line')