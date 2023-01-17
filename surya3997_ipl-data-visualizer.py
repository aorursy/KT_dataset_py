import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

from matplotlib import style



style.use('seaborn-whitegrid')



print(check_output(["ls", "../input"]).decode("utf8"))
df_del = pd.read_csv("../input/deliveries.csv")

df_mat = pd.read_csv("../input/matches.csv")
print(df_del.head())

print(df_del.info())
print(df_mat.head())

print(df_mat.info())
teams = set(df_mat["team1"]) | set(df_mat["team2"])

teams = list(teams)

teams.sort()
played = np.array([])

won = np.array([])

for team in teams:

    total_mat = df_mat.query("(team1 == @team or team2 == @team)")

    mat_won = total_mat.query("winner == @team")

    played = np.append(played, total_mat.shape[0])

    won = np.append(won, mat_won.shape[0])
x = np.array(range(len(teams)))



plt.bar(x, played, label='Lost or no result')

l2 = plt.bar(x, won, label='Matches won')

plt.xticks(x, teams, rotation=45, ha='right')

plt.subplots_adjust(bottom=0.27, left=0.1, right=0.88)

plt.xlabel('Team Names')

plt.ylabel('No. of Matches')

plt.legend(bbox_to_anchor=(1, 1))

plt.title('No. of matches played\nvs\nNo.of matches won')

plt.show()
win_percent = won / played * 100

ss = pd.DataFrame(win_percent, teams).sort_values(0, ascending=False)

ss.columns = ['Win Percent']

plt.close()

l1 = ss.plot.bar()

plt.xticks(x, teams, rotation=45, ha='right')

plt.subplots_adjust(bottom=0.27)

plt.legend()

plt.show()