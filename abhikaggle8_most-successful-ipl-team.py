import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
df = pd.read_csv("../input/matches.csv")
df = df[['team1', 'team2', 'winner']]
df.head(2)
teams = df.team1.unique()
teams
winPercent = []

for each_team in teams:
    played_matches = np.count_nonzero(df['team1'].astype(str).str.contains(each_team)) + np.count_nonzero(df['team2'].astype(str).str.contains(each_team))
    matches_won = np.count_nonzero(df['winner'].astype(str).str.contains(each_team))   
    winPercent.append(100 * (matches_won / played_matches))
    
winPercent, teams = zip(*sorted(zip(winPercent, teams))) #Sort teams as per winning percentage (Descending order)

plt.figure(figsize=(12,6))
plt.barh(range(len(winPercent)), winPercent, align='center')
plt.yticks(range(len(winPercent)), teams)
plt.title("IPL Teams: Winning Percentage")
plt.show()