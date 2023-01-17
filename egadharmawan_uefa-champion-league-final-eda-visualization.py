import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/uefa-champion-league-final-all-season-19552019/UEFA Champion League All Season.csv')

df.head()
df.isnull().sum()
df.info()
sns.set(rc={'figure.figsize':(18,9), 'lines.linewidth': 5, 'lines.markersize': 5, "axes.labelsize":15}, style="whitegrid")



best_team = df.club[df['position'] == 'winner'].value_counts()
sns.barplot(x=best_team, y=best_team.index, data=df)
most_team = df['club'].value_counts()
sns.barplot(x=most_team, y=most_team.index, data=df)
best_nation = df.nation[df['position'] == 'winner'].value_counts()
sns.barplot(x=best_nation.index, y=best_nation, data=df)
most_nation = df.nation.value_counts()
sns.barplot(x=most_nation.index, y=most_nation, data=df)
best_coach = df.coach[df['position'] == 'winner'].value_counts().nlargest(15)
sns.barplot(x=best_coach, y=best_coach.index, data=df)
most_coach = df.coach.value_counts().nlargest(15)
sns.barplot(x=most_coach, y=most_coach.index, data=df)
best_formation = df.formation[(df['position'] == 'winner') & (df['formation'] != 'unknown') ].value_counts()
sns.barplot(x=best_formation.index, y=best_formation, data=df)
most_mvp = df.mvp[df['mvp'] != 'unknown'].value_counts()
sns.barplot(x=most_mvp, y=most_mvp.index, data=df)
dates = []



c = df['season'].unique()

ser = [c[x:x+10] for x in range(0, len(c), 10)]



for x in range(len(ser)):

    s = ' '.join(ser[x])

    f1 = s[1:5]

    f2 = s[-5:-1]

    f3 = str(f1) + '-' + str(f2)

    dates.append(f3)

    

performance = pd.DataFrame(dates, columns=['Time'])

performance
RMCF = []



win = df[(df['position'] == 'winner') & (df['club'] == 'Real Madrid CF')]



for y in range(len(ser)):

    wins = 0

    for s in ser[y]:

        for x in win.season:

            if x == s:

                wins += 1

    RMCF.append(wins)

    

performance['Real Madrid CF'] = RMCF
sns.lineplot(x='Time', y='Real Madrid CF', data=performance)
ACM = []



win = df[(df['position'] == 'winner') & (df['club'] == 'AC Milan')]



for y in range(len(ser)):

    wins = 0

    for s in ser[y]:

        for x in win.season:

            if x == s:

                wins += 1

    ACM.append(wins)

    

performance['AC Milan'] = ACM
sns.lineplot(x='Time', y='AC Milan', data=performance)
LFC = []



win = df[(df['position'] == 'winner') & (df['club'] == 'Liverpool FC')]



for y in range(len(ser)):

    wins = 0

    for s in ser[y]:

        for x in win.season:

            if x == s:

                wins += 1

    LFC.append(wins)

    

performance['Liverpool FC'] = LFC
sns.lineplot(x='Time', y='Liverpool FC', data=performance)
BM = []



win = df[(df['position'] == 'winner') & (df['club'] == 'FC Bayern Munchen')]



for y in range(len(ser)):

    wins = 0

    for s in ser[y]:

        for x in win.season:

            if x == s:

                wins += 1

    BM.append(wins)

    

performance['FC Bayern Munchen'] = BM
sns.lineplot(x='Time', y='FC Bayern Munchen', data=performance)
BFC = []



win = df[(df['position'] == 'winner') & (df['club'] == 'Barcelona FC')]



for y in range(len(ser)):

    wins = 0

    for s in ser[y]:

        for x in win.season:

            if x == s:

                wins += 1

    BFC.append(wins)

    

performance['Barcelona FC'] = BFC
sns.lineplot(x='Time', y='Barcelona FC', data=performance)
AFCA = []



win = df[(df['position'] == 'winner') & (df['club'] == 'AFC Ajax')]



for y in range(len(ser)):

    wins = 0

    for s in ser[y]:

        for x in win.season:

            if x == s:

                wins += 1

    AFCA.append(wins)

    

performance['AFC Ajax'] = AFCA
sns.lineplot(x='Time', y='AFC Ajax', data=performance)