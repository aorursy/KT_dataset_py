import pandas as pd
import numpy as np
import seaborn as sns
import glob
# Import all files
all_files = glob.glob("../input/premier-league-standing-all-season-19922020/Premier League*.csv")

# Combine all DataFrame
all = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df.columns = ['Position','Club','Played','Won','Drawn','Lost','GF','GA','GD','Points','season']
    all.append(df)

# Sort DataFrame
df = pd.concat(all, axis=0, ignore_index=True, sort=False)
df = df.sort_values(['season','Position'], ascending=[True, True])
df.head()
df.info()
df.isnull().sum()
# Ignore this, this is just my setup on how to use seaborn
sns.set(rc={'figure.figsize':(18,9), 'lines.linewidth': 5, 'lines.markersize': 5, "axes.labelsize":15}, style="whitegrid")

# Get the best team
win = df.Club[df['Position'] == 1].value_counts()
sns.barplot(x=win, y=win.index, data=df)
# get all club on position 20 every season
lose = df.Club[df['Position'] == 20].value_counts()
sns.barplot(x=lose, y=lose.index, data=df)
# Get the top 5 most UCL participant from premier league
UCL = df.Club[df['Position'].between(1,4)].value_counts().nlargest(5)
sns.barplot(y=UCL, x=UCL.index, data=df)
# count all teams every season
Part = df.Club.value_counts()
sns.barplot(x=Part, y=Part.index, data=df)
df[df['Lost'] == 0]
point = df.groupby('Club')[['Points']].sum().sort_values('Points', ascending=False)
point
win = df.groupby('Club')[['Won']].sum().sort_values('Won', ascending=False)
played = df.groupby('Club')[['Played']].sum().sort_values('Played', ascending=False)

results = pd.merge(win, played, on=['Club'])
win_rate = []

for x in range(len(results.index)):
    result = results.Won[x] / results.Played[x]
    win_rate.append(result)
    
results['win_rate'] = win_rate
results
# First we need to make time column, in this case i put 4 season in one row
dates = []

c = df['season'].unique()
ser = [c[x:x+4] for x in range(0, len(c), 4)]

for x in range(len(ser)):
    s = ' '.join(ser[x])
    f1 = s[:4]
    f2 = s[-4:]
    f3 = str(f1) + '-' + str(f2)
    dates.append(f3)

# Make a different Dataframe and put time data into it
performance = pd.DataFrame(dates, columns=['Time'])
performance
MU = []

win = df[(df['Position'] == 1) & (df['Club'] == 'Manchester United')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    MU.append(wins)
    
performance['Manchester United'] = MU
sns.lineplot(x='Time', y='Manchester United', data=performance)
Chelsea = []

win = df[(df['Position'] == 1) & (df['Club'] == 'Chelsea')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    Chelsea.append(wins)
    
performance['Chelsea'] = Chelsea
sns.lineplot(x='Time', y='Chelsea', data=performance)
MC = []

win = df[(df['Position'] == 1) & (df['Club'] == 'Manchester City')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    MC.append(wins)
    
performance['Manchester City'] = MC
sns.lineplot(x='Time', y='Manchester City', data=performance)
ARS = []

win = df[(df['Position'] == 1) & (df['Club'] == 'Arsenal')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    ARS.append(wins)
    
performance['Arsenal'] = ARS
sns.lineplot(x='Time', y='Arsenal', data=performance)
df_man = pd.read_csv('../input/premier-league-standing-all-season-19922020/PL Manager All Season (1992-2020).csv')
df_man.head()
df_man.info()
# merge dataset
complete = pd.merge(df_man, df, on=['season', 'Club'])
complete.head()
# get managers name on standing position #1
best_man = complete.Name[complete['Position'] == 1].value_counts()
sns.barplot(y=best_man.index, x=best_man, data=complete)
longst_career = complete.Name.value_counts().head(8)
sns.barplot(y=longst_career.index, x=longst_career, data=complete)
AF = []

win = complete[(complete['Position'] == 1) & (complete['Name'] == 'Alex Ferguson')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    AF.append(wins)
    
performance['Alex Ferguson'] = AF
sns.lineplot(x='Time', y='Alex Ferguson', data=performance)
complete[(complete['Name'] == 'Alex Ferguson') & (complete['Position'] == 1)]
complete[(complete['Club'] == 'Manchester United') & (complete['Position'] == 1)]
All = performance[['Time', 'Manchester United', 'Alex Ferguson']].melt('Time', var_name='cols',  value_name='vals')
sns.lineplot(x="Time", y="vals", hue='cols', data=All)
JM = []

win = complete[(complete['Position'] == 1) & (complete['Name'] == 'José Mourinho')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    JM.append(wins)
    
performance['José Mourinho'] = JM
sns.lineplot(x='Time', y='José Mourinho', data=performance)
complete[(complete['Name'] == 'José Mourinho') & (complete['Position'] == 1)]
complete[(complete['Club'] == 'Chelsea') & (complete['Position'] == 1)]
All = performance[['Time', 'Chelsea', 'José Mourinho']].melt('Time', var_name='cols',  value_name='vals')
sns.lineplot(x="Time", y="vals", hue='cols', data=All)
AW = []

win = complete[(complete['Position'] == 1) & (complete['Name'] == 'Arsène Wenger')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    AW.append(wins)
    
performance['Arsène Wenger'] = AW
sns.lineplot(x='Time', y='Arsène Wenger', data=performance)
complete[(complete['Name'] == 'Arsène Wenger') & (complete['Position'] == 1)]
complete[(complete['Club'] == 'Arsenal') & (complete['Position'] == 1)]
All = performance[['Time', 'Arsenal', 'Arsène Wenger']].melt('Time', var_name='cols',  value_name='vals')
sns.lineplot(x="Time", y="vals", hue='cols', data=All)
JG = []

win = complete[(complete['Position'] == 1) & (complete['Name'] == 'Josep Guardiola')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    JG.append(wins)
    
performance['Josep Guardiola'] = JG
sns.lineplot(x='Time', y='Josep Guardiola', data=performance)
complete[(complete['Name'] == 'Josep Guardiola') & (complete['Position'] == 1)]
complete[(complete['Club'] == 'Manchester City') & (complete['Position'] == 1)]
All = performance[['Time', 'Manchester City', 'Josep Guardiola']].melt('Time', var_name='cols',  value_name='vals')
sns.lineplot(x="Time", y="vals", hue='cols', data=All)