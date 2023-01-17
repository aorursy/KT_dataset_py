import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(18,9), 'lines.linewidth': 5, 'lines.markersize': 5, "axes.labelsize":15}, style="whitegrid")
df = pd.read_csv('../input/fa-cup-final-all-season-1872-2020/FA emirates Cup Final.csv')
df.head()
df.info()
win_team = df['Winners'].value_counts().nlargest(8)
sns.barplot(x=win_team, y=win_team.index, data=df)
ru_team = df['Runners-up'].value_counts().nlargest(8)
sns.barplot(x=ru_team, y=ru_team.index, data=df)
app = pd.concat([df['Winners'],df['Runners-up']])
app_team = app.value_counts().nlargest(8)
sns.barplot(x=app_team, y=app_team.index, data=df)
mhwin = df['Winners'].value_counts() / app.value_counts()
mhwin_team = mhwin.fillna(0).sort_values(ascending=False).nlargest(25)
sns.barplot(x=mhwin_team, y=mhwin_team.index, data=df)
mlwin_team = mhwin.fillna(0).sort_values().head(25)
sns.barplot(x=mlwin_team, y=mlwin_team.index, data=df)
# Here we are going to perform feature engineering to create another feature that makes us easier to understand
# First we are going to create a new column call diff, this column contain only the difference of final score
# on every match

diff = []


for x in df.Score:
    result = int(x[0]) - int(x[2])
    diff.append(result)
    
df['diff'] = diff

# Now we want to make a column call match where its contain the year, winning team and the runner up.
# Remember what type Year columns is, it is an integer, we need to convert it to string before we can proceed.

date = []

for y in df.Year:
    result = str(y)
    date.append(result)
    
df['date'] = date

# Now we need all the component we want, lets create the match column.

df['match'] = df['date'] + ' ' + df['Winners'] + ' vs ' + df['Runners-up']

# lets take a quick look on our new dataset

big_win = df[['match','diff']].sort_values('diff', ascending=False).head(8)
big_win
sns.barplot(x='diff', y='match', data=big_win)
# Here we are going to need a new dataframe called performance, which contain time (year in decade), 
# and best team performance (how many times they won every decade)

df_dict = {n: df.iloc[n:n+10, :] 
           for n in range(0, len(df), 10)}

dates = []

set = list(df_dict)
for x in set:
    ds = df_dict[x].Year[x]
    da = ds - 9
    dt = str(da) + '-' + str(ds)
    dates.append(dt)

performance = pd.DataFrame(dates, columns=['Time'])
performance.head()
arsFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Arsenal'].count()
    arsFC.append(result)

performance['Arsenal'] = arsFC
sns.lineplot(x='Time', y='Arsenal', data=performance)
MUFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Manchester United'].count()
    MUFC.append(result)

performance['Manchester United'] = MUFC
sns.lineplot(x='Time', y='Manchester United', data=performance)
CheFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Chelsea'].count()
    CheFC.append(result)

performance['Chelsea'] = CheFC
sns.lineplot(x='Time', y='Chelsea', data=performance)
totFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Tottenham Hotspur'].count()
    totFC.append(result)

performance['Tottenham Hotspur'] = totFC
sns.lineplot(x='Time', y='Tottenham Hotspur', data=performance)
AVFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Aston Villa'].count()
    AVFC.append(result)

performance['Aston Villa'] = AVFC
sns.lineplot(x='Time', y='Aston Villa', data=performance)
# lets take a look at our new dataset

performance.head()
# We are going to produce line chart contain all 5 performance

All = performance.melt('Time', var_name='cols',  value_name='vals')
sns.lineplot(x="Time", y="vals", hue='cols', data=All)