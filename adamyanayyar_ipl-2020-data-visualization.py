#Libraries
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
csk = pd.read_csv('../input/ipl-2020-players-dataset/IPL 2020 - CSK.csv')
rr = pd.read_csv('../input/ipl-2020-players-dataset/IPL 2020 - RR.csv')
rcb = pd.read_csv('../input/ipl-2020-players-dataset/IPL 2020 - RCB.csv')
mi = pd.read_csv('../input/ipl-2020-players-dataset/IPL 2020 - MI.csv')
kxip = pd.read_csv('../input/ipl-2020-players-dataset/IPL 2020 - KXIP.csv')
kkr = pd.read_csv('../input/ipl-2020-players-dataset/IPL 2020 - KKR.csv')
dc = pd.read_csv('../input/ipl-2020-players-dataset/IPL 2020 - DC.csv')
srh = pd.read_csv('../input/ipl-2020-players-dataset/IPL 2020 - SRH.csv')
rcb.head(3)
data = pd.concat([csk,rr,rcb,mi,kxip,kkr,dc,srh], ignore_index=True)
data.sample(5)
data.sample(15)
#shape of dataframe
data.shape
data.Nationality.value_counts()
# For our analysis, let's remove the players with no previous data.
nostats = data['Matches'].isin(['No Past IPL Stats Found'])
data = data[~nostats]
data.shape
# data types
data.dtypes
def clean_runs(x):
    if isinstance(x,str):
        return (x.replace('*',''))
    return (x)
def clean_balls(x):
    if isinstance(x,str):
        return (x.replace(',',''))
    return (x)
def replace_dash(x):
    if isinstance(x,str):
        return (x.replace('-','0'))
    return (x)
data['Matches'] = data.Matches.astype(int)
data['Bat-Highest Score'] = data['Bat-Highest Score'].apply(clean_runs).astype(int)
data['Bat- Balls Faced'] = data['Bat- Balls Faced'].apply(clean_balls).astype(int)
data['Bat- Average'] = data['Bat- Average'].apply(replace_dash).astype(float)
data['Bat-Strike Rate'] = data['Bat-Strike Rate'].apply(replace_dash).astype(float)
data['Bowl-Balls'] = data['Bowl-Balls'].apply(clean_balls).astype(int)
data['Bowl-Runs Conceded'] = data['Bowl-Runs Conceded'].apply(clean_balls).astype(int)
data['Bowl-Average'] = data['Bowl-Average'].apply(replace_dash).astype(float).round(2)
data['Bowl-Economy'] = data['Bowl-Economy'].apply(replace_dash).astype(float).round(2)
data['Bowl-Strike Rate'] = data['Bowl-Strike Rate'].apply(replace_dash).astype(float).round(2)
data.dtypes
def Most_runs_over():
    print('Players who scored the most runs: ') 
    return data[['Name','Team','Matches','Bat-Runs','Bat- Average']].sort_values(by='Bat-Runs',ascending=False).reset_index(drop=True).head(5)

def Best_average_over():
    print('Players with best batting average: ') 
    return data[['Name','Team','Matches','Bat-Runs','Bat- Average']].sort_values(by='Bat- Average',ascending=False).reset_index(drop=True).head(5)

def Best_strikerate_over():
    data_team = data[data['Bat-Runs'] >= 500]
    print('Players with best batting strike rate(scored 500 runs or more): ') 
    return data_team[['Name','Team','Matches','Bat-Runs','Bat-Strike Rate']].sort_values(by='Bat-Strike Rate',ascending=False).reset_index(drop=True).head(5)

def Most_Notouts_over():
    print('Players with most not outs:')
    return data[['Name','Team','Matches','Bat-Runs','Bat- Not Out']].sort_values(by='Bat- Not Out',ascending=False).reset_index(drop=True).head(5)

def Most_catches_over():
    print('Players with most catches:')
    return data[['Name','Team','Matches','Catches']].sort_values(by='Catches',ascending=False).reset_index(drop=True).head(5)

def Most_centuries_over():
    print('Players with most centuries:')
    return data[['Name','Team','Matches','Bat-Runs','100']].sort_values(by='100',ascending=False).reset_index(drop=True).head(5)

def Most_6s():
    print('Players with most 6s:')
    return data[['Name','Team','Matches','Bat-Runs','6s']].sort_values(by='6s',ascending=False).reset_index(drop=True).head(5)

def Most_4s():
    print('Players with most 4s:')
    return data[['Name','Team','Matches','Bat-Runs','4s']].sort_values(by='4s',ascending=False).reset_index(drop=True).head(5)

def Most_wickets_over():
    print('Players with most wickets:')
    return data[['Name','Team','Matches','Bowl-Wickets','Bowl-Best Figures']].sort_values(by='Bowl-Wickets',ascending=False).reset_index(drop=True).head(5)

def Best_bowl_economy_over():
    data_team = data[data['Bowl-Wickets'] >= 10]
    print('Players with best bowling economy(10 wickets and above):')
    return data_team[['Name','Team','Matches','Bowl-Wickets','Bowl-Economy']].sort_values(by='Bowl-Economy').reset_index(drop=True).head(5)

def Best_bowl_average_over():
    data_team = data[data['Bowl-Wickets'] >= 10]
    print('Players with best bowling average(10 wickets and above):')
    return data_team[['Name','Team','Matches','Bowl-Wickets','Bowl-Average']].sort_values(by='Bowl-Average').reset_index(drop=True).head(5)

def four_wickets_over():
    print('Players with 4 wicket hauls:')
    return data[['Name','Team','Matches','Bowl-Wickets','Bowl-Best Figures','4W','5W']].sort_values(by='4W',ascending=False).reset_index(drop=True).head(5)

Most_runs_over()
Best_average_over()
Best_strikerate_over()
Most_Notouts_over()
Most_catches_over()
Most_centuries_over()
Most_6s()
Most_4s()
Most_wickets_over()
Best_bowl_economy_over()
Best_bowl_average_over()
four_wickets_over()
def Most_runs(x):
    data_team = data[data.Team == x]
    print('Players with most runs in ',x) 
    return data_team[['Name','Matches','Bat-Runs','Bat- Average']].sort_values(by='Bat-Runs',ascending=False).reset_index(drop=True).head(3)

def Best_strikerate(x):
    data_team = data[data.Team == x]
    data_team = data_team[data_team['Bat-Runs'] >= 1000]
    print('Players with best strike rate(1000 runs and above)in ',x) 
    return data_team[['Name','Matches','Bat-Runs','Bat-Strike Rate']].sort_values(by='Bat-Strike Rate',ascending=False).reset_index(drop=True).head(3)

def Most_Notouts(x):
    data_team = data[data.Team == x]
    print('Players with most not outs in', x)
    return data_team[['Name','Matches','Bat-Runs','Bat- Not Out']].sort_values(by='Bat- Not Out',ascending=False).reset_index(drop=True).head(3)

def Most_catches(x):
    data_team = data[data.Team == x]
    print('Players with most catches in', x)
    return data_team[['Name','Matches','Catches']].sort_values(by='Catches',ascending=False).reset_index(drop=True).head(3)

def Most_centuries(x):
    data_team = data[data.Team == x]
    print('Players with most centuries in', x)
    return data_team[['Name','Matches','Bat-Runs','100']].sort_values(by='100',ascending=False).reset_index(drop=True).head(2)

def Most_wickets(x):
    data_team = data[data.Team == x]
    print('Players with most wickets in', x)
    return data_team[['Name','Matches','Bowl-Wickets','Bowl-Best Figures']].sort_values(by='Bowl-Wickets',ascending=False).reset_index(drop=True).head(3)

def Best_bowl_economy(x):
    data_team = data[data.Team == x]
    data_team = data_team[data_team['Bowl-Wickets'] >= 10]
    print('Players with Best bowling economy(10 wickets and above) in', x)
    return data_team[['Name','Matches','Bowl-Wickets','Bowl-Economy']].sort_values(by='Bowl-Economy').reset_index(drop=True).head(3)

def Best_bowl_average(x):
    data_team = data[data.Team == x]
    data_team = data_team[data_team['Bowl-Wickets'] >= 10]
    print('Players with Best bowling average(10 wickets and above) in', x)
    return data_team[['Name','Matches','Bowl-Wickets','Bowl-Average']].sort_values(by='Bowl-Average').reset_index(drop=True).head(3)

def four_wickets(x):
    data_team = data[data.Team == x]
    data_team = data_team[data_team['Bowl-Wickets'] >= 10]
    print('Players with most 4 wicket hauls in', x)
    return data_team[['Name','Matches','Bowl-Wickets','Bowl-Best Figures','4W','5W']].sort_values(by='4W',ascending=False).reset_index(drop=True).head(3)

Most_runs('RCB')
Best_strikerate('KXIP')
Most_Notouts('CSK')
Most_centuries('KXIP')
Most_catches('DC')
Most_wickets('RR')
Best_bowl_economy('SRH')
Best_bowl_average('KKR')
four_wickets('DC')
plt.figure(figsize=(10,7))
sns.countplot(x='Team', hue='Nationality', data=data)
plt.title('Different Nationalities of Players in each team')
plt.figure(figsize=(10,7))
#data['Runs500'] = data[data['Bat-Runs'] >= 500]
sns.stripplot(x='Team', y='Bat-Runs', data=data,linewidth=1)
plt.title('Runs of players in each team')
dummies = pd.get_dummies(data['Nationality']).rename(columns=lambda x: 'Nationality_' + str(x))
df = pd.concat([data, dummies], axis=1)
df.drop(['Nationality'], inplace=True, axis=1)
df.dropna(inplace=True)
df.sample(5) #Created dummies for nationality 
plt.figure(figsize=(6,5))
sns.set()
ax = sns.countplot('Nationality_Indian', data=df,palette='bone')
ax.set_xlabel(xlabel = 'Nationality Indian', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Nationality of Players', fontsize = 20)
plt.show()
plt.figure(figsize=(10,7))
sns.set()
data_team = df[df['Bat-Runs'] >= 500]
ax = sns.violinplot(x='Team', y='Bat-Runs', hue='Nationality_Indian', data=data_team, palette='Set3',cut=0)
ax.set_xlabel('Teams', fontsize=14)
ax.set_ylabel('Runs scored', fontsize=14)
ax.set_title(label = 'Violin plot of Runs scored by players(500 runs and above)',fontsize=14)
plt.show()
plt.figure(figsize=(10,7))
sns.set()
data_team = df[df['Bat-Runs'] >= 500]
ax = sns.boxplot(x='Team', y='Bat-Strike Rate', hue='Nationality_Indian', data=data_team,palette='Set3')
ax.set_xlabel('Teams', fontsize=14)
ax.set_ylabel('Batting Strike Rate', fontsize=14)
ax.set_title(label = 'Distribution of Strike Rate of players(500 runs and above)',fontsize=14)
plt.show()
plt.figure(figsize=(10,7))
sns.set()
data_team = df[df['Bat-Runs'] >= 1000]
ax = sns.boxplot(y="Bat- Average", x="Team", hue='Nationality_Indian', data=data_team,palette='Set3')
ax.set_xlabel('Teams', fontsize=14)
ax.set_ylabel('Batting Average', fontsize=14)
ax.set_title(label = 'Distribution of Average Runs by players(1000 runs and above)',fontsize=14)
plt.show()
sns.set(style ="dark", palette="Set3")
plt.figure(figsize=(8,6))
plt.style.use('ggplot')
data_team = df[df['50'] >= 1]
ax = sns.distplot(data_team['50'],bins=10,kde=False)
ax.set_xlabel(xlabel = 'Number of Half-centuries', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Number of Half-centuries by players(1 and above)', fontsize = 20)
plt.show()
plt.figure(figsize=(10,7))
data_team = df[df['Bowl-Wickets'] >= 10]
ax = sns.violinplot(x='Team', y='Bowl-Wickets', hue='Nationality_Indian', data=data_team,cut=0)
ax.set_xlabel('Teams', fontsize=14)
ax.set_ylabel('Wickets Taken', fontsize=14)
ax.set_title(label = 'Violin plot of Wickets taken by players(10 and above)',fontsize=14)
plt.show()
plt.figure(figsize=(10,7))
data_team = df[df['Bowl-Average'] < 50]
dt = data_team[data_team['Bowl-Average'] > 0]
ax = sns.boxplot(x='Team', y='Bowl-Average', hue='Nationality_Indian', data=dt)
ax.set_xlabel('Teams', fontsize=14)
ax.set_ylabel('Bowling Average', fontsize=14)
ax.set_title(label = 'Distribution of Bowling Average of players',fontsize=14)
plt.show()
plt.figure(figsize=(10,7))
dt = df[df['Bowl-Economy'] > 0]
ax = sns.boxplot(x='Team', y='Bowl-Economy', hue='Nationality_Indian', data=dt)
ax.set_xlabel('Teams', fontsize=14)
ax.set_ylabel('Bowling Economy', fontsize=14)
ax.set_title(label = 'Distribution of Bowling Economy of players',fontsize=14)
plt.show()
plt.figure(figsize=(8,6))
plt.style.use('ggplot')
dt = df[df['Bat-Runs'] >= 500]
dtt = dt[dt['Bowl-Wickets'] > 10]
ax = sns.countplot(x='Team', hue='Nationality_Indian', data=dtt)
ax.set_xlabel(xlabel = 'Teams', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Number of All-Rounders in each team', fontsize = 20)
plt.show()