import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from math import pi
data = pd.read_csv('../input/efootball-pes-2020-all-players-csv/deets-updated.csv')
data.head()
for col in data.columns:

    if data[col].isnull().sum()>0:

        print(col,data[col].isnull().sum())
data.shape
data.describe()
data.rename(columns={'registered_position':'position','team_name':'club'},inplace=True)
#deleting duplicate rows



data.drop_duplicates(inplace=True)
col = ['name', 'club', 'league', 'nationality', 'region','height','weight','age','foot',

       'position','offensive_awareness','ball_control','dribbling','tight_possession','low_pass','lofted_pass',

      'finishing','heading','place_kicking','curl','speed','acceleration','kicking_power','jump','physical_contact','balance',

      'stamina','defensive_awareness','ball_winning','aggression','gk_awareness','gk_catching','gk_clearing','gk_reflexes',

      'gk_reach','weak_foot_usage','weak_foot_accuracy','form','injury_resistance','overall_rating','playing_style',

       'rating_stars']
len(col)
r_data = data[col]

r_data.head()
def club_name(name):

    return r_data[r_data['club']==name][['name','nationality','position','foot','age',

                                             'overall_rating','rating_stars']].sort_values('overall_rating',ascending=False)

club_name('MANCHESTER UNITED')

for col in r_data.columns:

    if r_data[col].isnull().sum()>0:

        print(col,r_data[col].isnull().sum())
print(r_data.columns)
def technique(r_data):

    return int(round((r_data[['ball_control','dribbling','heading','curl','balance','tight_possession']].mean()).mean()))



def passing(r_data):

    return int(round((r_data[['lofted_pass','low_pass']].mean()).mean()))



def mobility(r_data):

    return int(round((r_data[['speed','acceleration','stamina']].mean()).mean()))





def attacking(r_data):

    return int(round((r_data[['offensive_awareness','finishing','kicking_power']].mean()).mean()))



def defence(r_data):

    return int(round((r_data[['jump','physical_contact','defensive_awareness']].mean()).mean()))



def goal_kepping(r_data):

    return int(round((r_data[['gk_awareness','gk_catching','gk_clearing','gk_reflexes','gk_reach','place_kicking']].mean()).mean()))



def weak_foot(r_data):

    return int(round((r_data[['weak_foot_usage','weak_foot_accuracy']].mean()).mean()))

r_data['technique'] = r_data.apply(technique,axis=1)

r_data['passing'] = r_data.apply(passing,axis=1)

r_data['mobility'] = r_data.apply(mobility,axis=1)

r_data['attacking'] = r_data.apply(attacking,axis=1)

r_data['defence'] = r_data.apply(defence,axis=1)

r_data['goal_kepping'] = r_data.apply(goal_kepping,axis=1)

r_data['weak_foot'] = r_data.apply(weak_foot,axis=1)

player = r_data[['name', 'club', 'league', 'nationality', 'region','age','foot',

       'position','technique','passing','mobility','attacking','defence','goal_kepping','weak_foot',

        'form','injury_resistance','overall_rating','playing_style','rating_stars']]
player.head()


plt.rcParams['figure.figsize']=(10,5)

sns.countplot(x='region',hue='foot',data = r_data,palette='deep')

plt.title('Footballers from different region and their preferred foot in Pes2020',fontsize=20)

plt.show()



plt.rcParams['figure.figsize']=(20,10)

plt.style.use('fivethirtyeight')

ax=sns.relplot(x='rating_stars',y='form',data=r_data,kind='line')



plt.show()
plt.rcParams['figure.figsize']=(18,10)

plt.style.use('seaborn')



ax = sns.countplot(x='position',data=r_data,palette='GnBu_d')

ax.set_xlabel(xlabel='position')

ax.set_ylabel(ylabel='count of position')

ax.set_title('Count of players in different position',fontsize=20)

plt.show()
labels = data['weak_foot_usage'].unique()

size = data['weak_foot_usage'].value_counts()



colors = plt.cm.Wistia(np.linspace(0, 1, 5))

explode = [0, 0, 0.1,0]



plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)

plt.title('Distribution of Week Foot usage among Players', fontsize = 20)

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (15,8)

plt.style.use('seaborn-pastel')

#sns.set(style = "dark", palette = "deep", color_codes = True)

ax = sns.distplot(r_data['age'],bins=52,kde=False)

ax.set_xlabel(xlabel='Age')

ax.set_ylabel(ylabel='No. of players')

ax.set_title('Count of players in different Age',fontsize=20)

plt.show()
plt.rcParams['figure.figsize'] = (15,8)

plt.style.use('dark_background')

ax = sns.distplot(r_data['overall_rating'],bins=52,kde=False)

ax.set_xlabel(xlabel='overall_rating')

ax.set_ylabel(ylabel='No. of players')

ax.set_title('Count of players in different overall-rating',fontsize=20)

plt.show()
forward_pos = ['LWF','RWF','CF','SS']

total_for_players =r_data[r_data['position'].isin(forward_pos)].shape[0]

print('Total Forward Players in Pes 2020:',total_for_players)
best_for_ply = r_data[r_data['position'].isin(forward_pos)].sort_values(

    by='overall_rating',ascending=False).head(10)

best_for_ply[['name','club','nationality','position','foot','age','overall_rating']].reset_index(drop =True)
idx = 1

plt.rcParams['figure.figsize'] = (15,45)

plt.style.use('seaborn-white')

ax = plt.subplot(121)



plt.plot()



ax = plt.subplot(121,polar=True)

categories = ['technique','passing','mobility','attacking','defence']

for name, features in best_for_ply.groupby(best_for_ply['name'])[categories].mean().iterrows():



    N = len(categories)

    features = dict(features) 

    values = list(features.values())

    values += values[:1]



    angles = [n/float(N)*2*pi for n in range(N)]

    angles += angles[:1]

    

    ax = plt.subplot(9, 3, idx, polar=True)



    plt.xticks(angles[:-1], categories, color='black', size=8)



    ax.set_rlabel_position(0)

    plt.yticks([50,70,90], ['50','70','90'], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(name, size=11, y=1.1)

    

    idx += 1 







    
midfield_pos = ['CMF','AMF','RMF','LMF','DMF']

total_for_players =r_data[r_data['position'].isin(midfield_pos)].shape[0]

print('Total Midfield Players in Pes 2020:',total_for_players)
best_mid_ply = r_data[r_data['position'].isin(midfield_pos)].sort_values(

    by='overall_rating',ascending=False).head(10)

best_mid_ply[

    ['name','club','nationality','position','foot','age','overall_rating']].reset_index(drop =True)
idx = 1

plt.rcParams['figure.figsize'] = (15,45)

plt.style.use('seaborn-white')

ax = plt.subplot(121)



plt.plot()



ax = plt.subplot(121,polar=True)

categories = ['technique','passing','mobility','attacking','defence']

for name, features in best_mid_ply.groupby(best_mid_ply['name'])[categories].mean().iterrows():



    N = len(categories)

    features = dict(features) 

    values = list(features.values())

    values += values[:1]



    angles = [n/float(N)*2*pi for n in range(N)]

    angles += angles[:1]

    

    ax = plt.subplot(9, 3, idx, polar=True)



    plt.xticks(angles[:-1], categories, color='grey', size=8)



    ax.set_rlabel_position(0)

    plt.yticks([50,70,90], ['50','70','90'], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(name, size=11, y=1.1)

    

    idx += 1 

defensive_pos = ['RB','LB','CB']

total_for_players =r_data[r_data['position'].isin(defensive_pos)].shape[0]

print('Total Defense Players in Pes 2020:',total_for_players)
best_def_ply = r_data[r_data['position'].isin(defensive_pos)].sort_values(

    by='overall_rating',ascending=False).head(10)

best_def_ply[

    ['name','club','nationality','position','foot','age','overall_rating']].reset_index(drop =True)
idx = 1

plt.rcParams['figure.figsize'] = (15,45)

plt.style.use('seaborn-white')

ax = plt.subplot(121)



plt.plot()



ax = plt.subplot(121,polar=True)

categories = ['technique','passing','mobility','attacking','defence']

for name, features in best_def_ply.groupby(best_def_ply['name'])[categories].mean().iterrows():



    N = len(categories)

    features = dict(features) 

    values = list(features.values())

    values += values[:1]



    angles = [n/float(N)*2*pi for n in range(N)]

    angles += angles[:1]

    

    ax = plt.subplot(9, 3, idx, polar=True)



    plt.xticks(angles[:-1], categories, color='grey', size=8)



    ax.set_rlabel_position(0)

    plt.yticks([50,70,90], ['50','70','90'], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(name, size=11, y=1.1)

    

    idx += 1 

goalkeeping_pos = ['GK']

total_for_players =data[data['position'].isin(goalkeeping_pos)].shape[0]

print('Total Goalkepping Players in Pes 2020:',total_for_players)
best_gk_ply = data[data['position'].isin(goalkeeping_pos)].sort_values(

    by='overall_rating',ascending=False).head(10)

best_gk_ply[

    ['name','club','nationality','position','foot','age','overall_rating']].reset_index(drop =True)
idx = 1

plt.rcParams['figure.figsize'] = (15,45)

plt.style.use('seaborn-white')

ax = plt.subplot(121)



plt.plot()



ax = plt.subplot(121,polar=True)

categories = ['gk_awareness','gk_catching','gk_clearing','gk_reflexes','gk_reach','place_kicking']

for name, features in best_gk_ply.groupby(best_gk_ply['name'])[categories].mean().iterrows():



    N = len(categories)

    features = dict(features) 

    values = list(features.values())

    values += values[:1]



    angles = [n/float(N)*2*pi for n in range(N)]

    angles += angles[:1]

    

    ax = plt.subplot(9, 3, idx, polar=True)



    plt.xticks(angles[:-1], categories, color='grey', size=8)



    ax.set_rlabel_position(0)

    plt.yticks([50,70,90], ['50','70','90'], color="black", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(name, size=11, y=1.1)

    

    idx += 1 

r_data.sort_values(['overall_rating'], ascending=[False]).groupby(

    ['age'])[['position','name','club','nationality','foot','overall_rating']].first()

    

r_data.sort_values(['overall_rating'],ascending=[False]).groupby(

    ['position'])[['name','club','nationality','foot','age','overall_rating']].first().reset_index()

    
best_in_league = r_data[(r_data.league!='Free Agents') & (r_data.league!='Free Agent')].sort_values(['overall_rating'],ascending=[False]).groupby(

    ['league'])[['position','name','club','nationality','foot','age','overall_rating']].first()

best_in_league.reset_index()
best_in_nation = r_data.sort_values(['overall_rating'],ascending=[False]).groupby(

    ['nationality'])[['position','name','club','league','foot','age','overall_rating']].first().sort_values(['overall_rating'],ascending=[False])

best_in_nation.reset_index()
y_team = r_data.groupby(['club'])['age'].agg(average_age=('age', 'mean')).round(2).sort_values(

    by='average_age').reset_index().head(10)

print(y_team)

o_team = r_data.groupby(['club'])['age'].agg(average_age=('age', 'mean')).round(2).sort_values(

    by='average_age',ascending=False).reset_index().head(10)

print(o_team)

len(r_data.club.unique())
b_team = r_data[(r_data.league!='Free Agents') & (r_data.league!='Free Agent')].sort_values(by='overall_rating',ascending=False).groupby('club')['overall_rating'].nlargest(

    21).mean(level=0).sort_values(ascending=False).reset_index()['club']

    

b_team.head(10)
club_data = r_data.loc[r_data['club'].isin(list(b_team.head(10)))]

plt.rcParams['figure.figsize'] = (20,8)

plt.style.use('seaborn-bright')

ax = sns.boxplot(x="club", y="overall_rating", data=club_data)

ax.set_xlabel(xlabel='Club')

ax.set_ylabel(ylabel='Overall rating')

ax.set_title('Distribution of overall rating in major club',fontsize=20)

plt.xticks(rotation=90)

plt.show()
club_data = r_data.loc[r_data['club'].isin(list(b_team.head(10)))]

plt.rcParams['figure.figsize'] = (15,8)

plt.style.use('seaborn-dark')

ax = sns.boxenplot(x="club", y="age", data=club_data)

ax.set_xlabel(xlabel='Club')

ax.set_ylabel(ylabel='Overall rating')

ax.set_title('Distribution of Age in major club',fontsize=20)

plt.xticks(rotation = 90)

plt.show()