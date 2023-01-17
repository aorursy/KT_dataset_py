import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.utils.multiclass import unique_labels



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
matches = pd.read_csv('../input/matches.csv')

matches['match_id'] = matches['id']

matches = matches.drop('id', axis=1)

matches.info()
deli = pd.read_csv('../input/deliveries.csv')

deli.info()
df = deli.merge(matches, on='match_id', how='left')

df.info()
df.head()
teams = set(df.team1.unique()).union(set(df.team2.unique()))

players = set(df.batsman.unique()).union(set(df.bowler.unique())).union(set(df.non_striker.unique()))

players = list(players)

print('{} unique teams, {} unique players'.format(len(teams), len(players)))
season_data = [df.loc[df.season == s].copy() for s in df.season.unique()]



vectormap = dict()

for season in season_data:

    team_batsmen = season.groupby(['batting_team'])['batsman'].unique()

    season_name = season.season.unique()[0]

    for team, batsmen in team_batsmen.iteritems():

        vector = [i in batsmen for i in players]

        vectormap[season_name, team] = vector

        

def get_vec(row):

    s, t = row

    global vectormap

    return vectormap.get((s, t))



df['batt_vec'] = [get_vec(i) for i in df[['season', 'batting_team']].values]

df['bowl_vec'] = [get_vec(i) for i in df[['season', 'bowling_team']].values]

df['matchupvector'] = df.batt_vec + df.bowl_vec

df['team1won'] = df.winner == df.team1



est = RandomForestClassifier(n_jobs=-1, n_estimators=30)

# Take only one match instance as the vectors are mostly similar for each team

data = pd.concat([df.loc[df.match_id == i].head(1) for i in df.match_id.unique()])

X, y = np.array(list(data.matchupvector.values)), data.team1won

scores = cross_val_score(est, X, y, cv=10, scoring='roc_auc', verbose=5)



print(scores.mean()) #### 0.499825372208
player_encoder = LabelEncoder()

venue_encoder = LabelEncoder()

ump_encoder = LabelEncoder()



player_encoder.fit(players)

venue_encoder.fit(df.venue)

ump_encoder.fit(list(set(df.umpire1).union(set(df.umpire2))))



df['u1_e'] = ump_encoder.transform(df.umpire1)

df['u2_e'] = ump_encoder.transform(df.umpire2)

df['venue_e'] = venue_encoder.transform(df.venue)



df['batsman_e'] = player_encoder.transform(df.batsman)

df['non_striker_e'] = player_encoder.transform(df.non_striker)

df['bowler_e'] = player_encoder.transform(df.bowler)



y = df.total_runs * ((~df.player_dismissed.isnull()).map({True: -1, False: 1}))

df['target_ball_by_ball'] = y

df['will_be_out'] = y < 0
plt.subplots(figsize=(10, 5))

plt.subplot(121)

sns.countplot(df.target_ball_by_ball)

plt.title('-ve runs signify a player was dismissed')

plt.subplot(122)

sns.countplot(df.will_be_out)

plt.title('Will be out')
est = RandomForestClassifier(n_jobs=-1, n_estimators=100, class_weight='balanced')

X = df[['batsman_e', 'non_striker_e', 'bowler_e', 'over', 'ball', 'season',

        'inning']]

cross_val_score(est, X, df.will_be_out, cv=10, scoring='roc_auc').mean()
df['d_kind'] = df.dismissal_kind.fillna('not_out')

X = df.loc[df.will_be_out == True][['batsman_e', 'non_striker_e', 'bowler_e', 'over', 'ball', 'season',

        'inning']]

y = df.loc[df.will_be_out == True]['d_kind']

est = RandomForestClassifier(n_jobs=-1, n_estimators=100, class_weight='balanced')

d_kind_preds = cross_val_predict(est, X, y, cv=10, verbose=5)
c_m = confusion_matrix(y, d_kind_preds)

labels = unique_labels(y)

c_m_d = pd.DataFrame(c_m, columns=labels)

c_m_d.index = labels

sns.heatmap(c_m_d, annot=True)