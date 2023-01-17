 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_dir = '../input/'

df_seeds = pd.read_csv(data_dir + 'TourneySeeds.csv')

df_tour = pd.read_csv(data_dir + 'TourneyCompactResults.csv')
df_seeds.head()
df_tour.head()
df_tour.drop(labels=['Daynum', 'Wscore', 'Lscore', 'Wloc', 'Numot'], inplace=True, axis=1)
def seed_to_int(seed):

    """Get just the digits from the seeding. Return as int"""

    s_int = int(seed[1:3])

    return s_int

df_seeds['n_seed'] = df_seeds.Seed.apply(seed_to_int)

df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_winseeds = df_seeds.rename(columns={'Team':'Wteam', 'n_seed':'win_seed'})

df_lossseeds = df_seeds.rename(columns={'Team':'Lteam', 'n_seed':'loss_seed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'Wteam'])

df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'Lteam'])

df_concat['seed_diff'] = df_concat.win_seed - df_concat.loss_seed
df_wins = pd.DataFrame()

df_wins['seed_diff'] = df_concat['seed_diff']

df_wins['result'] = 1



df_losses = pd.DataFrame()

df_losses['seed_diff'] = -df_concat['seed_diff']

df_losses['result'] = 0



df_for_predictions = pd.concat((df_wins, df_losses))
X_train = df_for_predictions.seed_diff.values.reshape(-1,1)

y_train = df_for_predictions.result.values

X_train, y_train = shuffle(X_train, y_train)
logreg = LogisticRegression()

params = {'C': np.logspace(start=-5, stop=3, num=9)}

clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)

clf.fit(X_train, y_train)

print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
X = np.arange(-16, 16).reshape(-1, 1)

preds = clf.predict_proba(X)[:,1]
plt.plot(X, preds)

plt.xlabel('Team1 seed - Team2 seed')

plt.ylabel('P(Team1 will win)')
df_sample_sub = pd.read_csv(data_dir + 'sample_submission.csv')

n_test_games = len(df_sample_sub)
def get_year_t1_t2(id):

    """Return a tuple with ints `year`, `team1` and `team2`."""

    return (int(x) for x in id.split('_'))
X_test = np.zeros(shape=(n_test_games, 1))

for ii, row in df_sample_sub.iterrows():

    year, t1, t2 = get_year_t1_t2(row.id)

    # There absolutely must be a better way of doing this!

    t1_seed = df_seeds[(df_seeds.Team == t1) & (df_seeds.Season == year)].n_seed.values[0]

    t2_seed = df_seeds[(df_seeds.Team == t2) & (df_seeds.Season == year)].n_seed.values[0]

    diff_seed = t1_seed - t2_seed

    X_test[ii, 0] = diff_seed
preds = clf.predict_proba(X_test)[:,1]
clipped_preds = np.clip(preds, 0.05, 0.95)

df_sample_sub.pred = clipped_preds

df_sample_sub.head()
df_sample_sub.to_csv('logreg_on_seed.csv', index=False)