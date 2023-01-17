# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_concat = pd.read_csv('../input/WNCAATourneyCompactResults.csv')
df_concat.head()
df_concat['diff_score'] = df_concat.WScore - df_concat.LScore
df_seeds = pd.read_csv('../input/WNCAATourneySeeds.csv')
df_seeds.head()
df_seeds['seed_int'] = df_seeds['Seed'].apply(lambda seed: int(seed[1:]))
df_seeds.drop('Seed', axis=1, inplace=True)
df_seeds.head()
df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})

df_dummy = pd.merge(left=df_concat, right=df_winseeds, how='left', 
                    on=['Season','WTeamID'])
concat = pd.merge(left=df_dummy, right=df_lossseeds, 
                  on=['Season','LTeamID'])
concat['SeedDiff'] = concat.WSeed - concat.LSeed
concat.head()
concat.isnull().any()
win = pd.DataFrame()
win[['Season','diff_score','SeedDiff']] = concat[['Season','diff_score','SeedDiff']].copy()
win['result'] = 1

loss = pd.DataFrame()
loss[['Season','diff_score','SeedDiff']] = concat[['Season','diff_score','SeedDiff']].copy()
loss['diff_score'] = -loss['diff_score']
loss['SeedDiff'] = -loss['SeedDiff']
loss['result'] = 0

df_pred = pd.concat((win, loss))
df_pred.head()
%matplotlib inline
import matplotlib.pyplot as plt

plt.scatter(df_pred.SeedDiff, df_pred.diff_score)
plt.xlabel('SeedDiff')
plt.ylabel('Diff Score')
plt.show()
plt.scatter(df_pred.loc[df_pred.Season == 1998, 'SeedDiff'], 
            df_pred.loc[df_pred.Season == 1998, 'diff_score'])
plt.scatter(df_pred.loc[df_pred.Season == 2000, 'SeedDiff'], 
            df_pred.loc[df_pred.Season == 2000, 'diff_score'], c='r')
plt.scatter(df_pred.loc[df_pred.Season == 2002, 'SeedDiff'], 
            df_pred.loc[df_pred.Season == 2002, 'diff_score'], c='g')
plt.xlabel('SeedDiff')
plt.ylabel('Diff Score')
plt.show()
regressor = RandomForestRegressor()
regressor.fit(df_pred[['Season', 'SeedDiff']].values, df_pred.diff_score.values)
regressor.score(df_pred[['Season', 'SeedDiff']].values, df_pred.diff_score.values)
X_train = pd.DataFrame()
X_train[['Season','SeedDiff']] = df_pred[['Season','SeedDiff']].copy()
X_train['DiffScoreRegressor'] = regressor.predict(df_pred[['Season', 'SeedDiff']].values)
y_train = df_pred.result

X_train.head()
X_train = X_train.values
X_train[:3]
estimator = np.array([10, 50, 100, 300, 500])
depth = np.array([3, 4])
c = ['gini']
state = np.array(list(range(51)))
params = {'n_estimators':estimator, 'max_depth':depth, 
          'criterion':c, 'random_state':state}

model = RandomForestClassifier()

clf = GridSearchCV(model, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best{}'.format(clf.best_score_, clf.best_params_))
preds = clf.predict_proba(X_train)
plt.scatter(X_train[:,1], preds[:,1])
plt.xlabel('Team1 seed - Team2 seed')
plt.ylabel('P(Team1 will win)')
plt.show()
df_sample_sub = pd.read_csv('../input/WSampleSubmissionStage1.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

X_test = np.zeros((n_test_games,3))

for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    
    diff_seed = t1_seed - t2_seed
    X_test[ii,0], X_test[ii,1] = year, diff_seed
    #X_test[ii,2] = regressor.predict(X_test[ii,:2].reshape(-1, 1))
X_test[:,2] = regressor.predict(X_test[:,:2])
y_preds = clf.predict_proba(X_test)[:,1]
clipped_y_pred = np.clip(y_preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_y_pred
df_sample_sub.head()
df_sample_sub.to_csv('New_Feature-NCAA.csv', index=False)