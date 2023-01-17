# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.utils import shuffle

from sklearn.model_selection import GridSearchCV
reg_detailed_results = pd.read_csv("../input/RegularSeasonDetailedResults.csv")

reg_detailed_results[:6]
#reg_detailed_results.dtypes
reg_detailed_results[['Wscore','Lscore']].plot.hist(alpha = 0.5, bins =30, stacked = False)
seeds = pd.read_csv("../input/TourneySeeds.csv")

seeds[:6]
def split_seed(Seed):

    return (str(Seed[0]),int(Seed[1:3]))

print(type(split_seed('z16a')))



def get_seed_num(Seed):

    return split_seed(Seed)[1]

def get_seed_reg(Seed):

    return split_seed(Seed)[0]

#print(set(seeds['Seed']))

seeds['region'] = seeds['Seed'].apply(get_seed_reg)

seeds['num_seed'] = seeds['Seed'].apply(get_seed_num)

seeds.drop(labels=['Seed'], inplace=True, axis=1)

print(seeds[:6])



win_seeds = seeds.rename(columns={'Team':'Wteam', 'num_seed':'win_seed','region':'win_region'})

loss_seeds = seeds.rename(columns={'Team':'Lteam', 'num_seed':'loss_seed','region':'loss_region'})



print(win_seeds[:3])

print(loss_seeds[:3])
tourn_compact = pd.read_csv("../input/TourneyCompactResults.csv")

tourn_compact[:6]
tourn_full = pd.read_csv("../input/TourneyDetailedResults.csv")

tourn_full[:6]
# Follow other's log regression on seed
temp = pd.merge(left=tourn_compact, right=win_seeds, how='left', on=['Season', 'Wteam'])

full_data = pd.merge(left=temp, right=loss_seeds, on=['Season', 'Lteam'])

full_data['seed_diff'] = full_data['win_seed'] - full_data['loss_seed'] 

full_data[:6]
df_wins = pd.DataFrame()

df_wins['seed_diff'] = full_data['seed_diff']

df_wins['result'] = 1



df_losses = pd.DataFrame()

df_losses['seed_diff'] = -full_data['seed_diff']

df_losses['result'] = 0



df_for_predictions = pd.concat((df_wins, df_losses))

df_for_predictions[:6]

#len(df_for_predictions)
logreg = LogisticRegression()

params = {'C': np.logspace(start=-7, stop=8, num=15)}

clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)

clf.fit(df_for_predictions.seed_diff.values.reshape(-1,1), df_for_predictions.result.values)

print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
X = np.arange(-16, 16).reshape(-1, 1)

preds = clf.predict_proba(X)[:,1]



plt.plot(X, preds)

plt.xlabel('Team1 seed - Team2 seed')

plt.ylabel('P(Team1 will win)')