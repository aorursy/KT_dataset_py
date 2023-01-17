# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

data_dir = '../input/'
df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'NCAATourneyCompactResults.csv')

# We load detailed season data to calculate season average statistics for each team
df_reg_season_detailed = pd.read_csv(data_dir + 'RegularSeasonDetailedResults.csv')
df_reg_season_detailed.drop(labels=['WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WDR', 'WAst', 
                'WStl', 'WBlk', 'WPF', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LDR', 
                'LAst', 'LStl', 'LBlk', 'LPF', 'WLoc', 'NumOT', 'WOR', 'LOR'], 
                            inplace=True, axis=1)
df_reg_season_detailed.head()
yearList = range(2003,2019) #2003 is the first year we have detailed data for
teams_pd = pd.read_csv(data_dir + 'Teams.csv')
teamIDs = teams_pd['TeamID'].tolist()

rows = list()

for year in yearList:
    for team in teamIDs:
        df_curr_season = df_reg_season_detailed[df_reg_season_detailed.Season == year]       

        df_curr_team_wins = df_curr_season[df_curr_season.WTeamID == team]
        df_curr_team_losses = df_curr_season[df_curr_season.LTeamID == team]
        
        # no games played by them this year.. skip (current team didn't win or lose any games)
        if df_curr_team_wins.shape[0] == 0 and df_curr_team_losses.shape[0] == 0:
            continue;
        
        df_winteam = df_curr_team_wins.rename(columns={'WTeamID':'TeamID', 'WFGM':'FGM', 
                    'WFGA':'FGA', 'WTO':'TO', 'WScore':'Score', 'LScore':'OppScore'})
        
        # drop all columns except the ones we are using
        df_winteam = df_winteam[['TeamID', 'FGM', 'FGA', 'TO', 'Score', 'OppScore']]

        df_loseteam = df_curr_team_losses.rename(columns={'LTeamID':'TeamID', 'LFGM':'FGM',
                    'LFGA':'FGA', 'LTO':'TO', 'LScore':'Score', 'WScore':'OppScore'})
        # drop all columns except the ones we are using
        df_loseteam = df_loseteam[['TeamID', 'FGM', 'FGA', 'TO', 'Score', 'OppScore']] 

        # dataframe w/ all relevant stats from current year for current team
        df_curr_team = pd.concat((df_winteam, df_loseteam)) 

        wins = df_winteam.shape[0]
        FGPercent = df_curr_team['FGM'].sum() / df_curr_team['FGA'].sum()
        TurnoverAvg = df_curr_team['TO'].sum() / len(df_curr_team['TO'].values)
        PPG = df_curr_team['Score'].sum() / len(df_curr_team['Score'].values)
        OppPPG = df_curr_team['OppScore'].sum() / len(df_curr_team['OppScore'].values)

        # collect all data in rows list first for effeciency
        rows.append([year, team, wins, FGPercent, TurnoverAvg, PPG, OppPPG])

df_training_data = pd.DataFrame(rows, columns=['Season', 'TeamID', 'Wins', 'FGPercent', 
                                               'TOAvg', 'PPG', 'OppPPG'])
df_training_data.head()
df_seeds.head()
df_tour.head()
def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_seeds.head()
df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tour.head()
df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
df_concat.head()
df_winstats = df_training_data.rename(columns={'TeamID':'WTeamID', 'FGPercent':'WFGPercent', 
                            'TOAvg':'WTOAvg', 'PPG':'WPPG', 'OppPPG':'WOppPPG', 'Wins':'WWins'})
df_lossstats = df_training_data.rename(columns={'TeamID':'LTeamID', 'FGPercent':'LFGPercent',
                            'TOAvg':'LTOAvg', 'PPG':'LPPG', 'OppPPG':'LOppPPG', 'Wins':'LWins'})
df_dummy = pd.merge(left=df_concat, right=df_winstats, on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossstats, on=['Season', 'LTeamID'])
df_concat['FGPercentDiff'] = df_concat.WFGPercent - df_concat.LFGPercent
df_concat['TOAvgDiff'] = df_concat.WTOAvg - df_concat.LTOAvg
df_concat['PPGDiff'] = df_concat.WPPG - df_concat.LPPG
df_concat['OppPPGDiff'] = df_concat.WOppPPG - df_concat.LOppPPG
df_concat['WWinMargin'] = df_concat.WPPG - df_concat.WOppPPG
df_concat['LWinMargin'] = df_concat.LPPG - df_concat.LOppPPG
df_concat['WinMarginDiff'] = df_concat.WWinMargin - df_concat.LWinMargin
df_concat['WinDiff'] = df_concat.WWins - df_concat.LWins
 # drop all columns except the ones we are using
df_concat = df_concat[['Season', 'WTeamID', 'LTeamID', 'SeedDiff', 'FGPercentDiff', 
                       'TOAvgDiff', 'PPGDiff', 'OppPPGDiff', 'WinMarginDiff', 'WinDiff']]

# Note: We can have SeedDiff == 0 due to the First Four (68 teams)! Also Final Four onwards!
# Note: Pandas merges tossed out data from before 2003!
df_concat.head()
# We create positive and negative versions of the data so the 
# supervised learning algorithm has sample data of each class to classify

df_wins = pd.DataFrame()
df_wins['SeedDiff'] = df_concat['SeedDiff']
df_wins['FGPercentDiff'] = df_concat['FGPercentDiff']
df_wins['TOAvgDiff'] = df_concat['TOAvgDiff']
df_wins['PPGDiff'] = df_concat['PPGDiff']
df_wins['OppPPGDiff'] = df_concat['OppPPGDiff']
df_wins['WinMarginDiff'] = df_concat['WinMarginDiff']
df_wins['WinDiff'] = df_concat['WinDiff']
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['SeedDiff'] = -df_concat['SeedDiff']
df_losses['FGPercentDiff'] = -df_concat['FGPercentDiff']
df_losses['TOAvgDiff'] = -df_concat['TOAvgDiff']
df_losses['PPGDiff'] = -df_concat['PPGDiff']
df_losses['OppPPGDiff'] = -df_concat['OppPPGDiff']
df_losses['WinMarginDiff'] = -df_concat['WinMarginDiff']
df_losses['WinDiff'] = -df_concat['WinDiff']
df_losses['Result'] = 0

df_predictions = pd.concat((df_wins, df_losses))
df_predictions.head()
X_train = [list(a) for a in zip(df_predictions.SeedDiff.values, df_predictions.FGPercentDiff.values, 
                                df_predictions.TOAvgDiff.values, df_predictions.PPGDiff.values,
                                df_predictions.OppPPGDiff.values, df_predictions.WinMarginDiff.values,
                                df_predictions.WinDiff.values)]
X_train = np.array(X_train)
y_train = df_predictions.Result.values
X_train, y_train = shuffle(X_train, y_train)
# Neural Network
params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
mlp = MLPClassifier(learning_rate='adaptive')
clf = GridSearchCV(mlp, params, scoring='neg_log_loss')
clf.fit(X_train, y_train)
print('Best log_loss Multi Layer Perceptron Classifier: {}'.format(clf.best_score_))

# Gradient Boosted Classifier
GBC = GradientBoostingClassifier()
param_grid_GBC = {
    "n_estimators" : [100],
    "learning_rate" : [0.1, 0.05, 0.02, 0.01],
    "max_depth" : [1,2,3],
    "min_samples_leaf" : [1,3,5],
    "max_features" : [1.0, 0.3, 0.1]
}
clf = GridSearchCV(GBC, param_grid_GBC, scoring='neg_log_loss')
clf.fit(X_train, y_train)
print('Best log_loss Gradient Boosting Classifier: {}'.format(clf.best_score_))

# Random Forest Classifier
RFC = RandomForestClassifier()
param_grid_RFC = { 
    'n_estimators': [60, 120, 240],
    'max_features': ['auto', 'sqrt', 'log2']
}
clf = GridSearchCV(RFC, param_grid_RFC, scoring='neg_log_loss')
clf.fit(X_train, y_train)
print('Best log_loss Random Forest Classifier: {}'.format(clf.best_score_))

# K Nearest Neighbors Classifier
knn = KNeighborsClassifier()
k = np.arange(80)+1
parameters = {'n_neighbors': k}
clf = GridSearchCV(knn, parameters, scoring='neg_log_loss')
clf.fit(X_train, y_train)
print('Best log_loss K-Nearest Neighbors Classifier: {}'.format(clf.best_score_))

# SVC
SVC = svm.SVC(probability=True)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
tuned_parameters_preselected = [{'kernel': ['linear'], 'C': [10]}]
clf = GridSearchCV(SVC, tuned_parameters_preselected, scoring='neg_log_loss')
clf.fit(X_train, y_train)
print('Best log_loss Support Vector Classification: {}'.format(clf.best_score_))

# Logistic Regression
logreg = LogisticRegression()
params = {'C': np.logspace(start=-15, stop=15, num=31)} # {C: array[1^-15 , 1^-14, ... 1^15] }
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True) #sklearn model selection
clf.fit(X_train, y_train)
print('Best log_loss Logistic Regression: {}, with best C: {}'.format(clf.best_score_, 
                                                                      clf.best_params_['C']))

# Logistic Regression is typically the top-performer. We compute it last, and use 
# this classifier to make future predictions.

# SVC is typically a close second. Comment out Logistic Regression to use 
# the SVC classifier instead to make future predictions

# Keep in mind, the provided values are a single representation of our classifier's
# success! Depending on how the data is shuffled, each run of the program may yield
# a slightly different classifier (and thus different predictions/success rate)
# Create training data with the seeds varying from -10, 10
# All other features are zeroed out so the plot only shows
# the relationship between seed and P(team1 wins)
X1 = np.arange(-10, 10)
X2 = np.zeros(20, dtype=np.int)
X = [list(a) for a in zip(X1, X2, X2, X2, X2, X2, X2)]
X = np.array(X)

preds = clf.predict_proba(X)[:,1]

plt.plot(X1, preds)
plt.xlabel('Team1 seed - Team2 seed')
plt.ylabel('P(Team1 will win)')
df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage2.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))
X_test = np.zeros(shape=(n_test_games, 7))

for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii, 0] = diff_seed
    
    t1_FGPercent = df_training_data[(df_training_data.TeamID == t1) & 
                                    (df_training_data.Season == year)].FGPercent.values[0]
    t2_FGPercent = df_training_data[(df_training_data.TeamID == t2) & 
                                    (df_training_data.Season == year)].FGPercent.values[0]
    diff_FGPercent = t1_FGPercent - t2_FGPercent
    X_test[ii, 1] = diff_FGPercent
    
    t1_TOAvg = df_training_data[(df_training_data.TeamID == t1) & 
                                (df_training_data.Season == year)].TOAvg.values[0]
    t2_TOAvg = df_training_data[(df_training_data.TeamID == t2) & 
                                (df_training_data.Season == year)].TOAvg.values[0]
    diff_TOAvg = t1_TOAvg - t2_TOAvg
    X_test[ii, 2] = diff_TOAvg
    
    t1_PPG = df_training_data[(df_training_data.TeamID == t1) & 
                              (df_training_data.Season == year)].PPG.values[0]
    t2_PPG = df_training_data[(df_training_data.TeamID == t2) & 
                              (df_training_data.Season == year)].PPG.values[0]
    diff_PPG = t1_PPG - t2_PPG
    X_test[ii, 3] = diff_PPG
    
    t1_OppPPG = df_training_data[(df_training_data.TeamID == t1) & 
                                 (df_training_data.Season == year)].OppPPG.values[0]
    t2_OppPPG = df_training_data[(df_training_data.TeamID == t2) & 
                                 (df_training_data.Season == year)].OppPPG.values[0]
    diff_OppPPG = t1_OppPPG - t2_OppPPG
    X_test[ii, 4] = diff_OppPPG
    
    X_test[ii, 5] = diff_PPG - diff_OppPPG # Win Margin
    
    t1_Wins = df_training_data[(df_training_data.TeamID == t1) & 
                                 (df_training_data.Season == year)].Wins.values[0]
    t2_Wins = df_training_data[(df_training_data.TeamID == t2) & 
                                 (df_training_data.Season == year)].Wins.values[0]
    X_test[ii, 6] = t1_Wins - t2_Wins
preds = clf.predict_proba(X_test)[:,1]

clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
df_sample_sub.head()
df_sample_sub.to_csv('predictions.csv', index=False)