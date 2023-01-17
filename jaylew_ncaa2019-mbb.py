# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#The data has been cleaned for us already, so no concerns about that. Start by grabbing previous results, seeds, and MasseyOrdinals

data_dir = '../input/'

df_seeds = pd.read_csv(data_dir + 'stage2datafiles/NCAATourneySeeds.csv')

df_tour = pd.read_csv(data_dir + 'stage2datafiles/NCAATourneyCompactResults.csv')

df_massey = pd.read_csv(data_dir + 'masseyordinals/MasseyOrdinals.csv')
#Convert the seed values to an integer, we are not concerned about the conference

def seedToInt(seed):

    #Since each seed begins with a letter (W01), skip the first element

    return int(seed[1:3])



#Apply the function

df_seeds['int_seed'] = df_seeds['Seed'].apply(seedToInt)

df_seeds.drop('Seed', inplace=True, axis=1)

df_seeds.head()
#Only use the final massey ranking for the year

df_finalmassey = df_massey[df_massey['RankingDayNum']==max(df_massey['RankingDayNum'].values)]



#Need to group them by team and season, and average all the rankings from each system

df_gb = df_finalmassey.groupby(['TeamID', 'Season'])['OrdinalRank'].mean().unstack('TeamID')



#Reset the index to simplify grabbing the values. Not particularly clean, but it works.

df_SeasonTeamID = df_gb.reset_index()



#Get all of the seasons we now have rankings for

massey_seasons = list(df_SeasonTeamID['Season'].values)
#Don't care about the Day, score, location or OT right now

df_tour.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)

df_tour.tail()



#Subset the tour data using the first year in our massey data as the starting point

df_subs_tour = df_tour[df_tour['Season'] >= min(massey_seasons)]
#Fix the index and check how our data looks

df_subs_tour.reset_index(inplace=True, drop=True)

df_subs_tour.head()
#Now match up the Massey Rankings for each winning and losing team

WMassey = [0]*len(df_subs_tour)

LMassey = [0]*len(df_subs_tour)



for ind, row in df_subs_tour.iterrows():

    season = row['Season']

    wid = row['WTeamID']

    lid = row['LTeamID']

    WMassey[ind] = df_SeasonTeamID[df_SeasonTeamID['Season']==season][wid].values[0]

    LMassey[ind] = df_SeasonTeamID[df_SeasonTeamID['Season']==season][lid].values[0]
#Use the lists to create our new columns

df_subs_tour['WMassey'] = WMassey

df_subs_tour['LMassey'] = LMassey



df_subs_tour.head(10)
#Now make the df with seed differences and massey differences. Need to get a seed value for each winner/loser

df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'int_seed':'WSeed'})

df_loseseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'int_seed':'LSeed'})



#Merge the seeds and subsetted data from earlier

df_d = pd.merge(left=df_subs_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])

df_concat = pd.merge(left=df_d, right=df_loseseeds, on=['Season', 'LTeamID'])



#Calculate the difference columns

df_concat['SeedDiff'] = df_concat['WSeed'] - df_concat['LSeed']

df_concat['MasseyDiff'] = df_concat['WMassey'] - df_concat['LMassey']



#Confirm the data looks as we expect

df_concat.head()
#Build the dataframe to be used for our machine learning models

df_wins = pd.DataFrame()

df_wins['SeedDiff'] = df_concat['SeedDiff']

df_wins['MasseyDiff'] = df_concat['MasseyDiff']

df_wins['Result'] = 1



df_losses = pd.DataFrame()

df_losses['SeedDiff'] = -df_concat['SeedDiff']

df_losses['MasseyDiff'] = -df_concat['MasseyDiff']

df_losses['Result'] = 0



df_predictions = pd.concat((df_wins, df_losses))

df_predictions.head()
#Split the data into training predictors and results

X_train = df_predictions[['SeedDiff', 'MasseyDiff']].values

y_train = df_predictions['Result'].values



X_train, y_train = shuffle(X_train, y_train)
#Import libraries

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
#Build a random forest classifier and use GridSearch to determine the optimal number of trees

rf = RandomForestClassifier()

rf_params = {'n_estimators': [100, 200, 300, 400, 500, 1000]}

rf_grid = GridSearchCV(rf, rf_params, scoring='neg_log_loss', refit=True)

rf_grid.fit(X_train, y_train)



#Print our best result based on the neg_log_loss (this is the parameter for the competition)

print('Best log_loss: {:.4}, with best C: {}'.format(rf_grid.best_score_, rf_grid.best_params_['n_estimators']))
#Also build a logistic regression model using a similar method

logreg = LogisticRegression(solver='lbfgs')

params = {'C': [0.001,0.01, 1, 10, 100]}

log_grid = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)

log_grid.fit(X_train, y_train)



print('Best log_loss: {:.4}, with best C: {}'.format(log_grid.best_score_, log_grid.best_params_['C']))
#Get the sample submission and length

df_sub = pd.read_csv('../input/SampleSubmissionStage2.csv')

df_massey_2019 = pd.read_csv('../input/prelim2019_masseyordinals/Prelim2019_MasseyOrdinals.csv')

len_sub = len(df_sub)
#Grab the relevant massey ordinals for this year

df_massey_2019 = df_massey_2019[df_massey_2019['Season']==2019]

df_m = df_massey_2019.groupby('TeamID').mean()

df_m.reset_index(drop=False, inplace=True)

df_m.head()
#Function to extract year, team 1, team 2

def getYearTeams(ID):

    return (int(x) for x in ID.split('_'))
#Iterate through the submission matchups, and create the needed difference columns for the models

X_test = np.zeros(shape=(len_sub, 2))

for ii, row in df_sub.iterrows():

    year, t1, t2 = getYearTeams(row['ID'])

    t1_seed = df_seeds[(df_seeds['TeamID'] == t1) & (df_seeds['Season'] == year)]['int_seed'].values[0]

    t2_seed = df_seeds[(df_seeds['TeamID'] == t2) & (df_seeds['Season'] == year)]['int_seed'].values[0]

    t1_mass = df_m[(df_m['TeamID'] == t1)]['OrdinalRank'].values[0]

    t2_mass = df_m[(df_m['TeamID'] == t2)]['OrdinalRank'].values[0]

    diff_seed = t1_seed - t2_seed

    diff_mass = t1_mass - t2_mass

    X_test[ii, 0] = diff_seed

    X_test[ii, 1] = diff_mass
#Build the predictions, clip off any values which are too confident, limiting them in the [0.5,0.95] range.

log_preds = log_grid.predict_proba(X_test)[:,1]

rf_preds = rf_grid.predict_proba(X_test)[:,1]



clipped_logpreds = np.clip(log_preds, 0.05, 0.95)

clipped_rf = np.clip(rf_preds, 0.05, 0.95)



#Choose which model predictions to use in the final submission. Tried both, RF results were better.

df_sub['Pred'] = clipped_rf



#Confirm the length of our submission is ok

len(df_sub)
#Export the submission file

df_sub.to_csv('tut_w_mass_submission.csv', index=False)