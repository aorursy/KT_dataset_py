# This kernel creates basic logistic regression models and provides a 

# mechanism to select attributes and check results against tournaments since 2018



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# To create the final sumissions once the round2 data has been updated, set this to true

ROUND2 = False

DATA_PATH = '../input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/'

ROUND1_PATH = DATA_PATH + "WDataFiles_Stage1/"

ROUND2_PATH = DATA_PATH + "WDataFiles_Stage2/"
if ROUND2:

    tourney_cresults = pd.read_csv(ROUND2_PATH + 'WNCAATourneyCompactResults.csv')

else:

    tourney_cresults = pd.read_csv(ROUND1_PATH + 'WNCAATourneyCompactResults.csv')

    

tourney_cresults = tourney_cresults.loc[tourney_cresults['Season'] >= 2010]



training_set = pd.DataFrame()

training_set['Result'] = np.random.randint(0,2,len(tourney_cresults.index))

training_set['Season'] = tourney_cresults['Season'].values

training_set['Team1'] = training_set['Result'].values * tourney_cresults['WTeamID'].values + (1-training_set['Result'].values) * tourney_cresults['LTeamID'].values 

training_set['Team2'] = (1-training_set['Result'].values) * tourney_cresults['WTeamID'].values + training_set['Result'].values * tourney_cresults['LTeamID'].values 

# Calculate Delta Seeds

if ROUND2:

    seeds = pd.read_csv(ROUND2_PATH + 'WNCAATourneySeeds.csv')

else:

    seeds = pd.read_csv(ROUND1_PATH + 'WNCAATourneySeeds.csv')



seeds['Seed'] =  pd.to_numeric(seeds['Seed'].str[1:3], downcast='integer',errors='coerce')



def delta_seed(row):

    cond = (seeds['Season'] == row['Season'])

    return seeds[cond & (seeds['TeamID'] == row['Team1'])]['Seed'].iloc[0] - seeds[cond & (seeds['TeamID'] == row['Team2'])]['Seed'].iloc[0]



training_set['deltaSeed'] = training_set.apply(delta_seed,axis=1)
# Calculate win pct

if ROUND2:

    season_dresults = pd.read_csv(ROUND2_PATH + 'WRegularSeasonDetailedResults.csv')

else:

    season_dresults = pd.read_csv(ROUND1_PATH + 'WRegularSeasonDetailedResults.csv')



record = pd.DataFrame({'wins': season_dresults.groupby(['Season','WTeamID']).size()}).reset_index();

losses = pd.DataFrame({'losses': season_dresults.groupby(['Season','LTeamID']).size()}).reset_index();



record = record.merge(losses, how='outer', left_on=['Season','WTeamID'], right_on=['Season','LTeamID'])

record = record.fillna(0)

record['games'] = record['wins'] + record['losses']



def delta_winPct(row):

    cond1 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team1'])

    cond2 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team2'])

    return (record[cond1]['wins']/record[cond1]['games']).mean() - (record[cond2]['wins']/record[cond2]['games']).mean()



training_set['deltaWinPct'] = training_set.apply(delta_winPct,axis=1)





dfW = season_dresults.groupby(['Season','WTeamID']).sum().reset_index()

dfL = season_dresults.groupby(['Season','LTeamID']).sum().reset_index()



def get_points_for(row):

    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID']) 

    fld1 = 'WScore'

    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID']) 

    fld2 = 'LScore'

    retVal = dfW[wcond][fld1].sum()

    if len(dfL[lcond][fld2]) > 0:

        retVal = retVal + dfL[lcond][fld2].sum() 

    return retVal



def get_points_against(row):

    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID']) 

    fld1 = 'LScore'

    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID']) 

    fld2 = 'WScore'

    retVal = dfW[wcond][fld1].sum()

    if len(dfL[lcond][fld2]) > 0:

        retVal = retVal + dfL[lcond][fld2].sum() 

    return retVal



record['PointsFor'] = record.apply(get_points_for, axis=1)

record['PointsAgainst'] = record.apply(get_points_against, axis=1)
def get_remaining_stats(row, field):

    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID']) 

    fld1 = 'W' + field

    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID']) 

    fld2 = 'L'+ field

    retVal = dfW[wcond][fld1].sum()

    if len(dfL[lcond][fld2]) > 0:

        retVal = retVal + dfL[lcond][fld2].sum()

    return retVal



cols = ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']



for col in cols:

    print("Processing",col)

    record[col] = record.apply(get_remaining_stats, args=(col,), axis=1)



#record['FGprct'] = record['FGM'] / record['FGA']    

def delta_stat(row, field):

    cond1 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team1'])

    cond2 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team2'])

    return (record[cond1][field]/record[cond1]['games']).mean() - (record[cond2][field]/record[cond2]['games']).mean()



cols = ['PointsFor','PointsAgainst','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']



for col in cols:

    print("Processing",col)

    training_set['delta' + col] = training_set.apply(delta_stat,args=(col,),axis=1)



training_set.describe()

# Train a model on all of the data

import statsmodels.api as sm



# Field descriptions:

# deltaSeed: difference in team's seeds

# deltaWinPct: difference in the team's winning percentage 

# deltaPointsFor: difference in the average points scored per game

# deltaPointsAgainst: difference in the average points scored agains the teams

# deltaFGM: difference in the field goals made per game

# deltaFGA: difference in the field goals attempted per game

# deltaFGM3: difference in 3 point fields goals made per game

# deltaFGA3: difference in the 3 points fields goals attempted per game

# deltaFTM: difference in free throws made per game

# deltaFTA: difference in free throws attempted per game

# deltaOR: difference in offence rebounds per game

# deltaDR: difference in defensive rebounds per game

# deltaAst: difference in assists per game

# deltaTO: difference in turnovers per game

# deltaStl: difference in steals per game

# deltaBlk: difference in blocks per game

# deltaPF: difference in personal fouls per game



# You would probabaly want to select a subset of these attributes

cols = ['deltaSeed', 'deltaWinPct','deltaPointsFor','deltaPointsAgainst','deltaFGM','deltaFGA','deltaFGM3','deltaFGA3','deltaFTM',

        'deltaFTA','deltaOR','deltaDR','deltaAst','deltaTO','deltaStl','deltaBlk','deltaPF']

X = training_set[cols]

y = training_set['Result']



logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary2())       

# Stats from previous competitions



topprctvalues=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]



#                 2018     2019

winner=        [0.40681,0.32245,]

top5=          [0.42102,0.33531,]

topprctscores=[[0.43673,0.35553,], # 5%

               [0.44596,0.36147,], # 10%

               [0.45187,0.36786,], # 15%

               [0.45688,0.37425,], # 20%

               [0.46138,0.38114,], # 25%

               [0.46603,0.38672,], # 30%

               [0.47232,0.39384,], # 35%

               [0.47954,0.39988,], # 40%

               [0.48813,0.40669,], # 45%

               [0.49291,0.41073,], # 50%

               [0.50050,0.41809,], # 55%

               [0.51622,0.42810,], # 60%

               [0.53278,0.44893,], # 65%

               [0.55833,0.50060,], # 70%

               [0.58590,0.58947,], # 75%

               [0.63404,0.79043,], # 80%

               [0.69314,0.97132,], # 85%

               [0.83175,16.99557,], # 90%

               [1.03407,18.64027,], # 95%

               [13.22375,24.12266,],] # 100%

# Run cross validation on model against tournaments in 2003-2018.  For the tournaments since 2014, the winning score

# from the Kaggle tournament is displayed



# TODO: Choose some columns to build the logistic regression models

cols = ['deltaSeed', 'deltaWinPct', 'deltaPointsAgainst']



# TODO: Adjust winning probabilities by this percent for selected teams, based on your

biases = {

    '1163':1.2, # U Conn

    '1181':1.2, # Duke 

    '1437':1.2, # Villanova

    '1314':1.2, # UNC

}



errs = []

for year in range(2010,2020):

    print("Evaluation on tournament year",year)

    Xtrain = training_set[training_set['Season'] != year][cols]

    ytrain = training_set[training_set['Season'] != year]['Result']



    logit_model=sm.Logit(ytrain,Xtrain)

    result=logit_model.fit()



    Xtest = training_set[training_set['Season'] == year][cols]

    ytest = training_set[training_set['Season'] == year]['Result']

    

    pred = result.predict(Xtest)

    for bias in biases:

        pred.loc[(training_set['Season'] == year) & (training_set['Team1'] == int(bias))] = pred.loc[(training_set['Season'] == year) & (training_set['Team1'] == int(bias))] * biases[bias] 

        pred.loc[(training_set['Season'] == year) & (training_set['Team2'] == int(bias))] = pred.loc[(training_set['Season'] == year) & (training_set['Team2'] == int(bias))] / biases[bias] 

    

    pred.loc[(pred >= 0.9999)] = 0.9999

    pred.loc[(pred <= 0.0001)] = 0.0001



    pred.loc[training_set[training_set['Season'] == year]['Result'] == 0] = 1 - pred.loc[training_set[training_set['Season'] == year]['Result'] == 0] 

    err = -np.log(pred).mean()



    errs.append(err)



print("Mean log loss: ",np.mean(errs))







print("Log losses by season")

print("--------------------")

print("year","your score",sep="\t")

years1 = range(2010,2018)

for i in range(len(years1)):

    print(years1[i], "{0:.6f}".format(errs[i]),sep="\t")

    

print("Log losses by season")

print("--------------------")

print("year","your score","your result","winning score",sep="\t")    

years2 = range(2018,2020)

for i in range(len(years2)):

    result = None

    if errs[i+len(years1)] < winner[i]:

        result="Win competition"

    elif errs[i+len(years1)] < top5[i]:

        result="Top 5 score"

    else:

        for j in range(len(topprctvalues)):

            if errs[i+len(years1)] < topprctscores[j][i]:

                result = "Top " + str(topprctvalues[j]) + "%"

                break

        if result is None:

            result = "Worst score on Kaggle"



    print(years2[i], "{0:.6f}".format(errs[i+len(years1)]),result+"\t",winner[i],sep="\t")

## Create a submission file

if ROUND2:

    sub = pd.read_csv(DATA_PATH + 'WSampleSubmissionStage2_2020.csv')

else:

    sub = pd.read_csv(DATA_PATH + 'WSampleSubmissionStage1_2020.csv')



# Create predictor attributes (as above for the CV)

sub['Season'], sub['Team1'], sub['Team2'] = sub['ID'].str.split('_').str

sub[['Season', 'Team1', 'Team2']] = sub[['Season', 'Team1', 'Team2']].apply(pd.to_numeric)



sub['deltaSeed'] = sub.apply(delta_seed,axis=1)

sub['deltaWinPct'] = sub.apply(delta_winPct,axis=1)



cols = ['PointsFor','PointsAgainst','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']

for col in cols:

    print("Processing",col)

    sub['delta' + col] = sub.apply(delta_stat,args=(col,),axis=1)



# Build the final model

cols = ['deltaSeed', 'deltaWinPct', 'deltaPointsAgainst']



# TODO: Adjust winning probabilities by this percent for selected teams, based on your

biases = {

    '3163':1.2, # U Conn

    '3181':1.2, # Duke 

    '3437':1.2, # Villanova

    '3314':1.2, # UNC

}



Xtrain = training_set[cols]

ytrain = training_set['Result']



logit_model=sm.Logit(ytrain,Xtrain)

result=logit_model.fit()



# Make your predictions

Xtest = sub[cols]

pred = result.predict(Xtest)



# Bias results based on team preferences

for bias in biases:

    pred.loc[sub['Team1'] == int(bias)] = pred.loc[sub['Team1'] == int(bias)] * biases[bias] 

    pred.loc[sub['Team2'] == int(bias)] = pred.loc[sub['Team2'] == int(bias)] / biases[bias] 

    

pred.loc[(pred >= 0.9999)] = 0.9999

pred.loc[(pred <= 0.0001)] = 0.0001





# Manually adjust some predictions

# U Conn vs Duke

pred.loc[(sub['Team1'] == 3163) & (sub['Team2'] == 3181)] = 0.8



# Create Submission file

sub['Pred'] = pred

sub[['ID', 'Pred']].to_csv('submission.csv', index=False)
