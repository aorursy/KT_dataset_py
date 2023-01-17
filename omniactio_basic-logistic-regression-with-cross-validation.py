# Revision History

# Version 9: Updated submission with the Stage 2 data

# Version 8: Created sample submission

# Version 7: Fixed regular season stat bug

# Version 6: Added submission code



# This kernel creates basic logistic regression models and provides a 

# mechanism to select attributes and check results against tournaments since 2013



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

#tourney_cresults = pd.read_csv('../input/datafiles/NCAATourneyCompactResults.csv')

tourney_cresults = pd.read_csv('../input/stage2datafiles/NCAATourneyCompactResults.csv')

tourney_cresults = tourney_cresults.loc[tourney_cresults['Season'] >= 2003]



training_set = pd.DataFrame()

training_set['Result'] = np.random.randint(0,2,len(tourney_cresults.index))

training_set['Season'] = tourney_cresults['Season'].values

training_set['Team1'] = training_set['Result'].values * tourney_cresults['WTeamID'].values + (1-training_set['Result'].values) * tourney_cresults['LTeamID'].values 

training_set['Team2'] = (1-training_set['Result'].values) * tourney_cresults['WTeamID'].values + training_set['Result'].values * tourney_cresults['LTeamID'].values 

# Calculate Delta Seeds

#seeds = pd.read_csv('../input/datafiles/NCAATourneySeeds.csv')

seeds = pd.read_csv('../input/stage2datafiles/NCAATourneySeeds.csv')

seeds['Seed'] =  pd.to_numeric(seeds['Seed'].str[1:3], downcast='integer',errors='coerce')



def delta_seed(row):

    cond = (seeds['Season'] == row['Season'])

    return seeds[cond & (seeds['TeamID'] == row['Team1'])]['Seed'].iloc[0] - seeds[cond & (seeds['TeamID'] == row['Team2'])]['Seed'].iloc[0]



training_set['deltaSeed'] = training_set.apply(delta_seed,axis=1)
# Calculate Delta Ordinals

#mo = pd.read_csv('../input/masseyordinals/MasseyOrdinals.csv')

mo = pd.read_csv('../input/masseyordinals_thru_2019_day_128/MasseyOrdinals_thru_2019_day_128.csv')

mo = mo[(mo['RankingDayNum'] == 128) & (mo['Season'] >= 2003)] # See Note on MO



def delta_ord(row):

    cond =  (mo['Season'] == row['Season'])

    cond1 = (mo['TeamID'] == row['Team1']) & cond

    cond2 = (mo['TeamID'] == row['Team2']) & cond

    t1 = mo[cond1]['OrdinalRank'].mean()

    t2 = mo[cond2]['OrdinalRank'].mean()

    return  t1-t2



training_set['deltaMO'] = training_set.apply(delta_ord,axis=1)



# Calculate win pct

#season_dresults = pd.read_csv('../input/datafiles/RegularSeasonDetailedResults.csv')

season_dresults = pd.read_csv('../input/stage2datafiles/RegularSeasonDetailedResults.csv')

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

# deltaMO: difference in team's Massey Ordinals on day 128

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

cols = ['deltaSeed','deltaMO', 'deltaWinPct','deltaPointsFor','deltaPointsAgainst','deltaFGM','deltaFGA','deltaFGM3','deltaFGA3','deltaFTM',

        'deltaFTA','deltaOR','deltaDR','deltaAst','deltaTO','deltaStl','deltaBlk','deltaPF']

X = training_set[cols]

y = training_set['Result']



logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary2())       

# Stats from previous competitions



winner=[0.52951,0.438933,0.481309,0.438575,0.531942]

top5=[0.54772,0.453872,0.519706,0.461073,0.549876]

topprctvalues=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

topprctscores=[[0.55620,0.465207,0.536976,0.472893,0.576667,], # 5%

               [0.56035,0.469617,0.543597,0.479488,0.581399,], # 10%

               [0.56411,0.474137,0.550067,0.486432,0.586220,], # 15%

               [0.56958,0.478107,0.556954,0.491655,0.590554,], # 20%

               [0.57177,0.481519,0.563566,0.495595,0.594101,], # 25%

               [0.57495,0.486253,0.569619,0.500194,0.597493,], # 30%

               [0.57946,0.490536,0.577442,0.505821,0.601023,], # 35%

               [0.58676,0.496779,0.582541,0.511053,0.603920,], # 40%

               [0.59418,0.500862,0.585897,0.515844,0.608200,], # 45%

               [0.59921,0.508830,0.589459,0.522958,0.613297,], # 50%

               [0.60593,0.516735,0.594342,0.528533,0.621995,], # 55%

               [0.61288,0.523082,0.607036,0.534732,0.631227,], # 60%

               [0.61915,0.528358,0.627708,0.541466,0.645569,], # 65%

               [0.62780,0.539478,0.646980,0.556034,0.664821,], # 70%

               [0.65659,0.552809,0.682279,0.576019,0.688618,], # 75%

               [0.67932,0.574223,0.764058,0.605337,0.705595,], # 80%

               [0.73514,0.625012,0.865465,0.643514,0.776256,], # 85%

               [0.98764,0.692878,0.865465,0.693147,1.071796], # 90%

               [1.59656,0.882637,1.045207,0.897480,1.204277,], # 95%

               [19.18820,11.513103,9.092529,12.609623,18.092158,],] # 100%

# Run cross validation on model against tournaments in 2003-2018.  For the tournaments since 2014, the winning score

# from the Kaggle tournament is displayed



# TODO: Choose some columns to build the logistic regression models

cols = ['deltaMO', 'deltaWinPct', 'deltaPointsAgainst']



# TODO: Adjust winning probabilities by this percent for selected teams, based on your

biases = {

    '1163':1.2, # U Conn

    '1181':1.2, # Duke 

    '1437':1.2, # Villanova

    '1314':1.2, # UNC

}



errs = []

for year in range(2003,2019):

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

years1 = range(2003,2014)

for i in range(len(years1)):

    print(years1[i], "{0:.6f}".format(errs[i]),sep="\t")

    

print("Log losses by season")

print("--------------------")

print("year","your score","your result","winning score",sep="\t")    

years2 = range(2014,2019)

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

#sub = pd.read_csv('../input/SampleSubmissionStage1.csv')

sub = pd.read_csv('../input/SampleSubmissionStage2.csv')



# Create predictor attributes (as above for the CV)

sub['Season'], sub['Team1'], sub['Team2'] = sub['ID'].str.split('_').str

sub[['Season', 'Team1', 'Team2']] = sub[['Season', 'Team1', 'Team2']].apply(pd.to_numeric)



sub['deltaSeed'] = sub.apply(delta_seed,axis=1)

sub['deltaMO'] = sub.apply(delta_ord,axis=1)

sub['deltaWinPct'] = sub.apply(delta_winPct,axis=1)

cols = ['PointsFor','PointsAgainst','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']



for col in cols:

    print("Processing",col)

    sub['delta' + col] = sub.apply(delta_stat,args=(col,),axis=1)



# Build the final model

cols = ['deltaMO', 'deltaWinPct', 'deltaPointsAgainst']



# TODO: Adjust winning probabilities by this percent for selected teams, based on your

biases = {

    '1163':1.2, # U Conn

    '1181':1.2, # Duke 

    '1437':1.2, # Villanova

    '1314':1.2, # UNC

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

pred.loc[(sub['Team1'] == 1163) & (sub['Team2'] == 1181)] = 0.8



# Create Submission file

sub['Pred'] = pred

sub[['ID', 'Pred']].to_csv('submission.csv', index=False)
