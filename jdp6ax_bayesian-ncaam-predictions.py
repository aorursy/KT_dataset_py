import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split

# To model the Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
import re
players_2010 = pd.read_csv('../input/Players_2010.csv')
#Import events
events_2010 = pd.read_csv('../input/Events_2010.csv')
events_2011 = pd.read_csv('../input/Events_2011.csv')
events_2012 = pd.read_csv('../input/Events_2012.csv')
events_2013 = pd.read_csv('../input/Events_2013.csv')
events_2014 = pd.read_csv('../input/Events_2014.csv')
events_2015 = pd.read_csv('../input/Events_2015.csv')
events_2016 = pd.read_csv('../input/Events_2016.csv')
events_2017 = pd.read_csv('../input/Events_2017.csv')

#Import players
players_2010 = pd.read_csv('../input/Players_2010.csv')
players_2011 = pd.read_csv('../input/Players_2011.csv')
players_2012 = pd.read_csv('../input/Players_2012.csv')
players_2013 = pd.read_csv('../input/Players_2013.csv')
players_2014 = pd.read_csv('../input/Players_2014.csv')
players_2015 = pd.read_csv('../input/Players_2015.csv')
players_2016 = pd.read_csv('../input/Players_2016.csv')
players_2017 = pd.read_csv('../input/Players_2017.csv')

#Import Massey
massey = pd.read_csv('../input/MasseyOrdinals.csv')

#Import Other files
cities = pd.read_csv('../input/Cities.csv')
conferences = pd.read_csv('../input/Conferences.csv')
conf_tourney = pd.read_csv('../input/ConferenceTourneyGames.csv')
game_cities = pd.read_csv('../input/GameCities.csv')
tourney_results = pd.read_csv('../input/NCAATourneyCompactResults.csv')
detailed_tourney_results = pd.read_csv('../input/NCAATourneyDetailedResults.csv')
tourney_all = pd.read_csv('../input/NCAATourneySeedRoundSlots.csv')
tourney_seeds = pd.read_csv('../input/NCAATourneySeeds.csv')
tourney_slots = pd.read_csv('../input/NCAATourneySlots.csv')
reg_season = pd.read_csv('../input/RegularSeasonCompactResults.csv')
detailed_reg_season = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
seasons = pd.read_csv('../input/Seasons.csv')
second_tourney = pd.read_csv('../input/SecondaryTourneyCompactResults.csv')
detailed_second_tourney = pd.read_csv('../input/SecondaryTourneyTeams.csv')
team_coaches = pd.read_csv('../input/TeamCoaches.csv')
team_conferences = pd.read_csv('../input/TeamConferences.csv')
teams = pd.read_csv('../input/Teams.csv')
reg_season.head()
#Add the differential
reg_season['Diff'] = reg_season['WScore'] - reg_season['LScore'] 
reg_season['True_Outcome'] = np.where(reg_season['Diff'] > 5, 'Y','N')

#Example query
reg_season[(reg_season['Season'] == 2017) & (reg_season['True_Outcome'] == 'N')].head()

massey.head(5)
#What are the unique rankings? -- Looks like every team gets rated 1-351
massey['SystemName'].sort_values().unique()
""" Create a table of average of all end of year ratings """

#Drop duplicates from each rating system keep only max rating day
ratings = massey.sort_values('RankingDayNum', ascending = False).drop_duplicates(['Season','SystemName','TeamID'])

#testing - example of how to query on it 
ratings[(ratings['Season'] == 2017) & (ratings['SystemName'] == 'POM')].sort_values('OrdinalRank').head(5)
#Get the mean rating for each season team combo across all rating systems
ratings2 = ratings.groupby(['Season', 'TeamID'], as_index = False)['OrdinalRank'].mean()

ratings2.head(2)
#Merge with the team name table to get the team name 
ratings3 = ratings2.merge(teams, on = 'TeamID')
ratings3.head(2)
#Merge with the team name table to get the team name 

#testing - examples of how to query it
ratings3[ratings3['TeamName'] == 'Virginia'].sort_values('Season', ascending = False).head(3)

ratings3.Season.unique()
#only have ratings from 2003 - 2017
reg_season.head(2)

#Add winning team rank, merging on team ID and Season
schedule_ratings = reg_season[reg_season['Season']>2002] \
                .merge(ratings3[['Season','TeamID','OrdinalRank','TeamName']], \
                left_on = ['WTeamID','Season'], right_on = ['TeamID','Season'])

schedule_ratings.rename(columns = {'OrdinalRank':'WTeamRank', 'TeamName':'WTeamName'}, inplace = True)

#Add losing team rank, merging on team ID and Season
schedule_ratings2 = schedule_ratings.merge(ratings3[['Season','TeamID','OrdinalRank','TeamName']], \
                left_on = ['LTeamID','Season'], right_on = ['TeamID','Season'])

schedule_ratings2.rename(columns = {'OrdinalRank':'LTeamRank', 'TeamName':'LTeamName'}, inplace = True)

#Drop duplicate column
schedule_ratings2.drop(['TeamID_x','TeamID_y'], 1, inplace = True)

#Example - all UVA wins from 2017 where the differential is greater than 5
schedule_ratings2[(schedule_ratings2['Season'] == 2017) & (schedule_ratings2['WTeamName'] == 'Virginia') \
    & (schedule_ratings2['True_Outcome'] == 'Y')] \
    .sort_values('DayNum').head(3)

""" Lets use only the both team's seed and the avg. rating to predict winner using Naive Bayes """
tourney_seeds.head(2)
tourney_results.head(2)
#Step 1: Pull in the features (seeds and ratings)
tourney_results1 = tourney_results.merge(tourney_seeds, left_on = ['WTeamID','Season'] \
                                         , right_on = ['TeamID','Season'] )

tourney_results2 = tourney_results1.merge(tourney_seeds, left_on = ['LTeamID','Season'] \
                                         , right_on = ['TeamID','Season'] )

tourney_results3 = tourney_results2.merge(ratings3, left_on = ['WTeamID','Season'] \
                                         , right_on = ['TeamID','Season'] )

tourney_results4 = tourney_results3.merge(ratings3, left_on = ['LTeamID','Season'] \
                                         , right_on = ['TeamID','Season'] )

tourney_results5 = tourney_results4[['Season','WTeamID', 'LTeamID', 'TeamName_x', 'TeamName_y', \
                                    'Seed_x', 'Seed_y','OrdinalRank_x','OrdinalRank_y']] \
    .rename(columns = {'Seed_x':'WSeed','Seed_y': 'LSeed','TeamName_x' : 'WTeamName', \
                          'TeamName_y' : 'LTeamName', 'OrdinalRank_x' : 'WRank','OrdinalRank_y' : 'LRank'})

tourney_results5.head()
#selecting specific columns of tourney results
tourney_results6 = tourney_results5[['Season', 'WTeamID', 'LTeamID', 'WTeamName', 'LTeamName', \
                                    'WSeed', 'LSeed','WRank', 'LRank']]

#randomizing dataset
tourney_results6 = tourney_results6.sample(frac=1).reset_index(drop=True)

#splitting dataframe into two equal halves, so we can add multiple classifications for W/L
if (len(tourney_results6) % 2 > 0):
    print(len(tourney_results6))
    tourney_results6 = tourney_results6[:-1]
    print('trimming dataframe to even length so we can split')
    print(len(tourney_results6))
    
tourney_results6 = np.split(tourney_results6, 2)
#first half of new df
#0 is winners
tourney_results6[0]['Winner'] = 1
#renaming columns
tourney_results6[0].rename(columns = 
                       {
                            'LRank': 'Team2Rank',
                           'LSeed': 'Team2Seed',
                           'LTeamID': 'Team2ID',
                           'LTeamName': 'Team2Name',
                           'WRank': 'Team1Rank',
                           'WSeed': 'Team1Seed',
                           'WTeamID': 'Team1ID',
                           'WTeamName': 'Team1Name',
                       }, inplace = True)


#second half of new df
#0 is losers
tourney_results6[1]['Winner'] = 2
#renaming columns
tourney_results6[1].rename(columns = 
                       {
                            'LRank': 'Team1Rank',
                           'LSeed': 'Team1Seed',
                           'LTeamID': 'Team1ID',
                           'LTeamName': 'Team1Name',
                           'WRank': 'Team2Rank',
                           'WSeed': 'Team2Seed',
                           'WTeamID': 'Team2ID',
                           'WTeamName': 'Team2Name',
                       }, inplace = True)
#unioning two dataframe together
tourney_results7 = pd.concat([tourney_results6[0], tourney_results6[1]])
tourney_results7.head(2)
#Format seeds and rank

def removeletters(x):
    return re.sub('[^0-9]','', x)

tourney_results7['Team1Seed'] = tourney_results7['Team1Seed'].apply(lambda x: removeletters(x)).apply(int)
tourney_results7['Team2Seed'] = tourney_results7['Team2Seed'].apply(lambda x: removeletters(x)).apply(int)

#testing removal
tourney_results7.Team2Seed.value_counts().head(3)
features = ['Team1Rank', 'Team1Seed','Team2Rank','Team2Seed']

scaled_features = {}
for each in features:
    mean, std = tourney_results7[each].mean(), tourney_results7[each].std()
    scaled_features[each] = [mean, std]
    tourney_results7.loc[:, each] = (tourney_results7[each] - mean)/std


tourney_results7.head()
tourney_results8 = tourney_results7[['Team1Rank','Team2Rank','Team1Seed','Team2Seed','Winner']]
features = tourney_results8.values[:,:(tourney_results8.shape[1] -2)] #features sans target var
target = tourney_results8.values[:,(tourney_results8.shape[1] -1)] #ensure that target var is last col
features_train, features_test, target_train, target_test = train_test_split(features, \
                                                target, test_size = 0.2, random_state = 69)
target_test
clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)

accuracy_score(target_test, target_pred, normalize = True)

# Import confusion matrix functionality
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

# Create and format a confusion matrix
def conf_matrix(y_test, y_predict):

    # Create the raw confusion matrix
    conf = sk_confusion_matrix(y_test, y_predict)

    # Format the confusion matrix nicely
    conf = pd.DataFrame(data=conf)
    conf.columns.name = 'Predicted label'
    conf.index.name = 'Actual label'

    # Return the confusion matrix
    return conf
conf_matrix(target_test, target_pred)
from sklearn import metrics
print(metrics.classification_report(target_test, target_pred))
