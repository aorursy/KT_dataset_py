%matplotlib inline

import sqlite3 as lite

import pandas as pd

import numpy as np

from sklearn import preprocessing, cross_validation

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from matplotlib import style

import xml.etree.ElementTree as ET

style.use('ggplot')
database = '../input/database.sqlite'

conn = lite.connect(database)

#leagues = pd.read_sql_query("SELECT * from League", conn)

teams = pd.read_sql_query("SELECT * from Team", conn)

team_attributes = pd.read_sql_query("SELECT * from Team_Attributes", conn)

selected_teams = [9825, 8455, 8668, 8650, 8456, 10260, 10194, 8586]

team_attributes = team_attributes[team_attributes.team_api_id.isin(selected_teams)]

team_attributes = team_attributes[['team_api_id','buildUpPlaySpeed']]

teams = teams[teams.team_api_id.isin(selected_teams)]

matches = pd.read_sql_query("SELECT * from Match WHERE league_id=1729", conn)  

    
#selected teams consistenty in Premier League

                 ##ARS  CHE    EVE   LIV  MANC  MANU   STOKE  TOT

selected_teams = [9825, 8455, 8668, 8650, 8456, 10260, 10194, 8586]



teams = teams[teams.team_api_id.isin(selected_teams)]



matchesHome = matches[matches.home_team_api_id.isin(selected_teams)]

selectMatches = matchesHome[matchesHome.away_team_api_id.isin(selected_teams)]
shotson = selectMatches[['home_team_api_id','away_team_api_id','shoton','home_team_goal','away_team_goal']]

shotson = shotson[0:266]



homeTeamShotson = []

awayTeamShotson = []

      

for i in range(len(shotson)):



    #Column 'home_team_api_id' at index 7

    homeTeam = selectMatches.iloc[i,7]



    #XML Column 'Shoton' at index 78

    #Reading XML column from string

    tree = ET.fromstring(selectMatches.iloc[i,78])    

    lst = tree.findall('value')   

    numShoton = len(lst)

 

    #Counting home team shots on goal and away team shots on goal

    homeTeamCount = 0

    awayTeamCount = 0   

    

    #looping through values in Shoton

    for value in tree.findall("value"):

        #Getting team api id for each shot on goal

        team = value.find('team').text

        if(homeTeam==eval(team)):

            homeTeamCount+=1

        else:

            awayTeamCount+=1

            

    homeTeamShotson.append(homeTeamCount)

    awayTeamShotson.append(awayTeamCount)

        
shotson2 = selectMatches[['home_team_api_id','away_team_api_id','home_team_goal','away_team_goal']]

shotson2 = shotson2[0:266]



#away_team_shotson is index 0, home_team_shotson is index 1

home_and_away_shotson = pd.DataFrame({'home_team_shotson': homeTeamShotson, 'away_team_shotson': awayTeamShotson})
shotsonAgainstArsenal = []

goalsAgainstArsenal = []



for i in range(len(shotson2)):

    if(shotson.iloc[i,0] == 9825):

        goalsAgainstArsenal.append(shotson2.iloc[i,3])

        shotsonAgainstArsenal.append(home_and_away_shotson.iloc[i,0])

    elif(shotson.iloc[i,1] == 9825):

        goalsAgainstArsenal.append(shotson2.iloc[i,2])

        shotsonAgainstArsenal.append(home_and_away_shotson.iloc[i,1])

        

againstArsenal = pd.DataFrame({'shotson': shotsonAgainstArsenal, 'goals': goalsAgainstArsenal})        

againstArsenal = againstArsenal[againstArsenal.goals <= againstArsenal.shotson]
shotsonAgainstChelsea = []

goalsAgainstChelsea = []

#print(shotson)

#print(home_and_away_shotson)



#print(shotson2)





for i in range(len(shotson2)):

    if(shotson.iloc[i,0] == 8455):

        goalsAgainstChelsea.append(shotson2.iloc[i,3])

        shotsonAgainstChelsea.append(home_and_away_shotson.iloc[i,0])

    elif(shotson.iloc[i,1] == 8455):

        goalsAgainstChelsea.append(shotson2.iloc[i,2])

        shotsonAgainstChelsea.append(home_and_away_shotson.iloc[i,1])

againstChelsea = pd.DataFrame({'shotson': shotsonAgainstChelsea, 'goals': goalsAgainstChelsea})

againstChelsea.ix[~(againstChelsea['goals'] > againstChelsea['shotson'])]
x = np.array(againstArsenal['shotson'])

y = np.array(againstArsenal['goals'])



x_train, x_test, y_train, y_test= cross_validation.train_test_split(x, y, test_size=0.3)



clf = LinearRegression(n_jobs = -1)

clf.fit(x_train.reshape(len(x_train),1), y_train.reshape(len(y_train),1))

accuracy = accuracy_score(x_test.reshape(len(x_test),1), y_test.reshape(len(y_test),1))

print('Coefficients: \n', clf.coef_)

print(accuracy)
plt.scatter(x_train,y_train,color='blue')

plt.plot(x_test,clf.predict(x_test.reshape(len(x_test),1)),color = 'red',linewidth = 3)
x = np.array(againstChelsea['shotson'])

y = np.array(againstChelsea['goals'])



x_train, x_test, y_train, y_test= cross_validation.train_test_split(x, y, test_size=0.3)



clf = LinearRegression(n_jobs = -1)

clf.fit(x_train.reshape(len(x_train),1), y_train.reshape(len(y_train),1))

accuracy = clf.score(x_test.reshape(len(x_test),1), y_test.reshape(len(y_test),1))

print('Coefficients: \n', clf.coef_)

print(accuracy)
plt.scatter(x_train,y_train,color='blue')

plt.plot(x_test,clf.predict(x_test.reshape(len(x_test),1)),color = 'red',linewidth = 3)
shotsonAgainstEverton = []

goalsAgainstEverton = []



for i in range(len(shotson2)):

    if(shotson.iloc[i,0] == 8668):

        goalsAgainstEverton.append(shotson2.iloc[i,3])

        shotsonAgainstEverton.append(home_and_away_shotson.iloc[i,0])

    elif(shotson.iloc[i,1] == 8668):

        goalsAgainstEverton.append(shotson2.iloc[i,2])

        shotsonAgainstEverton.append(home_and_away_shotson.iloc[i,1])

againstEverton = pd.DataFrame({'shotson': shotsonAgainstEverton, 'goals': goalsAgainstEverton})



x = np.array(againstEverton['shotson'])

y = np.array(againstEverton['goals'])



x_train, x_test, y_train, y_test= cross_validation.train_test_split(x, y, test_size=0.3)



clf = LinearRegression(n_jobs = -1)

clf.fit(x_train.reshape(len(x_train),1), y_train.reshape(len(y_train),1))

accuracy = clf.score(x_test.reshape(len(x_test),1), y_test.reshape(len(y_test),1))

print('Coefficients: \n', clf.coef_)

print(accuracy)
plt.scatter(x_train,y_train,color='blue')

plt.plot(x_test,clf.predict(x_test.reshape(len(x_test),1)),color = 'red',linewidth = 3)
shotsonAgainstLiverpool = []

goalsAgainstLiverpool = []



for i in range(len(shotson2)):

    if(shotson.iloc[i,0] == 8650):

        goalsAgainstLiverpool.append(shotson2.iloc[i,3])

        shotsonAgainstLiverpool.append(home_and_away_shotson.iloc[i,0])

    elif(shotson.iloc[i,1] ==8650):

        goalsAgainstLiverpool.append(shotson2.iloc[i,2])

        shotsonAgainstLiverpool.append(home_and_away_shotson.iloc[i,1])

        

againstLiverpool = pd.DataFrame({'shotson': shotsonAgainstLiverpool, 'goals': goalsAgainstLiverpool})        

againstLiverpool.ix[~(againstLiverpool['goals'] > againstLiverpool['shotson'])]



plt.scatter(x_train,y_train,color='blue')

plt.plot(x_test,clf.predict(x_test.reshape(len(x_test),1)),color = 'red',linewidth = 3)
shotsonAgainstManCity = []

goalsAgainstManCity = []



for i in range(len(shotson2)):

    if(shotson.iloc[i,0] == 8456):

        goalsAgainstManCity.append(shotson2.iloc[i,3])

        shotsonAgainstManCity.append(home_and_away_shotson.iloc[i,0])

    elif(shotson.iloc[i,1] ==8456):

        goalsAgainstManCity.append(shotson2.iloc[i,2])

        shotsonAgainstManCity.append(home_and_away_shotson.iloc[i,1])

        

againstManCity = pd.DataFrame({'shotson': shotsonAgainstManCity, 'goals': goalsAgainstManCity})        

againstManCity.ix[~(againstManCity['goals'] > againstManCity['shotson'])]



plt.scatter(x_train,y_train,color='blue')

plt.plot(x_test,clf.predict(x_test.reshape(len(x_test),1)),color = 'red',linewidth = 3)
shotsonAgainstManCity = []

goalsAgainstManCity = []



for i in range(len(shotson2)):

    if(shotson.iloc[i,0] == 8456):

        goalsAgainstManCity.append(shotson2.iloc[i,3])

        shotsonAgainstManCity.append(home_and_away_shotson.iloc[i,0])

    elif(shotson.iloc[i,1] ==8456):

        goalsAgainstManCity.append(shotson2.iloc[i,2])

        shotsonAgainstManCity.append(home_and_away_shotson.iloc[i,1])

        

againstManCity = pd.DataFrame({'shotson': shotsonAgainstManCity, 'goals': goalsAgainstManCity})        

againstManCity.ix[~(againstManCity['goals'] > againstManCity['shotson'])]



plt.scatter(x_train,y_train,color='blue')

plt.plot(x_test,clf.predict(x_test.reshape(len(x_test),1)),color = 'red',linewidth = 3)