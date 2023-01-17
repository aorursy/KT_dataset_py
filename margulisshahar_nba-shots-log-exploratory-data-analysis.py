# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoup as bs
from IPython.display import Image 
import requests

%matplotlib inline  
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read csv files

dfShotsLog = pd.read_csv('../input/nba-shot-logs/shot_logs.csv')
#remove unnecessary columns
dfShotsLog=dfShotsLog.drop(['MATCHUP','CLOSEST_DEFENDER_PLAYER_ID','PTS','player_id'],axis=1)
players_stats_2014_2015=pd.read_csv('../input/nba-players-stats-20142015/players_stats.csv')# file with players details

#remove unnecessary columns
players_stats_2014_2015=players_stats_2014_2015.drop([ 'Games Played', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA',
'3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
'TOV', 'PF', 'EFF', 'AST/TOV', 'STL/TOV', 'Age','Birth_Place',
'Birthdate', 'Collage','Experience','Team','Weight',
'BMI'],axis=1)


#clean data
# function that gets input string (represents first name and last name )and indecation if reverse first name and last name
#return formated name
def formatPlayersName(inputString,isReverse=False):
    splitString=inputString.split()
    if(len(splitString)==1):
        return cleanStr(splitString[0],[".","-","'"])
    if(isReverse==True):
        newStr=cleanStr(splitString[1],[".","-",",","'"])+" "+cleanStr(splitString[0],[".","-",",","'"])
    else:
        newStr=cleanStr(splitString[0],[".","-",",","'"])+" "+cleanStr(splitString[1],[".","-",",","'"])
    
    return newStr
    
#help function that gets input string and list of chars to remove 
#return string with listChar removed
def cleanStr(inputString,listChar):
    newStr=inputString
    for char in listChar:
        newStr = newStr.replace(char, "")
    return newStr.lower()

# formating names in all columns that represent basket ball players name (we will use the names as key to merge data frame)
players_stats_2014_2015['Name'] = players_stats_2014_2015.apply(lambda x: formatPlayersName(x['Name']), axis=1)
dfShotsLog['player_name'] = dfShotsLog.apply(lambda x: formatPlayersName(x['player_name']), axis=1)
dfShotsLog['CLOSEST_DEFENDER'] = dfShotsLog.apply(lambda x: formatPlayersName(x['CLOSEST_DEFENDER'],True), axis=1)
# merging data 2 files based on players name (the data contain unique players names)
mergedWithShooterName=pd.merge(dfShotsLog, players_stats_2014_2015, how='inner', left_on=['player_name'], right_on=['Name'])
mergedWithDefenderName=pd.merge(mergedWithShooterName, players_stats_2014_2015, how='inner', left_on=['CLOSEST_DEFENDER'], right_on=['Name'])

#renaming data frame columns
mergedWithDefenderName.columns = ['GAME_ID','SHOOTERS_COURT_LOCATION', 'SHOOTERS_TEAM_FINAL_RESULT', 'SHOOTERS_FINAL_RESULT_DIFF', 'SHOT_NUMBER_IN_GAME', 'PERIOD',
       'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME','SHOT_DIST', 'PTS_TYPE',
       'SHOT_RESULT', 'DEFENDER_NAME','CLOSE_DEF_DIST','SHOT_RESULT_INT',
     'SHOOTER_NAME', 'SHOOTER_NAME_FROM_MERGE', 'SHOOTERS_HEIGHT',
    'SHOOTERS_POS', 'DEFENDER_NAME_FROM_MERGE', 'DEFENDER_HEIGHT',
      'DEFENDER_POS']
#remove unnecessary columns

dfWithRelevantColumns=mergedWithDefenderName.drop(['SHOOTER_NAME_FROM_MERGE','DEFENDER_NAME_FROM_MERGE'],axis=1)
leftHanded = pd.read_fwf('../input/left-handed-players/leftPlayers.txt',headers=True)
players=leftHanded['Names'].tolist()
players=[x.lower() for x in players if type(x) is str]
#adding to data columns that indicates if shooter/defender is left handed based .txt file with left handed players
dfWithRelevantColumns['SHOOTER_STRONG_HAND'] = dfWithRelevantColumns.apply(lambda x: 'LEFT' if x['SHOOTER_NAME'] in players else 'RIGHT', axis=1)
dfWithRelevantColumns['DEFENDER_STRONG_HAND'] = dfWithRelevantColumns.apply(lambda x: 'LEFT' if x['DEFENDER_NAME'] in players  else 'RIGHT', axis=1)

#reindex data frame with GAME_ID for performance Considerations for the following function
gameIdAsIndex=dfWithRelevantColumns.set_index('GAME_ID',drop=False)

#function that gets game_id data frame a record and break condition
#returns Sequence of basket made before current shot in current game if breakCondition=0
#returns Sequence of basket miss before current shot in current game if breakCondition=1

def calculateShotSequence(dfByGameId,record,breakCondition=0):
    if(record["SHOT_NUMBER_IN_GAME"]==1):
        return -1
    sortedDf=dfByGameId[(dfByGameId["SHOOTER_NAME"]==record["SHOOTER_NAME"]) & (dfByGameId["SHOT_NUMBER_IN_GAME"]<record["SHOT_NUMBER_IN_GAME"])][["SHOT_NUMBER_IN_GAME","SHOT_RESULT_INT"]].sort_values("SHOT_NUMBER_IN_GAME", ascending=False)
    count=0
    for index, row in sortedDf.iterrows():
        if(row["SHOT_RESULT_INT"]==breakCondition):
            break
        count=count+1
    return count

#adding column with information of Sequence of basket made before current shot in current game
gameIdAsIndex['SHOTS_IN_ROW'] = gameIdAsIndex.apply(lambda x: calculateShotSequence(gameIdAsIndex.loc[x["GAME_ID"]],x), axis=1)
#adding columns with information of Sequence of basket miss before current shot in current game
gameIdAsIndex['MISS_IN_ROW'] = gameIdAsIndex.apply(lambda x: calculateShotSequence(gameIdAsIndex.loc[x["GAME_ID"]],x,1), axis=1)


#sorting data frame by ["GAME_ID","SHOOTERS_NAME","SHOT_NUMBER_IN_GAME"]
gameIdAsIndex=gameIdAsIndex.sort_values(["GAME_ID","SHOOTER_NAME","SHOT_NUMBER_IN_GAME"])
#resteing index
dfReIndexef=gameIdAsIndex.reset_index(drop=True)
#saving data frame
dfReIndexef.to_csv("shotsLog.csv",index=False)

#reading df
shotsLogDf= pd.read_csv('shotsLog.csv')
columnsWithNa=shotsLogDf.columns[shotsLogDf.isna().any()].tolist()
#removing rows where data is na
shotsLogDf = shotsLogDf.dropna(subset=columnsWithNa)
#saving data frame
shotsLogDf.to_csv("shotsLogFile.csv",index=False)

shotsLogDf= pd.read_csv('shotsLogFile.csv')
#convert feet to meter
shotsLogDf['SHOT_DIST'] = shotsLogDf.SHOT_DIST.apply(lambda x: x*0.3048)
#convert feet to meter
shotsLogDf['CLOSE_DEF_DIST'] = shotsLogDf.CLOSE_DEF_DIST.apply(lambda x:x*0.3048)
#converting time to seconds
shotsLogDf['GAME_CLOCK'] = shotsLogDf.GAME_CLOCK.apply(lambda x:int(x.split(":")[0])*60+int(x.split(":")[1]))
# creating columns with HEIGHT_DIFF
shotsLogDf['HEIGHT_DIFF']=shotsLogDf['SHOOTERS_HEIGHT']-shotsLogDf['DEFENDER_HEIGHT']

#function that returns if shot is money time (default is if shot was taken 150 seconds to end of game and final result is under 5)
def isMoneyTime(record,finalScoreDiff=3,secondsToEnd=120):
    if((record["PERIOD"]>4) | (record["PERIOD"]==4 & abs(int(record["SHOOTERS_FINAL_RESULT_DIFF"]))<=finalScoreDiff & record["GAME_CLOCK"]<=secondsToEnd)):
        return 1
    else:
        return 0
# creating columns with IS_MONEY_TIME value
shotsLogDf['IS_MONEY_TIME'] = shotsLogDf.apply(lambda x: isMoneyTime(x), axis=1)
shotsLogDf.to_csv("shotsLogData.csv",index=False)


shotsLogData= pd.read_csv('shotsLogData.csv')
#function that gets integer and number to round and returns rounded number
#example:roundNumber(5.3,0.5) will return 5.5  ,roundNumber(13,5) will return 5
def roundNumber(x,n):
    return round((1/float(n))*float(x))*n
def isHomeTeanWin(x):
    if((x.SHOOTERS_COURT_LOCATION=="H") & (x.SHOOTERS_FINAL_RESULT_DIFF>0)):
        return 1
    elif ((x.SHOOTERS_COURT_LOCATION=="A") & (x.SHOOTERS_FINAL_RESULT_DIFF<0)):
        return 1
    else:
        return 0
def isMoneyTimeGame(x,scoreDiff=5):
    if(abs(int(x["SHOOTERS_FINAL_RESULT_DIFF"]))<scoreDiff):
        return True
    else:
        return False             
    
    
#adding labeled columns
shotsLogData['HEIGHT_DIFF'] = shotsLogData.HEIGHT_DIFF.apply(lambda x: roundNumber(x,5))
shotsLogData['SHOT_CLOCK'] = shotsLogData.SHOT_CLOCK.apply(lambda x: roundNumber(x,0.5))
shotsLogData['DRIBBLES'] = shotsLogData.DRIBBLES.apply(lambda x: roundNumber(x,1))
shotsLogData['TOUCH_TIME'] = shotsLogData.TOUCH_TIME.apply(lambda x: roundNumber(x,1))
shotsLogData['SHOT_DIST'] = shotsLogData.SHOT_DIST.apply(lambda x: roundNumber(x,0.5))
shotsLogData['CLOSE_DEF_DIST'] = shotsLogData.CLOSE_DEF_DIST.apply(lambda x: roundNumber(x,0.5))
shotsLogData['SHOOTERS_TEAM_FINAL_RESULT'] = shotsLogData.SHOOTERS_TEAM_FINAL_RESULT.apply(lambda x: 1 if x=='W' else 0)
shotsLogData['IS_HOME_TEAM_WIN']=shotsLogData.apply(lambda x: isHomeTeanWin(x), axis=1)
shotsLogData['IS_MONEY_TIME_GAME']=shotsLogData.apply(lambda x: isMoneyTimeGame(x), axis=1)
#creating df per match includes game id,indication if home team won,indication if money time game
dfByMatch=shotsLogData.groupby(['GAME_ID', 'IS_HOME_TEAM_WIN','IS_MONEY_TIME_GAME']).count().reset_index()[['GAME_ID', 'IS_HOME_TEAM_WIN','IS_MONEY_TIME_GAME']]
dfByMatch.to_csv("dfByMatch.csv",index=False)

#renaming data frame columns
shotsLogData=shotsLogData.drop(['GAME_ID','SHOOTERS_FINAL_RESULT_DIFF','GAME_CLOCK','SHOT_NUMBER_IN_GAME','SHOOTERS_HEIGHT','DEFENDER_HEIGHT','SHOT_RESULT'],axis=1)
shotsLogData.to_csv("shotsLogData.csv",index=False)
#reading shots log df and games df csv to df
shotsLogData= pd.read_csv('shotsLogData.csv')#, index_col='PassengerId')
dfByGame= pd.read_csv('dfByMatch.csv')#, index_col='PassengerId')
shotsLogData.describe().T

numericColumnsDf = shotsLogData.select_dtypes(include=[np.number])
plt.rcParams['figure.figsize'] = (10.0, 8.0)
sns.set_style()
corr = numericColumnsDf.corr()
sns.heatmap(corr,cmap="RdYlBu",vmin=-1,vmax=1)
plt.savefig('corrMap.png')


#show box plot of 'strange behaviour' columns)
shotsLogData.boxplot(column=['TOUCH_TIME'])
plt.savefig('BoxPlotTouchTime.png')
#show number of records by dribbles

shotsLogData.groupby(['DRIBBLES']).size()



# removing shots that are errors/outliers shots
cleanedDataShots=shotsLogData[(shotsLogData['SHOT_DIST']<=9) & (shotsLogData['DRIBBLES']<=21) & (shotsLogData['CLOSE_DEF_DIST']<=7) & (shotsLogData['TOUCH_TIME']>=0) & (shotsLogData['TOUCH_TIME']<18) & (shotsLogData['PERIOD']<=5) & (shotsLogData['SHOT_CLOCK']<=20)]

cleanedDataShots.to_csv("cleanedDataShots.csv",index=False)
cleanedDataShots= pd.read_csv('cleanedDataShots.csv')



ax=cleanedDataShots.groupby(['SHOOTERS_COURT_LOCATION','IS_MONEY_TIME']).mean()['SHOT_RESULT_INT'].unstack().plot.bar(yticks=np.linspace(0,0.5,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by court location and Money Time')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageByCourtLocationMoneyTime.png')


ax=cleanedDataShots.groupby(['SHOOTERS_COURT_LOCATION']).mean()['SHOT_RESULT_INT'].plot.bar(yticks=np.linspace(0,0.5,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by court location')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageByCourtLocation.png')


ax=cleanedDataShots.groupby(['IS_MONEY_TIME']).mean()['SHOT_RESULT_INT'].plot.bar(yticks=np.linspace(0,0.5,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by Money Time')
ax.set_ylabel('Shot Percenteage')
ax.set_xticklabels(['Money Time=No','Money Time=Yes'])
plt.savefig('ShotPercenteageByMoneyTime.png')


ax=cleanedDataShots.groupby(['IS_MONEY_TIME','SHOOTERS_POS']).mean()['SHOT_RESULT_INT'].unstack().plot.bar(yticks=np.linspace(0,0.5,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by Position and Money Time')
ax.set_ylabel('Shot Percenteage')
ax.set_xticklabels(['Money Time=No','Money Time=Yes'])
plt.savefig('ShotPercenteageByPositionMoneyTime.png')

ax=cleanedDataShots.groupby(['SHOOTERS_TEAM_FINAL_RESULT']).mean()['SHOT_RESULT_INT'].plot.bar(yticks=np.linspace(0,0.5,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by win/lose')
ax.set_ylabel('Shot Percenteage')
ax.set_xticklabels(['Lost','Won'])
plt.savefig('ShotPercenteageByWinLose.png')

ax=cleanedDataShots.groupby(['PERIOD','SHOOTERS_POS']).mean()['SHOT_RESULT_INT'].unstack().plot(xticks=cleanedDataShots['PERIOD'].unique(),yticks=np.linspace(0.3,0.6,20),figsize=(15,7),rot=0)
ax.set_title('Shot Percenteage by period and position')
plt.savefig('ShotPercenteageByPeriodPosition.png')


ax=cleanedDataShots.groupby(['SHOOTERS_POS']).mean()['SHOT_RESULT_INT'].plot.bar(yticks=np.linspace(0,0.6,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by position')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageByPosition.png')

ax=cleanedDataShots.groupby(['DRIBBLES']).mean()['SHOT_RESULT_INT'].plot(xticks=cleanedDataShots['DRIBBLES'].unique(),yticks=np.linspace(0,0.7,20),figsize=(15,7),rot=0)
ax.set_title('Shot Percenteage by dribbles ')
plt.savefig('ShotPercenteageByDribbles.png')

ax=cleanedDataShots.groupby(['SHOT_CLOCK']).mean()['SHOT_RESULT_INT'].plot(xticks=cleanedDataShots['SHOT_CLOCK'].unique().round(),yticks=np.linspace(0,0.7,20),figsize=(15,7),rot=0)
ax.set_title('Shot Percenteage by 24 shot clock')
plt.savefig('ShotPercenteageBy24Clock.png')

ax=cleanedDataShots.groupby(['TOUCH_TIME']).mean()['SHOT_RESULT_INT'].plot(xticks=cleanedDataShots['TOUCH_TIME'].unique(),yticks=np.linspace(0,0.7,20),figsize=(15,7),rot=0)
ax.set_title('Shot Percenteage by touch time ')
plt.savefig('ShotPercenteageTouchTime.png')

cleanedDataShots= pd.read_csv('cleanedDataShots.csv')#, index_col='PassengerId')
ax=cleanedDataShots.groupby(['SHOT_DIST']).mean()['SHOT_RESULT_INT'].plot.bar(yticks=np.linspace(0,0.7,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by shot dist')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageShotDist.png')


cleanedDataShots= pd.read_csv('cleanedDataShots.csv')#, index_col='PassengerId')


ax=cleanedDataShots.groupby(['CLOSE_DEF_DIST']).mean()['SHOT_RESULT_INT'].plot.bar(yticks=np.linspace(0,0.7,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by defender dist')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageDefenderDist.png')

ax=cleanedDataShots.groupby(['SHOT_DIST','SHOOTERS_POS']).mean()['SHOT_RESULT_INT'].unstack().plot.bar(stacked=True,xticks=cleanedDataShots['SHOT_DIST'].unique(),yticks=np.linspace(0,3.5,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage shot dist and position')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageShotDistPosition.png')

ax=cleanedDataShots.groupby(['SHOOTER_STRONG_HAND']).mean()['SHOT_RESULT_INT'].plot.bar(yticks=np.linspace(0,0.5,10),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by shooter strong hand')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageByShooterStrongHand.png')


ax=cleanedDataShots.groupby(['DEFENDER_STRONG_HAND']).mean()['SHOT_RESULT_INT'].plot.bar(yticks=np.linspace(0,0.5,10),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by defenders strong hand')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageByDefendersStrongHand.png')


ax=cleanedDataShots.groupby(['SHOTS_IN_ROW','SHOOTERS_POS']).mean()['SHOT_RESULT_INT'].unstack().plot(xticks=cleanedDataShots['SHOTS_IN_ROW'].unique(),yticks=np.linspace(0,0.7,20),figsize=(15,7),rot=0)
ax.set_title('Shot Percenteage position and shots made in a row')
ax.set_xlim((-1, 7))
plt.savefig('ShotPercenteagePositionAndShotsMadeInRow.png')

ax=cleanedDataShots.groupby(['MISS_IN_ROW','SHOOTERS_POS']).mean()['SHOT_RESULT_INT'].unstack().plot(xticks=cleanedDataShots['SHOTS_IN_ROW'].unique(),yticks=np.linspace(0,0.7,10),figsize=(15,7),rot=0)
ax.set_title('Shot Percenteage position and shots miss in a row')
ax.set_xlim((-1, 7))
plt.savefig('ShotPercenteagePositionAndShotsMissInRow.png')


ax=cleanedDataShots.groupby(['HEIGHT_DIFF','SHOOTERS_POS']).mean()['SHOT_RESULT_INT'].unstack().plot.bar(stacked=True,xticks=cleanedDataShots['SHOT_DIST'].unique(),yticks=np.linspace(0,3.5,20),figsize=(12, 5),rot=0)
ax.set_title('Shot Percenteage by Height diff and position')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageByHeightDiffPosition.png')

fig = plt.figure()
ax = fig.gca()
ax.hist(cleanedDataShots['SHOT_DIST'], bins=40, color='blue')
ax.set_xticks(cleanedDataShots['SHOT_DIST'].unique().round())
ax.set_yticks(np.linspace(0,13000,10))


ax.set_title('Histogram of shots dist')
ax.set_xlabel('Shot Dist [meters]')
plt.savefig('HistogramShotsDist.png')

dfByGame= pd.read_csv('dfByMatch.csv')

ax=dfByGame.groupby(['IS_MONEY_TIME_GAME']).mean()['IS_HOME_TEAM_WIN'].plot.bar(yticks=np.linspace(0,0.6,20),figsize=(12, 5), rot=0)
ax.set_xticklabels(['Money Time Game=False','Money Time Game=True'])
ax.set_title('Percenteage of Home Team Win by Money Time')
ax.set_ylabel('Percenteage of Home Team')
plt.savefig('PercenteageHomeTeamWinMoneyTime.png')
playersData=cleanedDataShots[(cleanedDataShots.SHOOTER_NAME=="kobe bryant") | (cleanedDataShots.SHOOTER_NAME=="lebron james") ]
playersData.to_csv("playersData.csv",index=False)
playersData= pd.read_csv('playersData.csv')
ax=playersData.groupby(['SHOOTER_NAME']).mean()['SHOT_RESULT_INT'].plot.bar(yticks=np.linspace(0,0.5,20),figsize=(12, 5),rot=0)
ax.set_title(' kobe bryant vs lebron james')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageByPlayer.png')
ax=playersData.groupby(['SHOOTER_NAME','IS_MONEY_TIME']).mean()['SHOT_RESULT_INT'].unstack().plot.bar(yticks=np.linspace(0,0.5,20),figsize=(12, 5),rot=0)
ax.set_title(' kobe bryant vs lebron james Money Time')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageByPlayerMoneyTime.png')
ax=playersData.groupby(['SHOOTER_NAME','SHOOTERS_COURT_LOCATION']).mean()['SHOT_RESULT_INT'].unstack().plot.bar(yticks=np.linspace(0,0.5,20),figsize=(12, 5),rot=0)
ax.set_title(' kobe bryant vs lebron court')
ax.set_ylabel('Shot Percenteage')
plt.savefig('ShotPercenteageByPlayerCourt.png')
Image(filename='../input/image-files-for-presentation/nba.PNG')

Image(filename='../input/image-files-for-presentation/nba_court.jpg')
Image(filename='../input/image-files-for-presentation/basket_ball_positions.PNG')
Image(filename='../input/image-files-for-presentation/shot_clock.PNG')
Image(filename='../input/image-files-for-presentation/michael_jordan.PNG')
Image(filename='../input/image-files-for-presentation/record_example.PNG')
Image(filename='../input/image-files-for-presentation/naValues.PNG')
Image(filename='../input/image-files-for-presentation/desc_plot.png')

Image(filename='../input/image-files-for-presentation/BoxPlotTouchTime.png')

Image(filename='../input/image-files-for-presentation/DribblesStatistic.png')

Image(filename='../input/image-files-for-presentation/corrMap.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByCourtLocation.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByMoneyTime.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByCourtLocationMoneyTime.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByPositionMoneyTime.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByPeriodPosition.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByPosition.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByDribbles.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageBy24Clock.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageTouchTime.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageShotDist.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageDefenderDist.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageShotDistPosition.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByShooterStrongHand.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByDefendersStrongHand.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteagePositionAndShotsMadeInRow.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteagePositionAndShotsMissInRow.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByHeightDiffPosition.png')

Image(filename='../input/image-files-for-presentation/HistogramShotsDist.png')

Image(filename='../input/image-files-for-presentation/PercenteageHomeTeamWinMoneyTime.png')

Image(filename='../input/image-files-for-presentation/lebronVsKobe.PNG')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByPlayer.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByPlayerMoneyTime.png')

Image(filename='../input/image-files-for-presentation/ShotPercenteageByPlayerCourt.png')

