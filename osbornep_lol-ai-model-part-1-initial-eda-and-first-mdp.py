import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import math
from scipy.stats import kendalltau

import timeit

import warnings
warnings.filterwarnings('ignore')
#kills = pd.read_csv('C:\\Users\\Phil\\Documents\\LoL Model\\kills.csv')
#matchinfo = pd.read_csv('C:\\Users\\Phil\\Documents\\LoL Model\\matchinfo.csv')
#monsters = pd.read_csv('C:\\Users\\Phil\\Documents\\LoL Model\\monsters.csv')
#structures = pd.read_csv('C:\\Users\\Phil\\Documents\\LoL Model\\structures.csv')

kills = pd.read_csv('../input/kills.csv')
matchinfo = pd.read_csv('../input/matchinfo.csv')
monsters = pd.read_csv('../input/monsters.csv')
structures = pd.read_csv('../input/structures.csv')

matchinfo.head()
# Add ID column based on last 16 digits in match address for simpler matching

matchinfo['id'] = matchinfo['Address'].astype(str).str[-16:]
kills['id'] = kills['Address'].astype(str).str[-16:]
monsters['id'] = monsters['Address'].astype(str).str[-16:]
structures['id'] = structures['Address'].astype(str).str[-16:]
matchinfo.head()
# Dragon became multiple types in patch v6.9 (http://leagueoflegends.wikia.com/wiki/V6.9) 
# so we remove and games before this change occured and only use games with the new dragon system
monsters['Type'].unique()
old_dragon_id = monsters[ monsters['Type']=="DRAGON"]['id'].unique()
old_dragon_id
monsters = monsters[ ~monsters['id'].isin(old_dragon_id)]
monsters[monsters['Type']=="DRAGON"]
# Again remove old games, we have some missing values (probably for other events) so remove this
# Create a column for the minute in which the kill took place
# Reassign the team column to a simpler Red/Blue accordingly for matching with other tables

kills = kills[ ~kills['id'].isin(old_dragon_id)]
kills = kills[ kills['Time']>0]

kills['Minute'] = kills['Time'].astype(int)

kills['Team'] = np.where( kills['Team']=="rKills","Red","Blue")
kills.head()
# For the Kills table, we need decided to group by the minute in which the kills took place and averaged 
# the time of the kills which we use later for the order of events

f = {'Time':['mean','count']}

killsGrouped = kills.groupby( ['id','Team','Minute'] ).agg(f).reset_index()
killsGrouped.columns = ['id','Team','Minute','Time Avg','Count']
killsGrouped = killsGrouped.sort_values(by=['id','Minute'])
killsGrouped.head(13)
# Repeat similar steps for the structures table

structures = structures[ ~structures['id'].isin(old_dragon_id)]
structures = structures[ structures['Time']>0]

structures['Minute'] = structures['Time'].astype(int)
structures['Team'] = np.where(structures['Team']=="bTowers","Blue",
                        np.where(structures['Team']=="binhibs","Blue","Red"))
structures2 = structures.sort_values(by=['id','Minute'])
structures2.head(13)
# Merge the two together
kills_structures = killsGrouped.merge(structures2[['id','Minute','Team','Time','Lane','Type']],
                                      on=['id','Minute','Team'],how='outer')
kills_structures.head(20)
# Again repeat same steps, we also map the types of dragon to a simpler 'Dragon' label

monsters = monsters[ ~monsters['id'].isin(old_dragon_id)]
monsters['Type2'] = np.where( monsters['Type']=="FIRE_DRAGON", "DRAGON",
                    np.where( monsters['Type']=="EARTH_DRAGON","DRAGON",
                    np.where( monsters['Type']=="WATER_DRAGON","DRAGON",       
                    np.where( monsters['Type']=="AIR_DRAGON","DRAGON",   
                             monsters['Type']))))

monsters = monsters[ monsters['Time']>0]

monsters['Minute'] = monsters['Time'].astype(int)

monsters['Team'] = np.where( monsters['Team']=="bDragons","Blue",
                   np.where( monsters['Team']=="bHeralds","Blue",
                   np.where( monsters['Team']=="bBarons", "Blue", 
                           "Red")))



monsters.head()
# Merge the monsters to our previously merged table
# This provides us with a table that has each event seperated by columns depending on what type of event it was
kills_structures_monsters = kills_structures.merge(monsters[['id','Minute','Team','Time','Type2']], on=['id','Minute'],how='outer')
kills_structures_monsters = kills_structures_monsters.sort_values(by=['id','Minute'])
kills_structures_monsters.head(5)
# Although this is a good start, information is repeated on the rows if multiple 
# events occured in the same minute.
#
# Therefore, I decided to let each event have its own row by stacking the tables
# on top of one another. We then add a more detailed time column and sort by this 
# so we know exactly which event came first (allowing for some errors with kill time
# being averaged).


stackedData = killsGrouped.append(structures2)
stackedData = stackedData.append(monsters[['id','Address','Team','Minute','Time','Type2']])

stackedData['Time2'] = stackedData['Time'].fillna(stackedData['Time Avg'])

stackedData = stackedData.sort_values(by=['id','Time2'])

stackedData['EventNum'] = stackedData.groupby('id').cumcount()+1

stackedData = stackedData[['id','EventNum','Team','Minute','Time2','Count','Type','Lane','Type2']]

stackedData.columns = ['id','EventNum','Team','Minute','Time','KillCount','StructType','StructLane','Monster']

stackedData.head(5)
# We then add an 'Event' column to merge the columns into one, where kills are now
# simple labelled as 'KILLS'

stackedData['Event'] = np.where(stackedData['KillCount']>0,"KILLS",None)
stackedData['Event'] = stackedData['Event'].fillna(stackedData['StructType'])
stackedData['Event'] = stackedData['Event'].fillna(stackedData['Monster'])

                        

stackedData.head(10)
stackedData['Event'].unique()

NumEventAnalysis = stackedData[['id','EventNum']].groupby('id').max().reset_index()

NumEventAnalysis2 = NumEventAnalysis.groupby('EventNum').count().reset_index()

NumEventAnalysis2.head()
plt.bar(NumEventAnalysis2['EventNum'],NumEventAnalysis2['id'] ,alpha=0.3)
plt.plot(NumEventAnalysis2['EventNum'],NumEventAnalysis2['id'])
plt.title('Distribution of Number of Events in Each Match (EXACT)')
plt.xlim(0,100)
plt.xlabel("Number of Events")
plt.ylabel("Number of Matches")
plt.show()
sns.distplot(NumEventAnalysis['EventNum'],bins=65)
plt.title('Distribution of Number of Events in Each Match (NORMAL DIST)')
plt.xlim(0,100)
plt.xlabel("Number of Events")
plt.ylabel("Number of Matches")
plt.show()
print("The max number of events for any team in a single game is:",NumEventAnalysis['EventNum'].max())
print("The min number of events for any team in a single game is:",NumEventAnalysis['EventNum'].min())
# We then create a table with just the unique match ids that we will use to merge our tables to shortly
matchevents = pd.DataFrame(stackedData['id'].unique())
matchevents.columns = ['id']

matchevents.head()

# WARNING: Takes a while to run

# This cell has a lot of steps but the idea is to:
#    1) Seperate the the events into each team (Red/Blue)
#    2) For each, go through each match and transpose the list of events into a single row
#    3) Stack a table that has the events for both team of the matches

bluerows = pd.DataFrame()
stackedData_blue = stackedData
stackedData_blue['EventBlue'] = np.where( stackedData_blue['Team']!="Red",stackedData_blue['Event'],np.nan)


redrows = pd.DataFrame()
stackedData_red = stackedData
stackedData_red['EventRed'] = np.where( stackedData_red['Team']=="Red",stackedData_red['Event'],np.nan)


for i in range(0,len(matchevents)):
    
    #Red Team Output
    stackedData_match_red = stackedData_red[stackedData_red['id'] == matchevents.iloc[i,0] ]
    
    redextract = stackedData_match_red.iloc[:,[1,11]]
    redextract.iloc[:,0] = redextract.iloc[:,0]-1
    redextract = redextract.set_index('EventNum')
    
    redrow = pd.DataFrame(redextract.transpose())
    redrow['id'] = (stackedData_match_red['id'].unique())
    
    redrows = redrows.append((redrow))
    redrows = redrows.reset_index(drop=True)
    
    
    
    #Blue Team Output
    stackedData_match_blue = stackedData_blue[stackedData_blue['id'] == matchevents.iloc[i,0] ]
    
    blueextract = stackedData_match_blue.iloc[:,[1,10]]
    blueextract.iloc[:,0] = blueextract.iloc[:,0]-1
    blueextract = blueextract.set_index('EventNum')
    
    bluerow = pd.DataFrame(blueextract.transpose())
    bluerow['id'] = (stackedData_match_blue['id'].unique())
    
    bluerows = bluerows.append((bluerow))
    bluerows = bluerows.reset_index(drop=True)
    
  
    
redrows = redrows.sort_values('id')
redrows.head(5)
bluerows = bluerows.sort_values('id')
bluerows.head(5)
# We can now merge these two tables for each team's events in the match to
# our table with just the match ids. We also add a column for the result of 
# the red team for the match and change column names according to which team 
# made the event.



matchevents2 = matchevents.merge(redrows,how='left',on='id')
matchevents3 = matchevents2.merge(bluerows,how='left',on='id')
    

    
matchevents4 = matchevents3.merge(matchinfo[['id','rResult','gamelength']], on='id',how='left')


matchevents4.columns = ['id',
'RedEvent1','RedEvent2','RedEvent3',
'RedEvent4','RedEvent5','RedEvent6','RedEvent7',
'RedEvent8','RedEvent9','RedEvent10','RedEvent11',
'RedEvent12','RedEvent13','RedEvent14','RedEvent15',
'RedEvent16','RedEvent17','RedEvent18','RedEvent19',
'RedEvent20','RedEvent21','RedEvent22','RedEvent23',
'RedEvent24','RedEvent25','RedEvent26','RedEvent27',
'RedEvent28','RedEvent29','RedEvent30','RedEvent31',
'RedEvent32','RedEvent33','RedEvent34','RedEvent35',
'RedEvent36','RedEvent37','RedEvent38','RedEvent39',
'RedEvent40','RedEvent41','RedEvent42','RedEvent43',
'RedEvent44','RedEvent45','RedEvent46','RedEvent47',
'RedEvent48','RedEvent49','RedEvent50','RedEvent51',
'RedEvent52','RedEvent53','RedEvent54','RedEvent55',
'RedEvent56','RedEvent57','RedEvent58','RedEvent59',
'RedEvent60','RedEvent61','RedEvent62','RedEvent63',
'RedEvent64','RedEvent65','RedEvent66','RedEvent67',
'RedEvent68','RedEvent69','RedEvent70','RedEvent71',
'RedEvent72','RedEvent73','RedEvent74','RedEvent75',
'RedEvent76','RedEvent77','RedEvent78','RedEvent79',
                        
                        
'BlueEvent1','BlueEvent2','BlueEvent3','BlueEvent4',
'BlueEvent5','BlueEvent6','BlueEvent7','BlueEvent8',
'BlueEvent9','BlueEvent10','BlueEvent11','BlueEvent12',
'BlueEvent13','BlueEvent14','BlueEvent15','BlueEvent16',
'BlueEvent17','BlueEvent18','BlueEvent19','BlueEvent20',
'BlueEvent21','BlueEvent22','BlueEvent23','BlueEvent24',
'BlueEvent25','BlueEvent26','BlueEvent27','BlueEvent28',
'BlueEvent29','BlueEvent30','BlueEvent31','BlueEvent32',
'BlueEvent33','BlueEvent34','BlueEvent35','BlueEvent36',
'BlueEvent37','BlueEvent38','BlueEvent39','BlueEvent40',
'BlueEvent41','BlueEvent42','BlueEvent43','BlueEvent44',
'BlueEvent45','BlueEvent46','BlueEvent47','BlueEvent48',
'BlueEvent49','BlueEvent50','BlueEvent51','BlueEvent52',
'BlueEvent53','BlueEvent54','BlueEvent55','BlueEvent56',
'BlueEvent57','BlueEvent58','BlueEvent59','BlueEvent60',
'BlueEvent61','BlueEvent62','BlueEvent63',
'BlueEvent64','BlueEvent65','BlueEvent66','BlueEvent67',
'BlueEvent68','BlueEvent69','BlueEvent70','BlueEvent71',
'BlueEvent72','BlueEvent73','BlueEvent74','BlueEvent75',
'BlueEvent76','BlueEvent77','BlueEvent78','BlueEvent79',
                        
                        'rResult','gamelength']



matchevents4.head(20)
# We now decided, for the purpose of calculating probabilities, to consider one team's perseperctive.
# Therefore, we make all events either positive or negative for red team but keep their label otherwise.

matchevents5=matchevents4
for j in range(1,len(list(redrows))):
    matchevents5['RedEvent'+str(j)] = '+'+ matchevents5['RedEvent'+str(j)].astype(str)
    matchevents5['BlueEvent'+str(j)] = '-'+ matchevents5['BlueEvent'+str(j)].astype(str)
    
    matchevents5 = matchevents5.replace('+nan',np.nan)
    matchevents5['RedEvent'+str(j)] =  matchevents5['RedEvent'+str(j)].fillna(
                                        (matchevents5['BlueEvent'+str(j)]).astype(str))
    
matchevents5.head()
# We take on the red event columns now  and re-add the end result of the game for red team (1=win, 0=loss)

RedMatchEvents = matchevents5.iloc[:,0:80]
RedMatchEvents['RedResult'] = matchevents5['rResult']
RedMatchEvents['MatchLength'] = matchevents5['gamelength']
RedMatchEvents.iloc[0:10]
RedMatchEvents[['RedEvent1','id']].groupby('RedEvent1').count()
RedMatchEvents[['RedEvent1','MatchLength']].groupby('RedEvent1').mean()
sns.boxplot(RedMatchEvents['RedEvent1'],RedMatchEvents['MatchLength'],RedMatchEvents['RedResult'],
            boxprops=dict(alpha=.7) )
plt.xticks(rotation=45)
plt.title('Distribution of Match Length by First Event and Match Result (Win = 1, Loss = 0)')
plt.ylim(0,100)
plt.xlabel('Event 1')
plt.plot([1.5, 1.5], [0, 100],'k', linewidth=2,alpha=0.8 )
plt.plot([3.5, 3.5], [0, 100],'k', linewidth=2,alpha=0.8 )
plt.plot([5.5, 5.5], [0, 100],'k', linewidth=2,alpha=0.8 )
plt.show()
# We can now use this to calculate some conditional probabilities as shown

TestData = RedMatchEvents

PwinGivenFirstBloodWon = ( (len(TestData[(TestData['RedEvent1']=="+KILLS")&(TestData['RedResult']==1)])/len(TestData))/
        (len( TestData[TestData['RedEvent1']=="+KILLS"])/len(TestData)) )
    
PwinGivenFirstBloodLost = ( (len(TestData[(TestData['RedEvent1']=="-KILLS")&(TestData['RedResult']==1)])/len(TestData))/
        (len( TestData[TestData['RedEvent1']=="-KILLS"])/len(TestData)) )


PwinGivenFirstTowerWon = ( (len(TestData[(TestData['RedEvent1']=="+OUTER_TURRET")&(TestData['RedResult']==1)])/len(TestData))/
        (len( TestData[TestData['RedEvent1']=="+OUTER_TURRET"])/len(TestData)) )
    
PwinGivenFirstTowerLost = ( (len(TestData[(TestData['RedEvent1']=="-OUTER_TURRET")&(TestData['RedResult']==1)])/len(TestData))/
        (len( TestData[TestData['RedEvent1']=="-OUTER_TURRET"])/len(TestData)) )


PwinGivenFirstDragonWon = ( (len(TestData[(TestData['RedEvent1']=="+DRAGON")&(TestData['RedResult']==1)])/len(TestData))/
        (len( TestData[TestData['RedEvent1']=="+DRAGON"])/len(TestData)) )
    
PwinGivenFirstDragonLost = ( (len(TestData[(TestData['RedEvent1']=="-DRAGON")&(TestData['RedResult']==1)])/len(TestData))/
        (len( TestData[TestData['RedEvent1']=="-DRAGON"])/len(TestData)) )


PwinGivenFirstRiftHeraldWon = ( (len(TestData[(TestData['RedEvent1']=="+RIFT_HERALD")&(TestData['RedResult']==1)])/len(TestData))/
        (len( TestData[TestData['RedEvent1']=="+RIFT_HERALD"])/len(TestData)) )
    
PwinGivenFirstRiftHeraldLost = ( (len(TestData[(TestData['RedEvent1']=="-RIFT_HERALD")&(TestData['RedResult']==1)])/len(TestData))/
        (len( TestData[TestData['RedEvent1']=="-RIFT_HERALD"])/len(TestData)) )





print("-------FIRST BLOOD--------------------------------")
print("P(Won | First Blood Taken):",PwinGivenFirstBloodWon)
print("P(Won | First Blood Lost):",PwinGivenFirstBloodLost)

print("")
print("-------FIRST TURRET-------------------------------")
print("P(Won | First Tower Won):",PwinGivenFirstTowerWon)
print("P(Won | First Tower Lost):",PwinGivenFirstTowerLost)

print("")
print("-------FIRST DRAGON-------------------------------")
print("P(Won | First Dragon Won):",PwinGivenFirstDragonWon)
print("P(Won | First Dragon Lost):",PwinGivenFirstDragonLost)

print("")
print("-------FIRST RIFT HERALD (NOTE: ONLY 17 GAMES)----")
print("P(Won | First Rift Herald Won):",PwinGivenFirstRiftHeraldWon)
print("P(Won | First Rift Herald Lost):",PwinGivenFirstRiftHeraldLost)

aggs = {'id':'count','MatchLength':'mean'}

RedMatchTWOEvents = (RedMatchEvents[['RedEvent1','RedEvent2','RedResult','id','MatchLength']].groupby(
        ['RedEvent1','RedEvent2','RedResult']).agg(aggs).reset_index())

RedMatchTWOEvents = RedMatchTWOEvents.sort_values(['RedEvent1','RedEvent2','RedResult'])

RedMatchTWOEventsWINS = RedMatchTWOEvents[RedMatchTWOEvents['RedResult']==1]
RedMatchTWOEventsLOSS = RedMatchTWOEvents[RedMatchTWOEvents['RedResult']==0]


# First merge the RedWin and RedLoss data tables
# Then remove events which only resulted in a win then calculate the total number of games that has these two events
# Use this total to calculate the prob of win and loss respectively 

RedMatchTWOEventsMERGED = RedMatchTWOEventsWINS.merge(RedMatchTWOEventsLOSS, how='left',on=['RedEvent1','RedEvent2'])


RedMatchTWOEventsMERGED = RedMatchTWOEventsMERGED[RedMatchTWOEventsMERGED['id_y']>0]
RedMatchTWOEventsMERGED['Total'] = RedMatchTWOEventsMERGED['id_x']+RedMatchTWOEventsMERGED['id_y']

RedMatchTWOEventsMERGED['ProbWIN'] = RedMatchTWOEventsMERGED['id_x']/RedMatchTWOEventsMERGED['Total'].sum()
RedMatchTWOEventsMERGED['ProbLOSS'] = RedMatchTWOEventsMERGED['id_y']/RedMatchTWOEventsMERGED['Total'].sum()

RedMatchTWOEventsMERGED['ProbE1ANDE2'] = RedMatchTWOEventsMERGED['Total']/(RedMatchTWOEventsMERGED['Total'].sum())

RedMatchTWOEventsMERGED['ProbWINgivenE1ANDE2'] = RedMatchTWOEventsMERGED['ProbWIN']/RedMatchTWOEventsMERGED['ProbE1ANDE2']
RedMatchTWOEventsMERGED['ProbLOSSgivenE1ANDE2'] = RedMatchTWOEventsMERGED['ProbLOSS']/RedMatchTWOEventsMERGED['ProbE1ANDE2']

# Create column to single binary digit for whether the first event is positive or negative

RedMatchTWOEventsMERGED['RedEvent1Gain'] = np.where(
                                (RedMatchTWOEventsMERGED['RedEvent1']=="+KILLS") |
                                (RedMatchTWOEventsMERGED['RedEvent1']=="+OUTER_TURRET") |
                                (RedMatchTWOEventsMERGED['RedEvent1']=="+DRAGON") |
                                (RedMatchTWOEventsMERGED['RedEvent1']=="+RIFT_HERALD") ,1,0
                                                   
                                                   
                                                   )
# Repeat for second event

RedMatchTWOEventsMERGED['RedEvent2Gain'] = np.where(
                                (RedMatchTWOEventsMERGED['RedEvent2']=="+KILLS") |
                                (RedMatchTWOEventsMERGED['RedEvent2']=="+OUTER_TURRET") |
                                (RedMatchTWOEventsMERGED['RedEvent2']=="+DRAGON") |
                                (RedMatchTWOEventsMERGED['RedEvent2']=="+RIFT_HERALD") ,1,0
                                                   
                                                   
                                                   )
# Create another column for combination of first and second event outcomes classification
RedMatchTWOEventsMERGED['Event1AND2Outcome'] = np.where(
    (RedMatchTWOEventsMERGED['RedEvent1Gain']==1)&(RedMatchTWOEventsMERGED['RedEvent2Gain']==1),"Both Positive",
                
    np.where(
        (((RedMatchTWOEventsMERGED['RedEvent1Gain']==1)&(RedMatchTWOEventsMERGED['RedEvent2Gain']==0))|
        ((RedMatchTWOEventsMERGED['RedEvent1Gain']==0)&(RedMatchTWOEventsMERGED['RedEvent2Gain']==1))),"One Positive",
    
    np.where(
        (RedMatchTWOEventsMERGED['RedEvent1Gain']==0)&(RedMatchTWOEventsMERGED['RedEvent2Gain']==0),"Neither Positive",
             "MISSING",)))

# Sort by highest probability of win to lowest
RedMatchTWOEventsMERGED = RedMatchTWOEventsMERGED.sort_values('ProbWINgivenE1ANDE2',ascending=False)

# Remove event combination with less than x number of games to remove possible outliers
RedMatchTWOEventsMERGED = RedMatchTWOEventsMERGED[RedMatchTWOEventsMERGED['Total']>=0]


RedMatchTWOEventsMERGED.head(5)

sns.pairplot(data = RedMatchTWOEventsMERGED, x_vars='ProbWINgivenE1ANDE2',y_vars='MatchLength_x',
           hue= 'Event1AND2Outcome', size=8)
plt.title('Probability of Winning Given the First Two Events against Average Game Duration, \n Coloured by Event 1 and 2 Outcomes')
plt.xlabel('Probability of Win GIVEN First Two Events')
plt.ylabel('Average Game Length')
plt.xlim([0,1])
plt.xticks(np.arange(0,1.1,0.1))
#plt.ylim([20,50])

plt.show()
RedMatchEvents.head()
# WARNING: Takes a while to run
# Replace all N/As with the match outcome so that our final state is either a Win or Loss
for i in range(1,80):
    RedMatchEvents['RedEvent'+str(i)] = RedMatchEvents['RedEvent'+str(i)].replace('-nan',RedMatchEvents['RedResult'].astype(str))
    RedMatchEvents['RedEvent'+str(i)] = RedMatchEvents['RedEvent'+str(i)].replace('+nan',RedMatchEvents['RedResult'].astype(str))
    #Print i for progress tracking
    #print(i)
RedMatchEvents.head()
RedMatchEvents[['RedEvent60','id']].groupby('RedEvent60').count()
RedMatchEvents2 = RedMatchEvents
# WARNING: Takes a little while to run

EventList = [
    #Positive Events
       '+KILLS', '+OUTER_TURRET', '+DRAGON', '+RIFT_HERALD', '+BARON_NASHOR',
       '+INNER_TURRET', '+BASE_TURRET', '+INHIBITOR', '+NEXUS_TURRET',
       '+ELDER_DRAGON',
    #Negative Events
       '-KILLS', '-OUTER_TURRET', '-DRAGON', '-RIFT_HERALD', '-BARON_NASHOR',
       '-INNER_TURRET', '-BASE_TURRET', '-INHIBITOR', '-NEXUS_TURRET',
       '-ELDER_DRAGON',
    #Game Win or Loss Events        
       '1','0']

RedMatchMDP = pd.DataFrame()

for i in range(1,79):
                              
    Event = i
    for j1 in range(0,len(EventList)):
        Event1 = EventList[j1]
        for j2 in range(0,len(EventList)):
            
            Event2 = EventList[j2]
            
            
            if  len(RedMatchEvents2[(RedMatchEvents2['RedEvent'+str(Event)]==Event1)])==0:
                continue
            #elif len(RedMatchEvents2[(RedMatchEvents2['RedEvent'+str(Event)]==Event1)&
            #                   (RedMatchEvents2['RedEvent'+str(Event+1)]==Event2) ])==0:
                continue
                
            else:
                TransProb = (
                    len(RedMatchEvents2[(RedMatchEvents2['RedEvent'+str(Event)]==Event1)&
                               (RedMatchEvents2['RedEvent'+str(Event+1)]==Event2) ])/

                    len(RedMatchEvents2[(RedMatchEvents2['RedEvent'+str(Event)]==Event1)])
                    )


            RedMatchMDP2 = pd.DataFrame({'StartState':Event,'EndState':Event+1,'Event1':Event1,'Event2':Event2,'Probability':TransProb},
                                  index=[0])
            RedMatchMDP = RedMatchMDP.append(RedMatchMDP2)
   
    #Print i for tracking progress
    #print(i)
    

RedMatchMDP = RedMatchMDP[['StartState','EndState','Event1','Event2','Probability']]
RedMatchMDP[(RedMatchMDP['StartState']==61)&(RedMatchMDP['Event1']=="+INHIBITOR")]
EndCondition = RedMatchMDP[
    ((RedMatchMDP['Event1']!="1")&(RedMatchMDP['Event2']=="1") )|
    ((RedMatchMDP['Event1']!="0")&(RedMatchMDP['Event2']=="0"))]

EndCondition = EndCondition.sort_values('Probability',ascending=False)

EndConditionGrouped = EndCondition[['StartState','Probability']].groupby('StartState').mean().reset_index()
EndConditionGrouped['CumProb'] = EndConditionGrouped['Probability'].cumsum()

EndConditionGrouped2 = EndCondition[['StartState','Probability']].groupby('StartState').sum().reset_index()
EndConditionGrouped2['CumProb'] = EndConditionGrouped2['Probability'].cumsum()

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0,0].bar(EndConditionGrouped['StartState'],EndConditionGrouped['Probability'] ,alpha=0.3)
axes[0,0].plot(EndConditionGrouped['StartState'],EndConditionGrouped['Probability'])
axes[0,0].set_title('Mean Probability Dist')
axes[0,0].set_xlabel("State")
axes[0,0].set_ylabel("Probability of Ending")
axes[0,0].set_xticks([],[])
axes[0,0].set_xlabel("")
axes[0,0].set_xlim([0,80])
axes[0,0].grid(False)

axes[0,1].bar(EndConditionGrouped['StartState'],EndConditionGrouped['CumProb'] ,alpha=0.3)
axes[0,1].plot(EndConditionGrouped['StartState'],EndConditionGrouped['CumProb'])
axes[0,1].set_title('Mean Cumulative Probability Dist')
axes[0,1].set_xlabel("State")
axes[0,1].set_ylabel("Cumlative Probability of Ending")
axes[0,1].set_xticks([])
axes[0,1].set_xlabel("")
axes[0,1].set_xlim([0,80])
axes[0,1].grid(False)

axes[1,0].bar(EndConditionGrouped2['StartState'],EndConditionGrouped2['Probability'] ,alpha=0.3)
axes[1,0].plot(EndConditionGrouped2['StartState'],EndConditionGrouped2['Probability'])
axes[1,0].set_title('Sum Probability Dist')
axes[1,0].set_xlabel("State")
axes[1,0].set_ylabel("Probability of Ending")
axes[1,0].set_xlim([0,80])
axes[1,0].grid(False)

axes[1,1].bar(EndConditionGrouped2['StartState'],EndConditionGrouped2['CumProb'] ,alpha=0.3)
axes[1,1].plot(EndConditionGrouped2['StartState'],EndConditionGrouped2['CumProb'])
axes[1,1].set_title('Sum Cumulative Probability Dist')
axes[1,1].set_xlabel("State")
axes[1,1].set_ylabel("Cumlative Probability of Ending")
axes[1,1].set_xlim([0,80])
axes[1,1].grid(False)

fig.suptitle("Probability of Game Ending in Each State Averaged and Summed over Varying Start Events")

fig.set_figheight(15)
fig.set_figwidth(15)
plt.show()
RedMatchMDP['Reward'] = 0

RedMatchMDP.head()
len(RedMatchMDP)
RedMatchMDP[(RedMatchMDP['StartState']==15)&(RedMatchMDP['Event1']=="+ELDER_DRAGON")]
alpha = 0.1
gamma = 0.9
num_episodes = 100
epsilon = 0.1

reward = RedMatchMDP['Reward']

StartState = 1
StartEvent = '+KILLS'
StartAction = '+OUTER_TURRET'
def MCModelv1(data, alpha, gamma, epsilon, reward, StartState, StartEvent, StartAction, num_episodes):
    
    # Initiatise variables appropiately
    
    data['V'] = 0
 
    
    outcomes = pd.DataFrame()
    episode_return = pd.DataFrame()
    actions_output = pd.DataFrame()
    
    for e in range(0,num_episodes):
        
        action = []

        current_state = StartState
        current_action = StartEvent
        next_action = StartAction 
   
        actions = pd.DataFrame()
 
        for a in range(0,100):
            
            action_table = pd.DataFrame()

            
            if (current_action=="1") | (current_action=="0") | (current_state==79):
                continue
            else:
                
                data_e = data[(data['StartState']==current_state)&(data['Event1']==current_action)]

                data_e = data_e.sort_values('Probability')
                data_e['CumProb'] = data_e['Probability'].cumsum()
                data_e['CumProb'] = np.round(data_e['CumProb'],4)

                
                rng = np.round(np.random.random()*data_e['CumProb'].max(),4)
                action_table = data_e[ data_e['CumProb'] >= rng]
                action_table = action_table[ action_table['CumProb'] == action_table['CumProb'].min()]
                action_table = action_table.reset_index()
                
                action = action_table['Event2'][0]
                
                if action == "1":
                    step_reward = 10*(gamma**a)
                elif action == "0":
                    step_reward = -10*(gamma**a)
                else:
                    step_reward = -0.005*(gamma**a)
                
                action_table['StepReward'] = step_reward
                

                action_table['Episode'] = e
                action_table['Action'] = a
                
                current_action = action
                current_state = current_state+1
                
                
                actions = actions.append(action_table)

        actions_output = actions_output.append(actions)
                
        episode_return = actions['StepReward'].sum()

                
        actions['Return']= episode_return
                
        data = data.merge(actions[['StartState','EndState','Event1','Event2','Return']], how='left',on =['StartState','EndState','Event1','Event2'])
        data['Return'] = data['Return'].fillna(0)    
             
        data['V'] = data['V'] + alpha*(data['Return']-data['V'])
        data = data.drop('Return', 1)
        
        
                
        if current_action=="1":
            outcome = "WIN"
        elif current_action=="0":
            outcome = "LOSS"
        else:
            outcome = "INCOMPLETE"
        outcome = pd.DataFrame({'Epsiode':[e],'Outcome':[outcome]})
        outcomes = outcomes.append(outcome)

        
        

        
   
        
    
        
    optimal_policy_table = data[ ( data['StartState']==StartState) & (data['Event1']==StartEvent)&(data['Event2']==StartAction)]
     
    for i in range(2,79):
        optimal_V = data[data['StartState']==i]['V'].max()
        optimal_policy = data[ ( data['V']==optimal_V) & (data['StartState']==i)]      
        optimal_policy_table = optimal_policy_table.append(optimal_policy)
                
    return(outcomes,actions_output,data,optimal_policy_table)
    
start_time = timeit.default_timer()


Mdl = MCModelv1(data=RedMatchMDP, alpha = alpha, gamma=gamma, epsilon = epsilon, reward = reward,
                StartState=StartState, StartEvent=StartEvent,StartAction=StartAction,
                num_episodes = num_episodes)

elapsed = timeit.default_timer() - start_time

print("Time taken to run model:",np.round(elapsed/60,2),"mins")
Mdl[3].head()
def MCModelv2(data, alpha, gamma, epsilon, reward, StartState, StartEvent, StartAction, num_episodes):
    
    # Initiatise variables appropiately
    
    data['V'] = 0
 
    
    outcomes = pd.DataFrame()
    episode_return = pd.DataFrame()
    actions_output = pd.DataFrame()
    
    for e in range(0,num_episodes):
        action = []

        current_state = StartState
        current_action = StartEvent
         
        
      
            
            
        actions = pd.DataFrame()
 
        for a in range(0,100):
            
            action_table = pd.DataFrame()

            
            if (current_action=="1") | (current_action=="0") | (current_state==79):
                continue
            else:
                
                data_e = data[(data['StartState']==current_state)&(data['Event1']==current_action)]

                data_e = data_e[data_e['Probability']>0]

                
                if (StartAction is None)&(a==0):
                    random_first_action = data_e.sample()
                    action_table = random_first_action
                    action_table = action_table.reset_index()
                    action = action_table['Event2'][0]
                elif (a==0):
                    action_table = data_e[ data_e['Event2'] ==StartAction]
                    action = StartAction
                else:
                    data_e = data_e.sort_values('Probability')
                    data_e['CumProb'] = data_e['Probability'].cumsum()
                    data_e['CumProb'] = np.round(data_e['CumProb'],4)
                    rng = np.round(np.random.random()*data_e['CumProb'].max(),4)
                    action_table = data_e[ data_e['CumProb'] >= rng]
                    action_table = action_table[ action_table['CumProb'] == action_table['CumProb'].min()]
                    action_table = action_table.reset_index()

                    action = action_table['Event2'][0]
                if action == "1":
                    step_reward = 10*(gamma**a)
                elif action == "0":
                    step_reward = -10*(gamma**a)
                else:
                    step_reward = -0.005*(gamma**a)

                action_table['StepReward'] = step_reward


                action_table['Episode'] = e
                action_table['Action'] = a

                current_action = action
                current_state = current_state+1


                actions = actions.append(action_table)

        actions_output = actions_output.append(actions)
                
        episode_return = actions['StepReward'].sum()

                
        actions['Return']= episode_return
                
        data = data.merge(actions[['StartState','EndState','Event1','Event2','Return']], how='left',on =['StartState','EndState','Event1','Event2'])
        data['Return'] = data['Return'].fillna(0)    
             
        data['V'] = data['V'] + alpha*(data['Return']-data['V'])
        data = data.drop('Return', 1)
        
        
                
        if current_action=="1":
            outcome = "WIN"
        elif current_action=="0":
            outcome = "LOSS"
        else:
            outcome = "INCOMPLETE"
        outcome = pd.DataFrame({'Epsiode':[e],'Outcome':[outcome]})
        outcomes = outcomes.append(outcome)

        
        

        
   
        
    
        if StartAction is None:
            optimal_policy_table = pd.DataFrame()
            for i in range(1,79):
                optimal_V = data[data['StartState']==i]['V'].max()
                optimal_policy = data[ ( data['V']==optimal_V) & (data['StartState']==i)]      
                optimal_policy_table = optimal_policy_table.append(optimal_policy)        
        else:
            optimal_policy_table = data[ ( data['StartState']==StartState) & (data['Event1']==StartEvent)&(data['Event2']==StartAction)]
            for i in range(2,79):
                optimal_V = data[data['StartState']==i]['V'].max()
                optimal_policy = data[ ( data['V']==optimal_V) & (data['StartState']==i)]      
                optimal_policy_table = optimal_policy_table.append(optimal_policy)

    return(outcomes,actions_output,data,optimal_policy_table)
    
alpha = 0.1
gamma = 0.9
num_episodes = 100
epsilon = 0.1

reward = RedMatchMDP['Reward']

StartState = 1
StartEvent = '+KILLS'
StartAction = None


start_time = timeit.default_timer()


Mdl2 = MCModelv2(data=RedMatchMDP, alpha = alpha, gamma=gamma, epsilon = epsilon, reward = reward,
                StartState=StartState, StartEvent=StartEvent,StartAction=None,
                num_episodes = num_episodes)

elapsed = timeit.default_timer() - start_time

print("Time taken to run model:",np.round(elapsed/60,2),"mins")
Mdl2[3].head(30)
def MCModelv3(data, alpha, gamma, epsilon, reward, StartState, StartEvent, StartAction, num_episodes):
    
    # Initiatise variables appropiately
    
    data['V'] = 0
    data_output = data
    
    outcomes = pd.DataFrame()
    episode_return = pd.DataFrame()
    actions_output = pd.DataFrame()
    
    for e in range(0,num_episodes):
        action = []

        current_state = StartState
        current_action = StartEvent
        
        
        data_e1 = data
    
    
        actions = pd.DataFrame()

        for a in range(0,100):
            
            action_table = pd.DataFrame()
       
           
            if (current_action=="1") | (current_action=="0") | (current_state==79):
                continue
            else:
                if a==0:
                    data_e1=data_e1
                    
                elif (len(individual_actions_count[individual_actions_count['Event2']=="+RIFT_HERALD"])==1):
                    data_e1_e1 = data_e1[(data_e1['Event2']!='+RIFT_HERALD')|(data_e1['Event2']!='-RIFT_HERALD')]
                    
                elif (len(individual_actions_count[individual_actions_count['Event2']=="-RIFT_HERALD"])==1):
                    data_e1 = data_e1[(data_e1['Event2']!='+RIFT_HERALD')|(data_e1['Event2']!='-RIFT_HERALD')]
                
                elif (len(individual_actions_count[individual_actions_count['Event2']=="+OUTER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Event2']!='+OUTER_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Event2']=="-OUTER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Event2']!='-OUTER_TURRET']
                    
                elif (len(individual_actions_count[individual_actions_count['Event2']=="+INNER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Event2']!='+INNER_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Event2']=="-INNER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Event2']!='-INNER_TURRET']
                    
                elif (len(individual_actions_count[individual_actions_count['Event2']=="+BASE_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Event2']!='+BASE_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Event2']=="-BASE_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Event2']!='-BASE_TURRET']
                    
                elif (len(individual_actions_count[individual_actions_count['Event2']=="+INHIBITOR"])==3):
                    data_e1 = data_e1[data_e1['Event2']!='+INHIBITOR']
                elif (len(individual_actions_count[individual_actions_count['Event2']=="-INHIBITOR"])==3):
                    data_e1 = data_e1[data_e1['Event2']!='-INHIBITOR']
                    
                elif (len(individual_actions_count[individual_actions_count['Event2']=="+NEXUS_TURRET"])==2):
                    data_e1 = data_e1[data_e1['Event2']!='+NEXUS_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Event2']=="-NEXUS_TURRET"])==2):
                    data_e1 = data_e1[data_e1['Event2']!='-NEXUS_TURRET']
                
                       
                else:
                    data_e1 = data_e1

                
                data_e = data_e1[(data_e1['StartState']==current_state)&(data_e1['Event1']==current_action)]
                
                data_e = data_e[data_e['Probability']>0]
                
                if (StartAction is None)&(a==0):
                    random_first_action = data_e.sample()
                    action_table = random_first_action
                    action_table = action_table.reset_index()
                    action = action_table['Event2'][0]
                elif (a==0):
                    action_table = data_e[ data_e['Event2'] ==StartAction]
                    action = StartAction
                else:
                    data_e = data_e.sort_values('Probability')
                    data_e['CumProb'] = data_e['Probability'].cumsum()
                    data_e['CumProb'] = np.round(data_e['CumProb'],4)
                    

                    rng = np.round(np.random.random()*data_e['CumProb'].max(),4)
                    action_table = data_e[ data_e['CumProb'] >= rng]
                    action_table = action_table[ action_table['CumProb'] == action_table['CumProb'].min()]
                    action_table = action_table.reset_index()

                    action = action_table['Event2'][0]
                if action == "1":
                    step_reward = 10*(gamma**a)
                elif action == "0":
                    step_reward = -10*(gamma**a)
                else:
                    step_reward = -0.005*(gamma**a)

                action_table['StepReward'] = step_reward


                action_table['Episode'] = e
                action_table['Action'] = a

                current_action = action
                current_state = current_state+1

                
                actions = actions.append(action_table)
                
                individual_actions_count = actions
            

        actions_output = actions_output.append(actions)
                
        episode_return = actions['StepReward'].sum()

                
        actions['Return']= episode_return
                
        data_output = data_output.merge(actions[['StartState','EndState','Event1','Event2','Return']], how='left',on =['StartState','EndState','Event1','Event2'])
        data_output['Return'] = data_output['Return'].fillna(0)    
             
        data_output['V'] = data_output['V'] + alpha*(data_output['Return']-data_output['V'])
        data_output = data_output.drop('Return', 1)
        
        
                
        if current_action=="1":
            outcome = "WIN"
        elif current_action=="0":
            outcome = "LOSS"
        else:
            outcome = "INCOMPLETE"
        outcome = pd.DataFrame({'Epsiode':[e],'Outcome':[outcome]})
        outcomes = outcomes.append(outcome)

        
        

        optimal_policy_table = pd.DataFrame()
   
        
        if (StartAction is None):
            
            optimal_policy_table =    data_output[ (data_output['StartState']==StartState)&(data_output['Event1']==StartEvent) &
                (data_output['V']==(data_output[(data_output['StartState']==StartState)&(data_output['Event1']==StartEvent)]['V'].max()))  ]
            for i in range(2,79):
                optimal_V = data_output[(data_output['StartState']==i)]['V'].max()
                optimal_policy = data_output[ ( data_output['V']==optimal_V) & (data_output['StartState']==i)]      
                optimal_policy_table = optimal_policy_table.append(optimal_policy)        
        else:
            optimal_policy_table = data_output[ ( data_output['StartState']==StartState) & (data_output['Event1']==StartEvent)&(data_output['Event2']==StartAction)]
            for i in range(2,79):
                optimal_V = data_output[data_output['StartState']==i]['V'].max()
                optimal_policy = data_output[ ( data_output['V']==optimal_V) & (data_output['StartState']==i)]      
                optimal_policy_table = optimal_policy_table.append(optimal_policy)
                
        if (StartAction is None):
            currentpath_action = StartEvent
            optimal_path = pd.DataFrame()

            for i in range(1,79):
                StartPathState = i
                nextpath_action = data_output [ (data_output['V'] == data_output[ (data_output['StartState']==StartPathState) & (data_output['Event1']==currentpath_action) ]['V'].max()) & 
                                               (data_output['StartState']==StartPathState) & (data_output['Event1']==currentpath_action)  ]
                if (nextpath_action['V'].max()==0):
                    break
                else:
                    nextpath_action = nextpath_action.reset_index(drop=True)
                    currentpath_action = nextpath_action['Event2'][0]
                    optimal_path = optimal_path.append(nextpath_action)
                    
        else:
            currentpath_action = StartEvent
            optimal_path = data_output[(data_output['StartState']==StartPathState) & (data_output['Event1']==currentpath_action) & (data_output['Event2']==StartAction) ]
            for i in range(2,79):
                StartPathState = i
                nextpath_action = data_output [ (data_output['V'] == data_output[ (data_output['StartState']==StartPathState) & (data_output['Event1']==currentpath_action) ]['V'].max()) & 
                                               (data_output['StartState']==StartPathState) & (data_output['Event1']==currentpath_action)  ]
                if (nextpath_action['V'].max()==0):
                    break
                else:

                    nextpath_action = nextpath_action.reset_index(drop=True)
                    currentpath_action = nextpath_action['Event2'][0]
                    optimal_path = optimal_path.append(nextpath_action)


                
                
                
    

        



    return(outcomes,actions_output,data_output,optimal_policy_table,optimal_path)
    
alpha = 0.1
gamma = 0.9
num_episodes = 100
epsilon = 0.1

reward = RedMatchMDP['Reward']

StartState = 1
StartEvent = '+KILLS'
StartAction = None


start_time = timeit.default_timer()


Mdl3 = MCModelv3(data=RedMatchMDP, alpha = alpha, gamma=gamma, epsilon = epsilon, reward = reward,
                StartState=StartState, StartEvent=StartEvent,StartAction=StartAction,
                num_episodes = num_episodes)

elapsed = timeit.default_timer() - start_time

print("Time taken to run model:",np.round(elapsed/60,2),"mins")
print("Avg Time taken per episode:", np.round(elapsed/num_episodes,2),"secs")
Mdl3[3].head()
Mdl3[4]