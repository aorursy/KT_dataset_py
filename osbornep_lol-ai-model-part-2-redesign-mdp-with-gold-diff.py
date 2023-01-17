import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import math
from scipy.stats import kendalltau

from IPython.display import clear_output
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

#gold = pd.read_csv('C:\\Users\\Phil\\Documents\\LoL Model\\gold.csv')
gold = pd.read_csv('../input/gold.csv')
gold = gold[gold['Type']=="golddiff"]
gold.head()
# Add ID column based on last 16 digits in match address for simpler matching

matchinfo['id'] = matchinfo['Address'].astype(str).str[-16:]
kills['id'] = kills['Address'].astype(str).str[-16:]
monsters['id'] = monsters['Address'].astype(str).str[-16:]
structures['id'] = structures['Address'].astype(str).str[-16:]
gold['id'] = gold['Address'].astype(str).str[-16:]

# Dragon became multiple types in patch v6.9 (http://leagueoflegends.wikia.com/wiki/V6.9) 
# so we remove and games before this change occured and only use games with the new dragon system

old_dragon_id = monsters[ monsters['Type']=="DRAGON"]['id'].unique()
old_dragon_id

monsters = monsters[ ~monsters['id'].isin(old_dragon_id)]
monsters = monsters.reset_index()

matchinfo = matchinfo[ ~matchinfo['id'].isin(old_dragon_id)]
matchinfo = matchinfo.reset_index()

kills = kills[ ~kills['id'].isin(old_dragon_id)]
kills = kills.reset_index()

structures = structures[ ~structures['id'].isin(old_dragon_id)]
structures = structures.reset_index()

gold = gold[ ~gold['id'].isin(old_dragon_id)]
gold = gold.reset_index()

gold.head(3)
#Transpose Gold table, columns become matches and rows become minutes

gold_T = gold.iloc[:,3:-1].transpose()
gold_T.head(3)
gold2 = pd.DataFrame()

start = timeit.default_timer()
for r in range(0,len(gold)):
    clear_output(wait=True)
    
    # Select each match column, drop any na rows and find the match id from original gold table
    gold_row = gold_T.iloc[:,r]
    gold_row = gold_row.dropna()
    gold_row_id = gold['id'][r]
    
    # Append into table so that each match and event is stacked on top of one another    
    gold2 = gold2.append(pd.DataFrame({'id':gold_row_id,'GoldDiff':gold_row}))
    
    
    stop = timeit.default_timer()
   
    if (r/len(gold)*100) < 5  :
        expected_time = "Calculating..."
        
    else:
        time_perc = timeit.default_timer()
        expected_time = np.round( ( (time_perc-start)/60 / (r/len(gold)) ),2)
        
  
        
    print("Current progress:",np.round(r/len(gold) *100, 2),"%")        
    print("Current run time:",np.round((stop - start)/60,2),"minutes")
    print("Expected Run Time:",expected_time,"minutes")
    

gold3 = gold2[['id','GoldDiff']]
gold3.head(3)
### Create minute column with index, convert from 'min_1' to just the number
gold3['Minute'] = gold3.index.to_series()
gold3['Minute'] = np.where(gold3['Minute'].str[-2]=="_", gold3['Minute'].str[-1],gold3['Minute'].str[-2:])
gold3['Minute'] = gold3['Minute'].astype(int)
gold3 = gold3.reset_index()
gold3 = gold3.sort_values(by=['id','Minute'])

gold3.head(3)

# Gold difference from data is relative to blue team's perspective,
# therefore we reverse this by simply multiplying amount by -1
gold3['GoldDiff'] = gold3['GoldDiff']*-1

gold3.head(10)
matchinfo.head(3)
gold4 = gold3

matchinfo2 = matchinfo[['id','rResult','gamelength']]
matchinfo2['gamlength'] = matchinfo2['gamelength'] + 1
matchinfo2['index'] = 'min_'+matchinfo2['gamelength'].astype(str)
matchinfo2['rResult2'] =  np.where(matchinfo2['rResult']==1,999999,-999999)
matchinfo2 = matchinfo2[['index','id','rResult2','gamelength']]
matchinfo2.columns = ['index','id','GoldDiff','Minute']


gold4 = gold4.append(matchinfo2)
gold4.tail()
kills = kills[ kills['Time']>0]

kills['Minute'] = kills['Time'].astype(int)

kills['Team'] = np.where( kills['Team']=="rKills","Red","Blue")
kills.head(3)
# For the Kills table, we need decided to group by the minute in which the kills took place and averaged 
# the time of the kills which we use later for the order of events

f = {'Time':['mean','count']}

killsGrouped = kills.groupby( ['id','Team','Minute'] ).agg(f).reset_index()
killsGrouped.columns = ['id','Team','Minute','Time Avg','Count']
killsGrouped = killsGrouped.sort_values(by=['id','Minute'])
killsGrouped.head(3)
structures = structures[ structures['Time']>0]

structures['Minute'] = structures['Time'].astype(int)
structures['Team'] = np.where(structures['Team']=="bTowers","Blue",
                        np.where(structures['Team']=="binhibs","Blue","Red"))
structures2 = structures.sort_values(by=['id','Minute'])

structures2 = structures2[['id','Team','Time','Minute','Type']]
structures2.head(3)
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

monsters = monsters[['id','Team','Time','Minute','Type2']]
monsters.columns = ['id','Team','Time','Minute','Type']
monsters.head(3)
GoldstackedData = gold4.merge(killsGrouped, how='left',on=['id','Minute'])
 
monsters_structures_stacked = structures2.append(monsters[['id','Team','Minute','Time','Type']])

GoldstackedData2 = GoldstackedData.merge(monsters_structures_stacked, how='left',on=['id','Minute'])

GoldstackedData2 = GoldstackedData2.sort_values(by=['id','Minute'])
GoldstackedData2.head(30)
GoldstackedData3 = GoldstackedData2
GoldstackedData3['Time2'] = GoldstackedData3['Time'].fillna(GoldstackedData3['Time Avg']).fillna(GoldstackedData3['Minute'])
GoldstackedData3['Team'] = GoldstackedData3['Team_x'].fillna(GoldstackedData3['Team_y'])
GoldstackedData3 = GoldstackedData3.sort_values(by=['id','Time2'])

GoldstackedData3['EventNum'] = GoldstackedData3.groupby('id').cumcount()+1

GoldstackedData3 = GoldstackedData3[['id','EventNum','Team','Minute','Time2','GoldDiff','Count','Type']]

GoldstackedData3.columns = ['id','EventNum','Team','Minute','Time','GoldDiff','KillCount','Struct/Monster']

GoldstackedData3.head(30)
GoldstackedData3[GoldstackedData3['GoldDiff']==999999].head(3)
# We then add an 'Event' column to merge the columns into one, where kills are now
# simple labelled as 'KILLS'

GoldstackedData3['Event'] = np.where(GoldstackedData3['KillCount']>0,"KILLS",None)
GoldstackedData3['Event'] = GoldstackedData3['Event'].fillna(GoldstackedData3['Struct/Monster'])

GoldstackedData3['Event'] = GoldstackedData3['Event'].fillna("NONE")

GoldstackedData3['GoldDiff2'] = np.where( GoldstackedData3['GoldDiff']== 999999,"WIN",
                                np.where( GoldstackedData3['GoldDiff']==-999999, 'LOSS',
                                         
    
                                np.where((GoldstackedData3['GoldDiff']<1000) & (GoldstackedData3['GoldDiff']>-1000),
                                        "EVEN",
                                np.where( (GoldstackedData3['GoldDiff']>=1000) & (GoldstackedData3['GoldDiff']<2500),
                                         "SLIGHTLY_AHEAD",
                                np.where( (GoldstackedData3['GoldDiff']>=2500) & (GoldstackedData3['GoldDiff']<5000),
                                         "AHEAD",
                                np.where( (GoldstackedData3['GoldDiff']>=5000),
                                         "VERY_AHEAD",
                                         
                                np.where( (GoldstackedData3['GoldDiff']<=-1000) & (GoldstackedData3['GoldDiff']>-2500),
                                         "SLIGHTLY_BEHIND",
                                np.where( (GoldstackedData3['GoldDiff']<=-2500) & (GoldstackedData3['GoldDiff']>-5000),
                                         "BEHIND",
                                np.where( (GoldstackedData3['GoldDiff']<=-5000),
                                         "VERY_BEHIND","ERROR"
                                        
                                        )))))))))

GoldstackedData3.head(3)
GoldstackedData3[GoldstackedData3['GoldDiff2']=="ERROR"]
GoldstackedData3[GoldstackedData3['Team']=='Blue'].head(10)
GoldstackedData3['Next_Min'] = GoldstackedData3['Minute']+1


GoldstackedData4 = GoldstackedData3.merge(gold4[['id','Minute','GoldDiff']],how='left',left_on=['id','Next_Min'],
                                         right_on=['id','Minute'])

GoldstackedData4.head(10)
GoldstackedData4[ GoldstackedData4['GoldDiff_y']== -999999].head(3)
GoldstackedData4['GoldDiff2_Next'] =  np.where( GoldstackedData4['GoldDiff_y']== 999999,"WIN",
                                np.where( GoldstackedData4['GoldDiff_y']==-999999, 'LOSS',
                                         
    
                                np.where((GoldstackedData4['GoldDiff_y']<1000) & (GoldstackedData4['GoldDiff_y']>-1000),
                                        "EVEN",
                                np.where( (GoldstackedData4['GoldDiff_y']>=1000) & (GoldstackedData4['GoldDiff_y']<2500),
                                         "SLIGHTLY_AHEAD",
                                np.where( (GoldstackedData4['GoldDiff_y']>=2500) & (GoldstackedData4['GoldDiff_y']<5000),
                                         "AHEAD",
                                np.where( (GoldstackedData4['GoldDiff_y']>=5000),
                                         "VERY_AHEAD",
                                         
                                np.where( (GoldstackedData4['GoldDiff_y']<=-1000) & (GoldstackedData4['GoldDiff_y']>-2500),
                                         "SLIGHTLY_BEHIND",
                                np.where( (GoldstackedData4['GoldDiff_y']<=-2500) & (GoldstackedData4['GoldDiff_y']>-5000),
                                         "BEHIND",
                                np.where( (GoldstackedData4['GoldDiff_y']<=-5000),
                                         "VERY_BEHIND","ERROR"
                                        
                                        )))))))))
GoldstackedData4 = GoldstackedData4[['id','EventNum','Team','Minute_x','Time','Event','GoldDiff2','GoldDiff2_Next']]
GoldstackedData4.columns = ['id','EventNum','Team','Minute','Time','Event','GoldDiff2','GoldDiff2_Next']

GoldstackedData4['Event'] = np.where( GoldstackedData4['Team']=="Red", "+"+GoldstackedData4['Event'],
                                np.where(GoldstackedData4['Team']=="Blue", "-"+GoldstackedData4['Event'], 
                                         GoldstackedData4['Event']))

#GoldstackedData4.head(10)
# Errors are caused due to game ending in minute and then there is no 'next_min' info for this game but our method expects there to be
GoldstackedData4 = GoldstackedData4[GoldstackedData4['GoldDiff2_Next']!="ERROR"]
GoldstackedData4[GoldstackedData4['GoldDiff2_Next']=="ERROR"]
GoldstackedDataFINAL = GoldstackedData4
GoldstackedDataFINAL['Min_State_Action_End'] = ((GoldstackedDataFINAL['Minute'].astype(str)) + "_"
                                       + (GoldstackedDataFINAL['GoldDiff2'].astype(str)) + "_"
                                       + (GoldstackedDataFINAL['Event'].astype(str)) + "_"  
                                       + (GoldstackedDataFINAL['GoldDiff2_Next'].astype(str))
                                      )

GoldstackedDataFINAL['MSAE'] = ((GoldstackedDataFINAL['Minute'].astype(str)) + "_"
                                       + (GoldstackedDataFINAL['GoldDiff2'].astype(str)) + "_"
                                       + (GoldstackedDataFINAL['Event'].astype(str)) + "_"  
                                       + (GoldstackedDataFINAL['GoldDiff2_Next'].astype(str))
                                      )

GoldstackedDataFINAL.head()
goldMDP = GoldstackedDataFINAL[['Minute','GoldDiff2','Event','GoldDiff2_Next']]
goldMDP.columns = ['Minute','State','Action','End']
goldMDP['Counter'] = 1
goldMDP.head()
goldMDP[goldMDP['End']=='ERROR'].head(3)
goldMDP2 = goldMDP.groupby(['Minute','State','Action','End']).count().reset_index()
goldMDP2['Prob'] = goldMDP2['Counter']/(goldMDP2['Counter'].sum())
goldMDP2.head()
goldMDP3 = goldMDP.groupby(['Minute','State','Action']).count().reset_index()
goldMDP3['Prob'] = goldMDP3['Counter']/(goldMDP3['Counter'].sum())
goldMDP3.head()
goldMDP4 = goldMDP2.merge(goldMDP3[['Minute','State','Action','Prob']], how='left',on=['Minute','State','Action'] )
goldMDP4.head(20)
goldMDP4['GivenProb'] = goldMDP4['Prob_x']/goldMDP4['Prob_y']
goldMDP4 = goldMDP4.sort_values('GivenProb',ascending=False)
goldMDP4['Next_Minute'] = goldMDP4['Minute']+1
goldMDP4[(goldMDP4['State']!=goldMDP4['End'])&(goldMDP4['Counter']>10)&(goldMDP4['State']!="WIN")&(goldMDP4['State']!="LOSS")].head(10)
def MCModelv4(data, alpha, gamma, epsilon, reward, StartState, StartMin, StartAction, num_episodes, Max_Mins):
    
    # Initiatise variables appropiately
    
    data['V'] = 0
    data_output = data
    
    outcomes = pd.DataFrame()
    episode_return = pd.DataFrame()
    actions_output = pd.DataFrame()
    V_output = pd.DataFrame()
    
    
    Actionist = [
       'NONE',
       'KILLS', 'OUTER_TURRET', 'DRAGON', 'RIFT_HERALD', 'BARON_NASHOR',
       'INNER_TURRET', 'BASE_TURRET', 'INHIBITOR', 'NEXUS_TURRET',
       'ELDER_DRAGON']
    
    
    for e in range(0,num_episodes):
        action = []
        
        current_min = StartMin
        current_state = StartState
        
        
        
        data_e1 = data
    
    
        actions = pd.DataFrame()

        for a in range(0,100):
            
            action_table = pd.DataFrame()
       
            # Break condition if game ends or gets to a large number of mins 
            if (current_state=="WIN") | (current_state=="LOSS") | (current_min==Max_Mins):
                continue
            else:
                if a==0:
                    data_e1=data_e1
                   
                elif (len(individual_actions_count[individual_actions_count['Action']=="+RIFT_HERALD"])==1):
                    data_e1_e1 = data_e1[(data_e1['Action']!='+RIFT_HERALD')|(data_e1['Action']!='-RIFT_HERALD')]
                    
                elif (len(individual_actions_count[individual_actions_count['Action']=="-RIFT_HERALD"])==1):
                    data_e1 = data_e1[(data_e1['Action']!='+RIFT_HERALD')|(data_e1['Action']!='-RIFT_HERALD')]
                
                elif (len(individual_actions_count[individual_actions_count['Action']=="+OUTER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='+OUTER_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-OUTER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='-OUTER_TURRET']
                    
                elif (len(individual_actions_count[individual_actions_count['Action']=="+INNER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='+INNER_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-INNER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='-INNER_TURRET']
                    
                elif (len(individual_actions_count[individual_actions_count['Action']=="+BASE_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='+BASE_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-BASE_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='-BASE_TURRET']
                    
                elif (len(individual_actions_count[individual_actions_count['Action']=="+INHIBITOR"])==3):
                    data_e1 = data_e1[data_e1['Action']!='+INHIBITOR']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-INHIBITOR"])==3):
                    data_e1 = data_e1[data_e1['Action']!='-INHIBITOR']
                elif (len(individual_actions_count[individual_actions_count['Action']=="+NEXUS_TURRET"])==2):
                    data_e1 = data_e1[data_e1['Action']!='+NEXUS_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-NEXUS_TURRET"])==2):
                    data_e1 = data_e1[data_e1['Action']!='-NEXUS_TURRET']
                
                       
                else:
                    data_e1 = data_e1
                    
                # Break condition if we do not have enough data    
                if len(data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)])==0:
                    continue
                else:             

                    if (a>0) | (StartAction is None):
                        random_action = data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)].sample()
                        random_action = random_action.reset_index()
                        current_action = random_action['Action'][0]
                    else:
                        current_action =  StartAction


                    data_e = data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)&(data_e1['Action']==current_action)]

                    data_e = data_e[data_e['GivenProb']>0]





                    data_e = data_e.sort_values('GivenProb')
                    data_e['CumProb'] = data_e['GivenProb'].cumsum()
                    data_e['CumProb'] = np.round(data_e['CumProb'],4)


                    rng = np.round(np.random.random()*data_e['CumProb'].max(),4)
                    action_table = data_e[ data_e['CumProb'] >= rng]
                    action_table = action_table[ action_table['CumProb'] == action_table['CumProb'].min()]
                    action_table = action_table.reset_index()


                    action = current_action
                    next_state = action_table['End'][0]
                    next_min = current_min+1


                    if next_state == "WIN":
                        step_reward = 10*(gamma**a)
                    elif next_state == "LOSS":
                        step_reward = -10*(gamma**a)
                    else:
                        step_reward = -0.005*(gamma**a)

                    action_table['StepReward'] = step_reward


                    action_table['Episode'] = e
                    action_table['Action_Num'] = a

                    current_action = action
                    current_min = next_min
                    current_state = next_state


                    actions = actions.append(action_table)

                    individual_actions_count = actions


        actions_output = actions_output.append(actions)
                
        episode_return = actions['StepReward'].sum()

                
        actions['Return']= episode_return
                
        data_output = data_output.merge(actions[['Minute','State','Action','End','Return']], how='left',on =['Minute','State','Action','End'])
        data_output['Return'] = data_output['Return'].fillna(0)    
             
        data_output['V'] = data_output['V'] + alpha*(data_output['Return']-data_output['V'])
        data_output = data_output.drop('Return', 1)
        
        V_outputs = pd.DataFrame({'Episode':[e],'V_total':[data_output['V'].sum()]})
        V_output = V_output.append(V_outputs)
        
                
        if current_state=="WIN":
            outcome = "WIN"
        elif current_state=="LOSS":
            outcome = "LOSS"
        else:
            outcome = "INCOMPLETE"
        outcome = pd.DataFrame({'Epsiode':[e],'Outcome':[outcome]})
        outcomes = outcomes.append(outcome)

        
   


    return(outcomes,actions_output,data_output,V_output)
    
alpha = 0.1
gamma = 0.9
num_episodes = 100
epsilon = 0.1


goldMDP4['Reward'] = 0
reward = goldMDP4['Reward']

StartMin = 15
StartState = 'EVEN'
StartAction = None
data = goldMDP4

Max_Mins = 50
start_time = timeit.default_timer()


Mdl4 = MCModelv4(data=data, alpha = alpha, gamma=gamma, epsilon = epsilon, reward = reward,
                StartMin = StartMin, StartState=StartState,StartAction=StartAction, 
                num_episodes = num_episodes, Max_Mins = Max_Mins)

elapsed = timeit.default_timer() - start_time

print("Time taken to run model:",np.round(elapsed/60,2),"mins")
print("Avg Time taken per episode:", np.round(elapsed/num_episodes,2),"secs")
Mdl4[1].head(10)
final_output = Mdl4[2]

final_output2 = final_output[(final_output['Minute']==StartMin)&(final_output['State']==StartState)].sort_values('V',ascending=False).head(10)
final_output2
sum_V_episodes = Mdl4[3]

plt.plot(sum_V_episodes['Episode'],sum_V_episodes['V_total'])
plt.xlabel("Epsiode")
plt.ylabel("Sum of V")
plt.title("Cumulative V by Episode")
plt.show()
from IPython.display import clear_output
def MCModelv5(data, alpha, gamma, epsilon, reward, StartState, StartMin, StartAction, num_episodes, Max_Mins):
    
    # Initiatise variables appropiately
    
    data['V'] = 0
    data_output = data
    
    outcomes = pd.DataFrame()
    episode_return = pd.DataFrame()
    actions_output = pd.DataFrame()
    V_output = pd.DataFrame()
    
    
    Actionist = [
       'NONE',
       'KILLS', 'OUTER_TURRET', 'DRAGON', 'RIFT_HERALD', 'BARON_NASHOR',
       'INNER_TURRET', 'BASE_TURRET', 'INHIBITOR', 'NEXUS_TURRET',
       'ELDER_DRAGON']
        
    for e in range(0,num_episodes):
        clear_output(wait=True)
        
        action = []
        
        current_min = StartMin
        current_state = StartState
        
        
        
        data_e1 = data
    
    
        actions = pd.DataFrame()

        for a in range(0,100):
            
            action_table = pd.DataFrame()
       
            # Break condition if game ends or gets to a large number of mins 
            if (current_state=="WIN") | (current_state=="LOSS") | (current_min==Max_Mins):
                continue
            else:
                if a==0:
                    data_e1=data_e1
                   
                elif (len(individual_actions_count[individual_actions_count['Action']=="+RIFT_HERALD"])==1):
                    data_e1_e1 = data_e1[(data_e1['Action']!='+RIFT_HERALD')|(data_e1['Action']!='-RIFT_HERALD')]
                    
                elif (len(individual_actions_count[individual_actions_count['Action']=="-RIFT_HERALD"])==1):
                    data_e1 = data_e1[(data_e1['Action']!='+RIFT_HERALD')|(data_e1['Action']!='-RIFT_HERALD')]
                
                elif (len(individual_actions_count[individual_actions_count['Action']=="+OUTER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='+OUTER_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-OUTER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='-OUTER_TURRET']
                    
                elif (len(individual_actions_count[individual_actions_count['Action']=="+INNER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='+INNER_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-INNER_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='-INNER_TURRET']
                    
                elif (len(individual_actions_count[individual_actions_count['Action']=="+BASE_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='+BASE_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-BASE_TURRET"])==3):
                    data_e1 = data_e1[data_e1['Action']!='-BASE_TURRET']
                    
                elif (len(individual_actions_count[individual_actions_count['Action']=="+INHIBITOR"])==3):
                    data_e1 = data_e1[data_e1['Action']!='+INHIBITOR']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-INHIBITOR"])==3):
                    data_e1 = data_e1[data_e1['Action']!='-INHIBITOR']
                elif (len(individual_actions_count[individual_actions_count['Action']=="+NEXUS_TURRET"])==2):
                    data_e1 = data_e1[data_e1['Action']!='+NEXUS_TURRET']
                elif (len(individual_actions_count[individual_actions_count['Action']=="-NEXUS_TURRET"])==2):
                    data_e1 = data_e1[data_e1['Action']!='-NEXUS_TURRET']
                
                       
                else:
                    data_e1 = data_e1
                    
                # Break condition if we do not have enough data    
                if len(data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)])==0:
                    continue
                else:             

                    
                    # Greedy Selection:
                    # If this is our first action and start action is non, select greedily. 
                    # Else, if first actions is given in our input then we use this as our start action. 
                    # Else for other actions, if it is the first episode then we have no knowledge so randomly select actions
                    # Else for other actions, we randomly select actions a percentage of the time based on our epsilon and greedily (max V) for the rest 
                    
                    
                    if   (a==0) & (StartAction is None):
                        random_action = data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)].sample()
                        random_action = random_action.reset_index()
                        current_action = random_action['Action'][0]
                    elif (a==0):
                        current_action =  StartAction
                    
                    elif (e==0) & (a>0):
                        random_action = data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)].sample()
                        random_action = random_action.reset_index()
                        current_action = random_action['Action'][0]
                    
                    elif (e>0) & (a>0):
                        epsilon = epsilon
                        greedy_rng = np.round(np.random.random(),2)
                        if (greedy_rng<=epsilon):
                            random_action = data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)].sample()
                            random_action = random_action.reset_index()
                            current_action = random_action['Action'][0]
                        else:
                            greedy_action = (
                            
                                data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)][
                                    
                                    data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)]['V']==data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)]['V'].max()
                                
                                ])
                                
                            greedy_action = greedy_action.reset_index()
                            current_action = greedy_action['Action'][0]
                            
                  
                    
                        
                   
                            
                        
                        
                        
                    


                    data_e = data_e1[(data_e1['Minute']==current_min)&(data_e1['State']==current_state)&(data_e1['Action']==current_action)]

                    data_e = data_e[data_e['GivenProb']>0]





                    data_e = data_e.sort_values('GivenProb')
                    data_e['CumProb'] = data_e['GivenProb'].cumsum()
                    data_e['CumProb'] = np.round(data_e['CumProb'],4)


                    rng = np.round(np.random.random()*data_e['CumProb'].max(),4)
                    action_table = data_e[ data_e['CumProb'] >= rng]
                    action_table = action_table[ action_table['CumProb'] == action_table['CumProb'].min()]
                    action_table = action_table.reset_index()


                    action = current_action
                    next_state = action_table['End'][0]
                    next_min = current_min+1


                    if next_state == "WIN":
                        step_reward = 10*(gamma**a)
                    elif next_state == "LOSS":
                        step_reward = -10*(gamma**a)
                    else:
                        step_reward = -0.005*(gamma**a)

                    action_table['StepReward'] = step_reward


                    action_table['Episode'] = e
                    action_table['Action_Num'] = a

                    current_action = action
                    current_min = next_min
                    current_state = next_state


                    actions = actions.append(action_table)

                    individual_actions_count = actions
                    
        print("Current progress:", np.round((e/num_episodes)*100,2),"%")

        actions_output = actions_output.append(actions)
                
        episode_return = actions['StepReward'].sum()

                
        actions['Return']= episode_return
                
        data_output = data_output.merge(actions[['Minute','State','Action','End','Return']], how='left',on =['Minute','State','Action','End'])
        data_output['Return'] = data_output['Return'].fillna(0)    
             
            
        data_output['V'] = np.where(data_output['Return']==0,data_output['V'],data_output['V'] + alpha*(data_output['Return']-data_output['V']))
        
        data_output = data_output.drop('Return', 1)

        
        for actions in data_output[(data_output['Minute']==StartMin)&(data_output['State']==StartState)]['Action'].unique():
            V_outputs = pd.DataFrame({'Index':[str(e)+'_'+str(actions)],'Episode':e,'StartMin':StartMin,'StartState':StartState,'Action':actions,
                                      'V':data_output[(data_output['Minute']==StartMin)&(data_output['State']==StartState)&(data_output['Action']==actions)]['V'].sum()
                                     })
            V_output = V_output.append(V_outputs)
        
        if current_state=="WIN":
            outcome = "WIN"
        elif current_state=="LOSS":
            outcome = "LOSS"
        else:
            outcome = "INCOMPLETE"
        outcome = pd.DataFrame({'Epsiode':[e],'Outcome':[outcome]})
        outcomes = outcomes.append(outcome)

        
   


    return(outcomes,actions_output,data_output,V_output)
    
alpha = 0.3
gamma = 0.9
num_episodes = 1000
epsilon = 0.2


goldMDP4['Reward'] = 0
reward = goldMDP4['Reward']

StartMin = 15
StartState = 'EVEN'
StartAction = None
data = goldMDP4

Max_Mins = 50
start_time = timeit.default_timer()


Mdl5 = MCModelv5(data=data, alpha = alpha, gamma=gamma, epsilon = epsilon, reward = reward,
                StartMin = StartMin, StartState=StartState,StartAction=StartAction, 
                num_episodes = num_episodes, Max_Mins = Max_Mins)

elapsed = timeit.default_timer() - start_time

print("Time taken to run model:",np.round(elapsed/60,2),"mins")
print("Avg Time taken per episode:", np.round(elapsed/num_episodes,2),"secs")
Mdl5[3].sort_values(['Episode','Action']).head(10)
V_episodes = Mdl5[3]

plt.figure(figsize=(20,10))

for actions in V_episodes['Action'].unique():
    plot_data = V_episodes[V_episodes['Action']==actions]
    plt.plot(plot_data['Episode'],plot_data['V'])
plt.xlabel("Epsiode")
plt.ylabel("V")
plt.title("V for each Action by Episode")
#plt.show()

final_output = Mdl5[2]


final_output2 = final_output[(final_output['Minute']==StartMin)&(final_output['State']==StartState)]
final_output3 = final_output2.groupby(['Minute','State','Action']).sum().sort_values('V',ascending=False).reset_index()
final_output3[['Minute','State','Action','V']]
single_action1 = final_output3['Action'][0]
single_action2 = final_output3['Action'][len(final_output3)-1]

plot_data1 = V_episodes[(V_episodes['Action']==single_action1)]
plot_data2 = V_episodes[(V_episodes['Action']==single_action2)]

plt.plot(plot_data1['Episode'],plot_data1['V'], label = single_action1)
plt.plot(plot_data2['Episode'],plot_data2['V'], label = single_action2)
plt.xlabel("Epsiode")
plt.ylabel("V")
plt.legend()
plt.title("V by Episode for the Best/Worst Actions given the Current State")
plt.show()