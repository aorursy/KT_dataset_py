import numpy as np
import pandas as pd 
import os
pd.options.display.max_rows = 999
data_2007 = pd.read_excel('../input/nba odds 2007-08.xlsx')
data_2008 = pd.read_excel('../input/nba odds 2008-09.xlsx')
data_2009 = pd.read_excel('../input/nba odds 2009-10.xlsx')
data_2010 = pd.read_excel('../input/nba odds 2010-11.xlsx')
data_2011 = pd.read_excel('../input/nba odds 2011-12.xlsx')
data_2012 = pd.read_excel('../input/nba odds 2012-13.xlsx')
data_2013 = pd.read_excel('../input/nba odds 2013-14.xlsx')
data_2014 = pd.read_excel('../input/nba odds 2014-15.xlsx')
data_2015 = pd.read_excel('../input/nba odds 2015-16.xlsx')
data_2016 = pd.read_excel('../input/nba odds 2016-17.xlsx')
data_2017 = pd.read_excel('../input/nba odds 2017-18.xlsx')

data_2007.head(50)
#Create a cleaning function to modify all the datasets into a usable format for analysis:
def clean(dataset):
    #get rid of extra stuff. I will be focusing on closed odds only. 
    clean_data = dataset.drop(labels=['Rot','1st','2nd','3rd','4th','2H','Open'], axis =1)
    clean_data.rename(columns={'Date':'date','VH':'home_away','Team':'team','Final':'points_scored','Close':'points_spread','ML':'moneyline'}, inplace=True)
    clean_data.replace('V','A',inplace=True)
    
    #reorganize columns so that each game has one line instead of home team stats stacked on away team stats
    away = clean_data.iloc[::2]
    away = away.reset_index(drop=True)
    away.rename(columns={'team':'away_team','points_spread':'points_over_under','moneyline':'away_moneyline','points_scored':'away_points_scored'}, inplace=True)
    away.drop(labels=['date','home_away'],inplace=True, axis=1)
    
    home = clean_data.iloc[1::2]
    home = home.reset_index(drop=True)
    home = home.drop(labels='home_away',axis=1)
    home.rename(columns={'team':'home_team','moneyline':'home_moneyline','points_scored':'home_points_scored'}, inplace=True)
    
    #Here is where I realise I forgot to do a bunch of cleaning stuff and then I do it
    final = pd.concat([home,away],axis=1)
    final = final[['date','home_team','home_points_scored','home_moneyline','away_team','away_points_scored','away_moneyline','points_spread','points_over_under']]
    final.rename(columns={'points_spread':'odds1','points_over_under':'odds2'}, inplace=True)
    final.replace('pk',0,inplace=True)
    final['betting_spread'] = final[['odds1','odds2']].min(axis=1)
    final['actual_spread'] = abs(final['home_points_scored']-final['away_points_scored'])
    final.drop(labels=['odds1','odds2','betting_spread', 'actual_spread'],inplace=True, axis=1)
          
    return final
#noticed there were some missing odds, I will locate them here:
clean_2007 = clean(data_2007)
errors = clean_2007[(clean_2007['home_moneyline']== 'NL') | (clean_2007['away_moneyline']== 'NL')]
errors
#in 2007 New York won the superbowl. The only New York NBA team in 2007 was the New York Knicks.
#Since we only care about Knicks games, I will remove the errored rows altogether.
clean_2007 = clean_2007.drop(clean_2007.index[[32,1092,1284]])
knicks_2007 = clean_2007[(clean_2007['home_team'] == 'NewYork') | (clean_2007['away_team'] == 'NewYork')]
knicks_2007.head(10)
#make a simulator for bettings. moneyline_range lets you only bet on close games. I don't want to bet on extreme underdogs. 
def bettor(start_index, end_index, moneyline_range, team, wager, df):
    
    winnings = 0
    
    for index, row in df.iterrows():
        #if the game is within the range we selected:
        if (index >= start_index) & (index <= end_index):
            #if we bet on the home team and the odds are within our range
            if (row['home_team'] == team) & (abs(row['home_moneyline']) <= moneyline_range):
                #if we won
                if row['home_points_scored'] > row['away_points_scored']:
                    if row['home_moneyline'] < 0:
                        #winnings if we were favored
                        winnings += ( (wager/row['home_moneyline'])*wager )
                        print(winnings,'home win')
                    else:
                        #winnings if we were underdogs
                        winnings += ( (row['home_moneyline']/wager)*wager )   
                        print(winnings, 'home win')
                #if we lost
                else:
                    winnings -= wager
                    print(winnings, 'home loss')
                               
                
            #if we bet on the away team 
            elif (row['away_team'] == team) & (abs(row['away_moneyline']) <= moneyline_range):
                #if we won
                if row['away_points_scored'] > row['home_points_scored']:
                    if row['away_moneyline'] < 0:
                        #winnings if we were favored
                        winnings += ( (wager/row['home_moneyline'])*wager )
                        print(winnings, 'away win')
                    else:
                        #winnings if we were underdogs
                        winnings += ( (row['away_moneyline']/wager)*wager )
                        print(winnings,'away win')
                 #if we lost
                else:
                    winnings -= wager
                    print(winnings, 'away loss')
            else:
                print('we did not bet')
                
    return winnings
   
    
#testing bettor function on the whole season:
bettor(0,1218,300,'NewYork',100,knicks_2007)
#let's only bet on the games for the month after New York won the superbowl:
#New York wins the Superbowl on Feb 3, 2008. This would apply to the 2007-2008 NBA season which I refer to as 2007.
bettor(885,1127,300,'NewYork',100, knicks_2007)
clean_2008 = clean(data_2008)
clean_2009 = clean(data_2009)
clean_2010 = clean(data_2010)
clean_2011 = clean(data_2011)
clean_2012 = clean(data_2012)
clean_2013 = clean(data_2013)
clean_2014 = clean(data_2014)
clean_2015 = clean(data_2015)
clean_2016 = clean(data_2016)
clean_2017 = clean(data_2017)
#I'll ignore the 2008-2009 season because Pittsburgh won the Superbowl and doesn't have an NBA team
#New Orleans wins the Superbowl on Feb 7, 2010
nola_2009 = clean_2009[(clean_2009['home_team'] == 'NewOrleans') | (clean_2009['away_team'] == 'NewOrleans')]
nola_2009
bettor(762,942,300,'NewOrleans',100, nola_2009)
#Green Bay wins the Superbowl on Feb 6, 2011. There is no NBA team in Green bay. I'll ignore this year.
#New York wins the Superbowl on Feb 5, 2012. At this point there is still only the Knicks in New York. 
#clean_2011.head(50)
ny_2011 = clean_2011[(clean_2011['home_team'] == 'NewYork') | (clean_2011['away_team'] == 'NewYork')]
ny_2011
bettor(361,567,300,'NewYork',100, ny_2011)
#Baltimore wins the Superbowl on Feb 3, 2013. They don't have an NBA team. 
#Seattle wins the Superbowl on Feb 2, 2014. They don't have an NBA team.
#New England wins the Superbowl on Feb 1, 2015. I will assume the Boston Celtics are in the same 'city'.
#clean_2014.head(20)
celtics_2014 = clean_2014[(clean_2014['home_team'] == 'Boston') | (clean_2014['away_team'] == 'Boston')]
celtics_2014
bettor(724,877,300,'Boston',100, celtics_2014)
#Denver wins the Superbowl on Feb 7, 2016.
#clean_2015.head(20)
nuggets_2015 = clean_2015[(clean_2015['home_team'] == 'Denver') | (clean_2015['away_team'] == 'Denver')]
nuggets_2015
bettor(776,946,300,'Denver',100, nuggets_2015)
#New England wins the Superbowl on Feb 5, 2017.
#clean_2016.head(20)
celtics_2016 = clean_2016[(clean_2016['home_team'] == 'Boston') | (clean_2016['away_team'] == 'Boston')]
celtics_2016
bettor(792,943,300,'Boston',100, celtics_2016)

