# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.offline as py
py.init_notebook_mode(connected=False)
import plotly.graph_objs as go
import plotly.tools as tls
import datetime as dt
import networkx as nx 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
matches=pd.read_csv('../input/ContinousDataset.csv') 
# Any results you write to the current directory are saved as output.
def hwin(s):
      if (s['Winner'] == s['Team 1']  and s['Venue_Team1'] == "Home"):
              return s['Team 1']
      elif (s['Winner'] == s['Team 2']  and s['Venue_Team2'] == "Home"):
              return s['Team 2']              
      else :
          return np.nan

def awin(s):
      if (s['Winner'] == s['Team 1']  and s['Venue_Team1'] in( "Away",'Neutral')):
              return s['Team 1']
      elif (s['Winner'] == s['Team 2']  and s['Venue_Team2'] in( "Away",'Neutral')):
              return s['Team 2']
              
      else :
          return np.nan

def hloss(s):
      if (s['Winner'] != s['Team 1']  and s['Venue_Team1'] == "Home"):
              return s['Team 1']
      elif (s['Winner'] != s['Team 2']  and s['Venue_Team2'] == "Home"):
              return s['Team 2']
              
      else :
          return np.nan

def aloss(s):
      if (s['Winner'] != s['Team 1']  and s['Venue_Team1'] in( "Away",'Neutral')):
              return s['Team 1']
      elif (s['Winner'] != s['Team 2']  and s['Venue_Team2'] in( "Away",'Neutral')):
              return s['Team 2']          
      else :
          return np.nan

def FirstInningsWinner(s):
      if (s['Winner'] == s['Team 1']  and s['Innings_Team1'] in( "First")):
              return s['Team 1']
      elif (s['Winner'] == s['Team 2']  and s['Innings_Team2'] in( "First")):
              return s['Team 2']          
      else :
          return np.nan

matches['Year']=matches['Match Date'].str[-4:]
matches_played_byteams=pd.concat([matches['Team 1'],matches['Team 2']])
matches_played_byteams=matches_played_byteams.value_counts().reset_index()
matches_played_byteams.columns=['Team','Total Matches']
matches['HomeWin']= matches.apply(hwin,axis=1)
matches['AwayWin']= matches.apply(awin,axis=1)
matches['HomeLoss']= matches.apply(hloss,axis=1)
matches['AwayLoss']= matches.apply(aloss,axis=1)
matches['FistInningsWinner']= matches.apply(FirstInningsWinner,axis=1)
matches_played_byteams.set_index('Team',inplace=True)


Winner=matches['Winner'].value_counts().sort_values( ascending= False ).reset_index()
Winner.set_index('index',inplace=True)
HomeWin=matches['HomeWin'].value_counts().sort_values( ascending= False ).reset_index()
HomeWin.set_index('index',inplace=True)
AwayWin=matches['AwayWin'].value_counts().sort_values( ascending= False ).reset_index()
AwayWin.set_index('index',inplace=True)
HomeLoss=matches['HomeLoss'].value_counts().sort_values( ascending= False ).reset_index()
HomeLoss.set_index('index',inplace=True)
AwayLoss=matches['AwayLoss'].value_counts().sort_values( ascending= False ).reset_index()
AwayLoss.set_index('index',inplace=True)

FistInningsWinner=matches['FistInningsWinner'].value_counts().sort_values( ascending= False ).reset_index()
FistInningsWinner.set_index('index',inplace=True)

matches_played_byteams['Total Matches'] =matches_played_byteams['Total Matches']/2
matches_played_byteams['wins']=Winner['Winner']/2
matches_played_byteams['HomeWin']=HomeWin['HomeWin']/2
matches_played_byteams['AwayWin']=AwayWin['AwayWin']/2
matches_played_byteams['HomeLoss']=HomeLoss['HomeLoss']/2
matches_played_byteams['AwayLoss']=AwayLoss['AwayLoss']/2
matches_played_byteams['FistInningsWinner']=FistInningsWinner['FistInningsWinner']/2
matches_played_byteams['SecondInningsWinner']=matches_played_byteams['wins']-matches_played_byteams['FistInningsWinner']
matches_played_byteams['HomeTotal']=matches_played_byteams['HomeWin']+ matches_played_byteams['HomeLoss']
matches_played_byteams['AwayTotal']=matches_played_byteams['AwayWin']+ matches_played_byteams['AwayLoss']


print(matches_played_byteams)


barWidth = 0.85
plt.subplots(figsize=(12,6))
plt.bar(matches_played_byteams.index, matches_played_byteams['Total Matches'], color='#b5ffb9', edgecolor='white', width=barWidth)
plt.bar(matches_played_byteams.index, matches_played_byteams['wins'], color='#f9bc86', edgecolor='red', width=barWidth)
plt.xticks(matches_played_byteams.index, matches_played_byteams.index,rotation='vertical')
plt.xlabel("group")
plt.title('Total Match vs wins',size=25)
plt.show()        


barWidth = 0.85
plt.subplots(figsize=(12,6))
plt.bar(matches_played_byteams.index, matches_played_byteams['HomeTotal'], color='#b5ffb9', edgecolor='white', width=barWidth)
plt.bar(matches_played_byteams.index, matches_played_byteams['HomeWin'], color='#f9bc86', edgecolor='red', width=barWidth)
plt.xticks(matches_played_byteams.index, matches_played_byteams.index,rotation='vertical')
plt.xlabel("group")
plt.title('Home Ground vs wins',size=25)
plt.show()        

s=matches_played_byteams[['FistInningsWinner','SecondInningsWinner']].plot(kind="bar",figsize=(12,6),fontsize=12) 
plt.xlabel("group")
plt.title('InningWisewins',size=25)
plt.legend(loc = 'upper right')
plt.show()        



barWidth = 0.85
plt.subplots(figsize=(12,6))
plt.bar(matches_played_byteams.index, matches_played_byteams['AwayTotal'], color='#b5ffb9', edgecolor='white', width=barWidth)
plt.bar(matches_played_byteams.index, matches_played_byteams['AwayWin'], color='#f9bc86', edgecolor='red', width=barWidth)
plt.xticks(matches_played_byteams.index, matches_played_byteams.index,rotation='vertical')
plt.xlabel("group")
plt.title('Outside Ground vs wins',size=25)
plt.show()        

DFGrounds=matches['Ground'].value_counts().reset_index()
DFGrounds.columns=['Ground','Total Matches']
DFGrounds = DFGrounds.sort_values(by ='Total Matches', ascending = False)[:20]
DFGrounds.set_index('Ground',inplace=True)
barWidth = 0.85
plt.subplots(figsize=(12,6))
plt.bar(DFGrounds.index, DFGrounds['Total Matches'], color='#b5ffb9', edgecolor='white', width=barWidth)
plt.xticks(rotation='vertical')
plt.xlabel("group")
plt.title('Total Matches played per ground',size=25)
plt.show()        

HostCountry=matches['Host_Country'].value_counts().reset_index()
HostCountry.columns=['Host_Country','Total Matches']
HostCountry = HostCountry.sort_values(by ='Total Matches', ascending = False)
HostCountry.set_index('Host_Country',inplace=True)
barWidth = 0.85
plt.subplots(figsize=(12,6))
plt.bar(HostCountry.index, HostCountry['Total Matches'], color='#b5ffb9', edgecolor='white', width=barWidth)
plt.xticks(rotation='vertical')
plt.xlabel("group")
plt.title('Total Matches hosted by Country',size=25)
plt.show()        


Year=matches['Year'].value_counts().sort_values( ascending= False ).reset_index()
Year.columns=['Year','Total Matches']
Year.set_index('Year',inplace=True)
print(Year)
barWidth = 0.85
plt.subplots(figsize=(12,6))
plt.bar(Year.index, Year['Total Matches'], color='#b5ffb9', edgecolor='white', width=barWidth)
plt.xticks(Year.index, Year.index,rotation='vertical')
plt.xlabel("group")
plt.title('Total Matches per year',size=25)
plt.show()        


def interactions(year,color):
    df  = matches[matches["Year"] == year][["Team 1","Team 2"]]
    G   = nx.from_pandas_dataframe(df,"Team 1","Team 2")    
    plt.figure(figsize=(10,9))    
    nx.draw_kamada_kawai(G,with_labels = True,
                         node_size  = 2500,
                         node_color = color,
                         node_shape = "h",
                         edgecolor  = "k",
                         linewidths  = 5 ,
                         font_size  = 13 ,
                         alpha=.8)
    
    plt.title("Interaction between teams :" + str(year) , fontsize =13 , color = "navy")
    
interactions(2014,"r")  