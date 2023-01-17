# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt

from pylab import rcParams



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/data.csv')

df.describe()
df.columns

sb.set()

sb.pairplot(df[['Age','Overall','Potential','Stamina','Strength']])

# putting constraints to take players from top-6 clubs in world to form the best-11 team!

selected_players=[]

for i in range(len(df)-1):

    if( df.Overall[i]>85 and df.Potential[i]>80 and df.Special[i]>2000 and df.Agility[i]>75 and df.Stamina[i]>55 and df.Strength[i]>55 and (df.Club[i]=='Juventus' or df.Club[i]=='Real Madrid' or df.Club[i]=='FC Barcelona' or df.Club[i]=='Paris Saint-Germain'

        or df.Club[i]=='Manchester United' or df.Club[i]=='Chelsea') and (df.Crossing[i]>75 and df.Finishing[i]>75 and df.HeadingAccuracy[i]>60 and df.Volleys[i]>65 and df.Dribbling[i]>72 and df.BallControl[i]>73 and df.Acceleration[i]>75  and df.SprintSpeed[i]>75       

           and df.Balance[i]>60) ) :

            selected_players.append(i)  #Index of selected Forwards

            print(df.Name[i])

    elif (df.Aggression[i]>60 and df.Interceptions[i]>70 and df.Positioning[i]>72 and df.StandingTackle[i]>72 and 

         df.SlidingTackle[i]>69 and df.Marking[i]>79 and  (df.Club[i]=='Juventus' or df.Club[i]=='Real Madrid' or df.Club[i]=='FC Barcelona' 

          or df.Club[i]=='Paris Saint-Germain' or df.Club[i]=='Manchester United' or df.Club[i]=='Chelsea') ):

            selected_players.append(i) #Index of selected Mid players

            print(df.Name[i])

    elif (df.GKDiving[i]>=90 and df.GKHandling[i]>=85 and  df.GKKicking[i]>=85 and  df.GKReflexes[i]>=92 and (df.Club[i]=='Juventus' or

         df.Club[i]=='Real Madrid' or df.Club[i]=='FC Barcelona' or df.Club[i]=='Paris Saint-Germain' or df.Club[i]=='Manchester United')):

            selected_players.append(i) #Index of Goalkeeper

            print(df.Name[i])
#age of players in team

sb.countplot(df.Age[selected_players],data=df,color='green')

plt.title('Age of selected players')


        
# To know value based overall performance

plt.scatter(x=df.Overall[selected_players],y=df.Value[selected_players],marker='*',cmap='hot',color='red')  

# most players from single club

rcParams['figure.figsize']=10,15

sb.countplot(df.Club[selected_players],color='orange')

df.Club[selected_players].max() 
# to identify best players in club

for i in selected_players:

    if df.Overall[i]==df.Overall[selected_players].max():

        print('Overall best: '+df.Name[i]+' '+str(df.Overall[selected_players].max()))

    elif df.Acceleration[i]==df.Acceleration[selected_players].max():

        print('Max accelaration: '+df.Name[i]+' '+str(df.Acceleration[selected_players].max()))

    elif df.Interceptions[i]==df.Interceptions[selected_players].max():

        print('Max Interceptions: '+df.Name[i]+' '+str(df.Interceptions[selected_players].max()))

    elif df.Aggression[i]==df.Aggression[selected_players].max():

        print('Most aggresive: '+df.Name[i]+' '+str(df.Aggression[selected_players].max()))

    elif df.GKDiving[i]==df.GKDiving[selected_players].max():

        print('Best goalkeeper: '+df.Name[i]+' '+str(df.GKDiving[selected_players].max()))