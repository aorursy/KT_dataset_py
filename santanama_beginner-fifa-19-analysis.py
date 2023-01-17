import numpy as np 

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

from matplotlib import *

%matplotlib inline

import seaborn as sns

import re



print('OK')



#Opening Data

data= pd.read_csv('../input/fifa19/data.csv', index_col=0) 

data.head()
#Data Size(Rows, Columns)

data.shape
#Data Correlaction

data_corr= data.corr()

plt.figure(figsize= (30,18))

sns.heatmap(data= data_corr, annot= True)

plt.title('Full Df Correlaction Heatmap', fontsize= 30)

plt.show()
#All Data collumns

data.columns
drop= data.drop(columns=['Flag','Club Logo','Photo','ID','Composure','Marking','StandingTackle','GKDiving','GKHandling','GKKicking',

                      'GKPositioning','GKReflexes','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties',

                      'SlidingTackle','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower',

                      'Jumping','RB','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy',

                      'LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM','ST','RS','LW',

                      'LF','CF','RF','Joined','Loaned From','Skill Moves','Work Rate','Weak Foot','LS','Body Type','Real Face','Special', 'Release Clause'])



#Renaming columns

df= drop.rename(columns={'Preferred Foot':'Preferred_Foot', 'International Reputation':

                         'International_Reputation', 'Jersey Number': 'Jersey_Number', 

                         'Contract Valid Until':'Contract_Valid_Until'})

df.head()
df.info()
#How many countries does Fifa 19 has?

df.Nationality.nunique()
#Number Of Players By Country

country_data= data.Nationality.value_counts()

country= pd.DataFrame(country_data)

country.reset_index(level=0 , inplace=True)

country_count= country.rename(columns= {'index': 'Country', 'Nationality': 'Number of Players'})



#Plotting

plt.figure(figsize = (28,15))

sns.barplot(data=country_count, x='Country', y='Number of Players', color= 'lightblue')

plt.title('Number Of Players By Country', fontsize= 25)

sns.set_style("whitegrid")

plt.xticks(rotation= 90)

plt.show()
#Country Overall

nat= df.groupby('Nationality')['Overall'].mean()

nat_ovr= pd.DataFrame(nat)

nat_ovr.reset_index()



#Plotting

plt.figure(figsize = (10,8))

sns.distplot(a= nat_ovr['Overall'], kde= False)

plt.title('Country Overall', fontsize= 15)

plt.show()
#Preferred Foot By Players

preferred =  df.Preferred_Foot.value_counts()

group3 = pd.DataFrame(preferred)

group3.reset_index(level=0, inplace=True) 

foot = group3.rename(columns={'index': 'Foot', 'Preferred_Foot': 'Number of players'})



#Plotting

plt.figure(figsize = (10,8))

sns.barplot(data= foot, x= 'Number of players', y= 'Foot' , color= 'orange')

plt.title('Preferred Foot by Players', fontsize= 15)

sns.set_style("whitegrid")

plt.show()
#Preferred Foot By Position

df_pf= df.groupby(['Preferred_Foot', 'Position']).size()

df2= pd.DataFrame(df_pf)

df3= df2.reset_index()

df3.rename(columns={0: 'Size'}, inplace=True)

df3.set_index('Preferred_Foot')



#Plotting

plt.figure(figsize = (25,10))

sns.barplot(x='Position', y="Size", hue='Preferred_Foot', data=df3) 

plt.title('Preferred Foot By Position', fontsize=15)

sns.set_style("whitegrid")

plt.show()
#Age Normal Distribution

plt.figure(figsize= (10,8))

sns.kdeplot(data=data['Age'], shade=True, color= 'green')

plt.title('Age Normal Distribution', fontsize=15)

sns.set_style("whitegrid")

plt.legend()

plt.show()
#Age & Overall Comparision

plt.figure(figsize= (10,8))

sns.scatterplot(x=df['Age'], y=df['Overall'], color= 'green')

plt.title('Age & Overall Comparision', fontsize=15)

plt.legend()

plt.show()



#If we take a look, the most 80+ players are found between 20-35 y/o.
#Age & Potential

plt.figure(figsize= (10,8))

sns.scatterplot(x=df['Age'], y=df['Potential'], color= 'red')

plt.title('Age & Overall Comparision', fontsize=15)

plt.legend()

plt.show()