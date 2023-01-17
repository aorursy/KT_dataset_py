import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from scipy.interpolate import interp1d





drop_cols = ['National_Position','National_Kit','Club_Kit','Club_Joining','Contract_Expiry']



#Read_csv will read a csv file and put it in pandas DataFrame.

Players_Data = pd.read_csv('../input/FullData.csv')

#Setting the date colum to a datetime format.

Players_Data['Club_Joining'] = pd.to_datetime(Players_Data['Club_Joining'])

#Sorting players by date to keep only the leatest attributes in a  descending order,as some players

#are registered in different clubs due to transfers.

Players_Data = Players_Data[~Players_Data.Club_Position.isnull()].sort_values('Club_Joining', ascending=False)

#keeping the first as it will be the leatest attributes and club for the player.

Players_Data = Players_Data.drop_duplicates(subset='Name', keep='first')

#Delete the unwanted columes

Players_Data = Players_Data.drop(drop_cols, axis=1)

#Resort Players by making top players first

Players_Data = Players_Data.sort_values('Rating', ascending=False)

#Re-arrange the DataFrame's indexing

#drop=True --> To delete the old indexing after sorting

Players_Data = Players_Data.reset_index(drop=True)

Players_Data.head()
#Get age of all players

age_player = Players_Data["Age"]



sns.kdeplot(age_player,shade=True, color="b")

plt.xlabel('Ages', fontsize=30)

plt.ylabel('Denisty', fontsize=30)

plt.title('Players Ages Probabilites')
#Get ratings of all players

rating_player = Players_Data["Rating"]



sns.kdeplot(rating_player,shade=True, color="y")

plt.xlabel('Ratings', fontsize=30)

plt.ylabel('Denisty', fontsize=30)

plt.title('Players Rating Probabilites')
Free_Agents = Players_Data[Players_Data.Club == 'Free Agents']

Club_Agents = Players_Data[Players_Data.Club != 'Free Agents']



fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(222)

ax1.set_title('Free Agenst Age-Rating Relation')

ax2 = fig.add_subplot(221)

ax2.set_title('Club Agenst Age-Rating Relation')



sns.kdeplot(Free_Agents.Age,Free_Agents.Rating, shade=True,cmap="Reds",ax=ax1)

sns.kdeplot(Club_Agents.Age,Club_Agents.Rating, shade=True,cmap="Blues",ax=ax2)
sns.kdeplot(Free_Agents.Age, Free_Agents.Rating,cmap="Reds", shade=True, shade_lowest=False)

sns.kdeplot(Club_Agents.Age,Club_Agents.Rating ,cmap="Blues", shade=True, shade_lowest=False)
#Lists of all positions

Defenders_POS =['CB','RCB','LCB','RB','LB','LWB','RWB']

Strikers_POS =['CF','RW','LW','RF','LF','RS','LS','ST']

Midfielders_POS =['CDM','CM','CAM','RM','LM','LCM','RCM','LDM','RDM','LAM','RAM']

GoalKeepers_POS =['GK']



#Get all defenders data

defenders = Players_Data[Players_Data.Club_Position.isin(Defenders_POS)]

#Groupby Age and Ratings as These are the only data we need for rating analysis

group = defenders.groupby("Age")["Rating"].mean().reset_index()

#Get defenders age data

age_DF = group["Age"]

#Get defenders rating data

rating_DF = group["Rating"]

#Interpolate for a better plotting-->[to predict the missing values between given data]

FAR_DF = interp1d(age_DF, rating_DF, kind='cubic')





#Get all strikers data

Strikers = Players_Data[Players_Data.Club_Position.isin(Strikers_POS)]

group = Strikers.groupby("Age")["Rating"].mean().reset_index()

age_ST = group["Age"]

rating_ST = group["Rating"]

FAR_ST = interp1d(age_ST, rating_ST, kind='cubic')





#Get all midfielders data

Midfielders = Players_Data[Players_Data.Club_Position.isin(Midfielders_POS)]

group = Midfielders.groupby("Age")["Rating"].mean().reset_index()

age_MF = group["Age"]

rating_MF = group["Rating"]

FAR_MF = interp1d(age_MF, rating_MF, kind='cubic')



#Get all goalkeepers data

goalkeepers = Players_Data[Players_Data.Club_Position.isin(GoalKeepers_POS)]

group = goalkeepers.groupby("Age")["Rating"].mean().reset_index()

age_GK = group["Age"]

rating_GK = group["Rating"]

FAR_GK = interp1d(age_GK, rating_GK, kind='cubic')



#Players age range to be predictied on using interpolated data

agenew = np.linspace(18,38, num=100, endpoint=True)





plt.xlabel('Age', fontsize=30)

plt.ylabel('Rating', fontsize=30)

plt.title('Mean of Players(XI) Ratings According To Age',fontsize=14)

plt.plot( agenew, FAR_MF(agenew), "-", agenew, FAR_DF(agenew), "-",agenew, FAR_GK(agenew), "-",agenew, FAR_ST(agenew), "-")

plt.legend(["Midfielders", "Defenders","Goalkeepers","Forwards"], loc='best', fontsize = 20)

#function to calculate correlation between attributes

def correlation(x,y):

    

    std_x = (x-x.mean())/x.std(ddof=0)

    std_y = (y-y.mean())/y.std(ddof=0)



    return (std_x*std_y).mean()
#function to add attributes to each player based on fifa calculations

def Calculate_Players_Attributes(test):



    for index, row in test.iterrows():

        

        test['BallSkills'] = (test['Ball_Control']+test['Dribbling'])/2

        test['Defence'] = (test['Marking']+test['Standing_Tackle']+test['Sliding_Tackle'])/3

        test['Mental'] = (test['Aggression']+test['Reactions']+test['Attacking_Position']+test['Interceptions']+test['Vision'])/5

        test['Passing'] = (test['Crossing']+test['Short_Pass']+test['Long_Pass'])/3

        test['Physical'] = (test['Acceleration']+test['Stamina']+test['Strength']+test['Balance']+test['Speed']+test['Agility']+test['Jumping'])/7

        test['Shooting'] = (test['Heading']+test['Shot_Power']+test['Finishing']+test['Long_Shots']+test['Curve']+test['Freekick_Accuracy']+test['Penalties']+test['Volleys'])/8

        test['GoalKeeping'] = (test['GK_Positioning']+test['GK_Diving']+test['GK_Handling']+test['GK_Reflexes']+test['GK_Kicking'])/5

    

    return test    

                                                                                                                                          
#we could have used apply,but this way is simpler

Players_attributes = Calculate_Players_Attributes(Players_Data)

Players_attributes.head()
print(correlation(Players_attributes['Defence'],Players_attributes['Mental']))
print(correlation(Players_attributes['Defence'],Players_attributes['Aggression']))
print(correlation(Players_attributes['Defence'],Players_attributes['Interceptions']))


sns.regplot(x="Interceptions", y="Defence", data=Players_attributes,color="k")

sns.jointplot(x='Interceptions',y='Defence',data=Players_attributes, kind='hex', color="k")



#Get all defenders data

defenders = Players_attributes[Players_attributes.Club_Position.isin(Defenders_POS)]

sns.jointplot(x='Interceptions',y='Rating',data=defenders, kind='hex', color="k")
sns.regplot(x="Aggression", y="Defence", data=Players_attributes,color="b")

sns.jointplot(x='Aggression',y='Defence',data=Players_attributes, kind='hex', color="b")



#Get all defenders data

defenders = Players_attributes[Players_attributes.Club_Position.isin(Defenders_POS)]

sns.jointplot(x='Aggression',y='Rating',data=defenders, kind='hex', color="b")
print(correlation(Players_attributes['Passing'],Players_attributes['BallSkills']))
print(correlation(Players_attributes['Shooting'],Players_attributes['BallSkills']))