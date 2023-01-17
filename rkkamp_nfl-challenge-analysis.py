
import numpy as np 
import pandas as pd 


import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
#Combining a few tables to get a clean look at the details of plays that resulted in concussions

video_review = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
video_review.head()
position = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
position.head()
concussion_play_data = pd.merge(video_review,position)
concussion_play_data.head()
concussion_play_data.Role.value_counts()
concussion_play_data.groupby('Player_Activity_Derived').Role.value_counts()
team = {'GL':'Punting','PLT':'Punting','PLG':'Punting','PLS':'Punting','PRG':'Punting','PRT':'Punting','GR':'Punting','PLW':'Punting','PRW':'Punting','PC':'Punting','PPR':'Punting','P':'Punting','VRo':"Receiving",'VRi':"Receiving",'PDR1':"Receiving",'PDR2':"Receiving",'PDR3':"Receiving",'PDL3':"Receiving",'PDL2':"Receiving",'PDL1':"Receiving",'PLR':"Receiving",'PLM':"Receiving",'PLL':"Receiving",'PFB':"Receiving",'PR':"Receiving",'PDL4':"Receiving",'VLo':"Receiving",'VR':'Receiving'}

concussion_play_data['Team']=concussion_play_data.Role.replace(team)
concussion_play_data.Team.value_counts()
concussion_play_data.groupby('Team')['Player_Activity_Derived'].value_counts()
sns.countplot(x='Player_Activity_Derived', data=concussion_play_data, hue='Team')
concussion_play_data['Turnover_Related'].value_counts()
concussion_play_data.groupby('Team').Role.value_counts()
sns.countplot(x='Role', hue='Team', data = concussion_play_data)
concussion_play_data['Primary_Impact_Type'].value_counts()
sns.countplot(x='Primary_Impact_Type', data=concussion_play_data)
#Create a unique identifier for each game by combining GameKey and PlayID

concussion_play_data['GameKey_PlayID'] = concussion_play_data['GameKey'].map(str) + concussion_play_data['PlayID'].map(str)
concussion_play_data.head()
#Assign True value for all concussions in this data set
concussion_play_data['Concussion'] = 'True'
#Read in remaining play data (including concussion punts)

all_plays = pd.read_csv('../input/play-informationcsv/play_information.csv')
all_plays.info()
#A little offline manipulation for time's sake. 
#Here Territory = 0 for a punt from the punting team's side of the field; Territory = 1 for a punt from the receiving team's side of the field

all_plays.head()
#To merge tables, check that PlayID is unique

all_plays.PlayID.value_counts()
# PlayID is not unique! So we create a unique identifier to merge the tables by combining GameKey and PlayID and removing the lone duplicate
all_plays['GameKey_PlayID'] = all_plays['GameKey'].map(str) + all_plays['PlayID'].map(str)
all_plays['GameKey_PlayID'].value_counts()
#Visualize the only duplicate from our unique identifier

all_plays[all_plays['GameKey_PlayID']=='613849']
#Remove duplicate by adding a and b

all_plays.iloc[697, all_plays.columns.get_loc('GameKey_PlayID')] = '613849a'
all_plays.iloc[6124, all_plays.columns.get_loc('GameKey_PlayID')] = '613849b'
#Assigns True to 'fair_catch' if the 'PlayDescription' contains the phrase 'fair catch'

all_plays['fair_catch'] = all_plays['PlayDescription'].str.contains('fair catch')
#Assigns True to 'Touchback' if the 'PlayDescription' contains the phrase 'Touchback'

all_plays['Touchback'] = all_plays['PlayDescription'].str.contains('Touchback')
#Assigns True to 'out_of_bounds' if the 'PlayDescription' contains the phrase 'out_of_bounds'

all_plays['out_of_bounds'] = all_plays['PlayDescription'].str.contains('out of bounds')
#Assigns True to 'out_of_bounds' if the 'PlayDescription' contains the phrase 'out_of_bounds'

all_plays['downed'] = all_plays['PlayDescription'].str.contains('downed')
#Manipulate the string for 'PlayDescription' to capture the distance of the punt

new = all_plays['PlayDescription'].str.split(pat='punts',n=1, expand=True)
all_plays['split1'] = new[0]
all_plays['split2'] = new[1]
all_plays['distance'] = all_plays['split2'].str.extract('(\d+)')
all_plays.drop(columns=["split1","split2"], inplace=True)
all_plays['distance']=all_plays['distance'].fillna(0)
all_plays['distance'] = all_plays['distance'].map(int)
#Great, let's merge tables to get the full universe of all punt plays from the 16-17 seasons with our new variables
full_data = pd.merge(all_plays, concussion_play_data, on='GameKey_PlayID', how='left')
full_data.info()
full_data[full_data['Concussion'] == "True"]
# 'Concussion' will be our target, so we fill the concussion column with False for NA

full_data['Concussion']=full_data['Concussion'].fillna('False')
full_data.info()
#Because the 'YardLine' field reports numbers on the 0-50-0 scale of a football field, I did a little manipulation to be able to show where the play orinigated on one axis

full_data['starting_yard'] = full_data['YardLine'].str.extract('(\d+)')
full_data['starting_yard'] = full_data['starting_yard'].map(int)
def yardline (row):
    if row['Territory'] == 0:
        return row['starting_yard']
    if row['Territory'] == 1:
        return 100-row['starting_yard']

full_data['100yd'] = full_data.apply (lambda row: yardline (row), axis=1)
full_data.head()
full_data.drop(columns='starting_yard')
full_data.sort_values('100yd', ascending=False)
#It would also be helpful to encode a variable for whether or not the play resulted in a return, because it sure seems like returns are dangerous

def returned (row):
    if row['fair_catch'] == 1:
        return False
    if row['Touchback'] == 1:
        return False
    if row['out_of_bounds'] == 1:
        return False
    if row['downed'] == 1:
        return False
    if row['distance'] == 0:
        return False
    return True

full_data['returned'] = full_data.apply (lambda row: returned (row),axis=1)        
full_data.head()
full_data.info()
from sklearn.tree import DecisionTreeClassifier
target_name = 'Concussion'
feats = ['Territory','fair_catch','Touchback','out_of_bounds','downed']
X = full_data[feats]
y = full_data[target_name]
import graphviz 
from sklearn.tree import export_graphviz

treeclf = DecisionTreeClassifier()

treeclf.fit(X, y)
dot_data = export_graphviz(treeclf, out_file=None, feature_names=feats)

graph = graphviz.Source(dot_data)  
graph
treeclf_light = DecisionTreeClassifier(min_samples_split=100, min_samples_leaf=100, max_depth = 7)

treeclf_light.fit(X, y)
dot_data = export_graphviz(treeclf_light, out_file=None, feature_names=feats)

graph = graphviz.Source(dot_data)  
graph
#What was the deal with that punt of 0 yards? It wasn't a punt at all. The concussion occurred on a fake.

full_data['PlayDescription'].loc[2749]
#Going from the more simplistic classification based on the side of the field to look more closely at where the play originated by yardline

target_name = 'Concussion'
feats = ['100yd','fair_catch','Touchback','out_of_bounds','downed']
X = full_data[feats]
y = full_data[target_name]

treeclf_light = DecisionTreeClassifier(min_samples_split=100, min_samples_leaf=100, max_depth = 4)

treeclf_light.fit(X, y)
dot_data = export_graphviz(treeclf_light, out_file=None, feature_names=feats)

graph = graphviz.Source(dot_data)  
graph
#Adding a static variable for graphing purposes
full_data['punt'] = "Punt"
g = sns.catplot(x='100yd', y='Concussion', data=full_data,orient='h', height= 7, aspect = 2)
#One bar
g = sns.catplot(x='100yd', y='punt', data=full_data, hue='returned',orient='h', height= 7, aspect = 2, palette = 'Greys', alpha=1)
#Two bars (just for different visualization types)

g = sns.catplot(x='100yd', y='returned', data=full_data,orient='h',height= 7, aspect = 2, palette='Blues')
g = sns.catplot(x='100yd', y='punt', data=full_data,orient='h', hue='fair_catch',height= 7, aspect = 2, palette='Blues', alpha=.8)
#Two bars (just for different visualization types)

g = sns.catplot(x='100yd', y='fair_catch', data=full_data,orient='h',height= 7, aspect = 2, palette='Blues')
full_data['downed'].value_counts()
full_data['fair_catch'].value_counts()
full_data[full_data['100yd']<31].fair_catch.value_counts()
full_data[full_data['100yd']<31].downed.value_counts()
full_data[(full_data['100yd']<31) & (full_data['fair_catch']==True)]
full_data[full_data['100yd']<31].distance.mean()
full_data[full_data['100yd']<31].returned.value_counts()
full_data.returned.value_counts()
full_data.groupby('returned')['100yd'].mean()
full_data.groupby('returned')['100yd'].median()
full_data.groupby('fair_catch')['100yd'].mean()
full_data.groupby('fair_catch')['100yd'].median()
g = sns.catplot(x='100yd', y='punt', data=full_data, hue='out_of_bounds',orient='h', height= 7, aspect = 2,palette = 'Greens')
g = sns.catplot(x='100yd', y='out_of_bounds', data=full_data,orient='h',height= 7, aspect = 2, palette='Greens')
g = sns.catplot(x='100yd', y='punt', data=full_data, hue='downed',orient='h', height= 7, aspect = 2,palette = 'Reds')
g = sns.catplot(x='100yd', y='downed', data=full_data,orient='h',height= 7, aspect = 2, palette='Reds')
g = sns.catplot(x='100yd', y='punt', data=full_data, hue='Touchback',orient='h', height= 7, aspect = 2,palette = 'Purples')
g = sns.catplot(x='100yd', y='Touchback', data=full_data,orient='h',height= 7, aspect = 2, palette='Purples')
