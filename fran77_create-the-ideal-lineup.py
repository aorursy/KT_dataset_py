# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # graph

import matplotlib.pyplot as plt

from matplotlib.patches import Arc, Rectangle, ConnectionPatch

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')

data.drop(data.columns[[0]], axis=1, inplace=True)
data.head()
len(data)
# Overall / Age
data.Age.mean()
sns.set(rc={'figure.figsize':(9,7)})

sns.distplot(data.Age)

plt.axvline(data.Age.mean())
sns.lineplot(x="Age", y="Overall", data=data[data.Age <40])
# It seems a player is at his pike for at 31, let's verify this

overall_age = round(data[data.Age<40].groupby('Age')['Overall'].mean(),2)

overall_age = overall_age.reset_index()
overall_age[overall_age.Overall == max(overall_age.Overall)]
# Overall / Value
data['Value'].head()
data.Value  =data.Value.str[1:]
def value_to_float(x):

    if 'K' in x:

        return float(x.replace('K', '')) * 1000

    if 'M' in x:

        return float(x.replace('M', '')) * 1000000
data.Value = data.Value.apply(value_to_float)
sns.lineplot(x="Overall", y="Value", data=data)
# It shows an exponential curve as the overall is rising

# Until a 70 Overall, the curve doesn't rise a lot (less than 1M)

# The max is achieved at 92 Overall with a mean value of 118.5M (Neymar) 
overall_value = round(data.groupby('Overall')['Value'].mean(),2)

overall_value = overall_value.reset_index()

overall_value[overall_value.Value == max(overall_value.Value)]
# Overall / Wage
data.Wage  =data.Wage.str[1:]

data.Wage = data.Wage.apply(value_to_float)
sns.lineplot(x="Overall", y="Wage", data=data)
# Again the curve is exponential
# Preferred foot repartion 
sns.countplot(data['Preferred Foot'])
round(data.groupby('Preferred Foot').count()['ID'] / len(data),2)
# Country repartition and the best country
sns.countplot(data.Nationality, order=data.Nationality.value_counts().iloc[:10].index)
nationalities = data.groupby(['Nationality', 'Overall'])['ID'].count()
nationalities = nationalities.reset_index()
nationalities.columns = ['Nationality', 'Overall', 'Count']
nationalities.Overall = nationalities.Overall * nationalities.Count
nationalities = nationalities.groupby('Nationality').sum()
nationalities['Mean'] = nationalities['Overall'] / nationalities['Count']
nationalities = nationalities.reset_index()
nationalities = nationalities[nationalities.Nationality.isin(data.Nationality.value_counts().iloc[:10].index)]

nationalities = nationalities.sort_values(['Mean'], ascending=False)
g = sns.barplot(x = nationalities.Nationality, y = nationalities.Mean)

g.set_xticklabels(g.get_xticklabels(),rotation=90)
# Potential for teenagers
potential = data.sort_values(['Potential'], ascending = False)

potential = potential[potential.Age <=20][:10]
g = sns.barplot(x='Name', y='Potential', data=potential)

g.set_xticklabels(g.get_xticklabels(),rotation=90)
potential[['Name', 'Nationality', 'Club','Potential']]
# Best teams
teams_count = data.groupby('Club')['Name'].count()

teams_count = teams_count.reset_index()
teams_overall = data.groupby('Club')['Overall'].sum()

teams_overall = teams_overall.reset_index()
teams = pd.merge(teams_count, teams_overall, on='Club')
teams['Mean'] = teams['Overall'] / teams['Name']
best_teams = teams.sort_values(['Mean'], ascending=False)[:15]
g = sns.barplot(x='Club', y='Mean', data=best_teams)

g.set_xticklabels(g.get_xticklabels(),rotation=90)
# And now, only with the 11 best players on each team
team_11 = data.groupby('Club')['Overall'].apply(lambda grp: grp.nlargest(11).sum())
team_11 = team_11.reset_index()
team_11['Overall'] /= 11
team_11 = team_11.sort_values(['Overall'], ascending=False)[:15]
g = sns.barplot(x='Club', y='Overall', data=team_11)

g.set_xticklabels(g.get_xticklabels(),rotation=90)
# Best players by post
data.Position.unique()
positions = data[data.Position.notnull()]
best_positions = positions.loc[positions.groupby('Position')['Overall'].idxmax()]
best_positions = best_positions[['Name', 'Nationality', 'Club', 'Position', 'Overall']]

best_positions
# Best lineup
# GK : goalkeeper

# CB : Center Back

# LCB : Left Center Back

# RCB : Right Center Back

# LB, RB : Fullback

# LWB, RWB : Wing back

# CDM : Center Defensive Mildfield

# LDM : Left center midfield

# RDM : Right center midfield

# CM : Center Mildield

# LCM : Right Center Mildield

# RCM : Left Center Mildield

# CAM : Center Attacking Mildfield

# LAM : Left Attacking Mildfield   

# RAM : Right Attacking Mildfield   

# LM : Left Mildfield

# LW : Left Winger

# LF : Left Forward

# RM : Right Midfield

# RW : Right Winger

# RF : Right Forward

# CF : Center Forward

# LS : Left Striker

# RS : Right Striker

# ST : Striker
# We will use the 4-4-2 schema which is the most classic
data.Position.unique()
data['Post'] = data['Position'].map({'GK':'GK', 'CB':'CB', 'LB':'LB', 'RB':'RB', 'RLB':'RB', 'RCB':'RB', 'LWB': 'LB', 'RWB':'RB', 'CDM':'CM', 

                                     'RDM':'CM', 'LDM':'CM', 'CM':'CM', 'LCM':'CM', 'RCM':'CM', 'CAM':'CM', 'LAM':'CM', 'RAM':'CM', 'LW':'LM', 

                                     'LM':'LM', 'RM':'RM', 'RW':'RM', 'RF':'ST', 'LF':'ST', 'CF':'ST', 'LS':'ST', 'RS':'ST', 'ST':'ST'})
data.Post.unique()
post = data[data.Post.notnull()]
post = post[['Name', 'Nationality', 'Club', 'Post', 'Overall', 'Jersey Number']]
post.head()
best_lineup = post.loc[post.groupby('Post')['Overall'].head(2).index]
best_posts = best_lineup.groupby('Post').first().reset_index()
best_lineup = best_lineup[(best_lineup.Post.isin(['CB', 'CM', 'ST'])) | (best_lineup.Name.isin(best_posts.Name.tolist()))]
best_lineup = best_lineup.sort_values(['Post'])
best_lineup
def draw_pitch(ax):

    # Create pitch



    # Pitch Outline & Centre Line

    rect1 = Rectangle((0,0), 90, 120, color='green', edgecolor= "white")

    plt.plot([0,90],[60,60], color="white")



    # Back Penalty Area

    plt.plot([65,25], [16,16],color="white")

    plt.plot([65,65], [0,16],color="white")

    plt.plot([25,25], [0,16],color="white")



    # Top Penalty Area

    plt.plot([65,65], [120,104],color="white")

    plt.plot([65,25], [104,104],color="white")

    plt.plot([25,25], [120, 104],color="white")



    # Left 6-yard Box

    plt.plot([54,54], [0,4.9],color="white")

    plt.plot([54,36], [4.9,4.9],color="white")

    plt.plot([36,36], [0,4.9],color="white")



    # Right 6-yard Box

    plt.plot([54,54], [120,115.1],color="white")

    plt.plot([54,36], [115.1,115.1],color="white")

    plt.plot([36,36], [120,115.1],color="white")



    # Prepare Circles

    centreCircle = plt.Circle((45,60),10, color="white",fill=False)

    centreSpot = plt.Circle((45,60),0.71,color="white")

    leftPenSpot = plt.Circle((45,9.7),0.71,color="white")

    rightPenSpot = plt.Circle((45,110),0.71,color="white")

    

    # Create players position

    goal = plt.Circle((45,2),1.2, color="red")

    cb1 = plt.Circle((35,25),1.2, color="red")

    cb2 = plt.Circle((55,25),1.2, color="red")

    lb = plt.Circle((15,35),1.2, color="red")

    rb = plt.Circle((75,35),1.2, color="red")

    cm1 = plt.Circle((35,55),1.2, color="red")

    cm2 = plt.Circle((55,55),1.2, color="red")

    lm = plt.Circle((15,70),1.2, color="red")

    rm = plt.Circle((75,70),1.2, color="red")

    st1 = plt.Circle((35,95),1.2, color="red")

    st2 = plt.Circle((55,95),1.2, color="red")



    # Draw circles

    ax.add_patch(rect1)

    ax.add_patch(centreCircle)

    ax.add_patch(centreSpot)

    ax.add_patch(leftPenSpot)

    ax.add_patch(rightPenSpot)

    

    # Draw players

    ax.add_patch(goal)

    ax.add_patch(cb1)

    ax.add_patch(cb2)

    ax.add_patch(lb)

    ax.add_patch(rb)

    ax.add_patch(cm1)

    ax.add_patch(cm2)

    ax.add_patch(lm)

    ax.add_patch(rm)

    ax.add_patch(st1)

    ax.add_patch(st2)

    

    # Add player Name

    ax.annotate(best_lineup.iloc[0]['Name'], xy = (31,27), color='white')

    ax.annotate(best_lineup.iloc[1]['Name'], xy = (51,27), color='white')

    ax.annotate(best_lineup.iloc[2]['Name'], xy = (28,57), color='white')

    ax.annotate(best_lineup.iloc[3]['Name'], xy = (51,57), color='white')

    ax.annotate(best_lineup.iloc[4]['Name'], xy = (41,6), color='white')

    ax.annotate(best_lineup.iloc[5]['Name'], xy = (11,37), color='white')

    ax.annotate(best_lineup.iloc[6]['Name'], xy = (11,72), color='white')

    ax.annotate(best_lineup.iloc[7]['Name'], xy = (68,37), color='white')

    ax.annotate(best_lineup.iloc[8]['Name'], xy = (70,72), color='white')

    ax.annotate(best_lineup.iloc[9]['Name'], xy = (31,97), color='white')

    ax.annotate(best_lineup.iloc[10]['Name'], xy = (46,97), color='white')
fig=plt.figure()

fig.set_size_inches(10, 12)

plt.grid(False)

ax=fig.add_subplot(1,1,1)

draw_pitch(ax)

plt.show()
# Jersey Number distribution (all and the best players)
jersey = data[data['Jersey Number'].notnull()]
sns.distplot(jersey['Jersey Number'])
jersey.groupby('Jersey Number').count()['ID'].sort_values(ascending=False).head(10)
# Let's see the number by post
post.groupby('Post').apply(lambda x: x['Jersey Number'][x.Overall.idxmax()])
data.columns
labels = np.array(['Finishing', 'BallControl', 'SprintSpeed', 'Agility', 'Dribbling', 'Vision'])
messi = data.loc[data[data.Name == "L. Messi"].index[0],labels].values

ronaldo = data.loc[data[data.Name == "Cristiano Ronaldo"].index[0],labels].values



angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot

messi = np.concatenate((messi,[messi[0]]))

ronaldo = np.concatenate((ronaldo,[ronaldo[0]]))



angles = np.concatenate((angles,[angles[0]]))
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(111, polar=True)

ax1.plot(angles, messi, 'o-', linewidth=2, label = 'Messi')

ax1.fill(angles, messi, alpha=0.25)

ax1.set_thetagrids(angles * 180/np.pi, labels)

ax1.grid(True)



ax2 = fig.add_subplot(111, polar=True)

ax2.plot(angles, ronaldo, 'o-', linewidth=2, label = 'Ronaldo')

ax2.fill(angles, ronaldo, alpha=0.25)

ax2.set_thetagrids(angles * 180/np.pi, labels)

ax2.grid(True)



plt.legend(bbox_to_anchor=(1,1))