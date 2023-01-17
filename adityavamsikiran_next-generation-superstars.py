#Loading Data

import pandas as pd

import numpy as np

fifa_data = pd.read_csv("../input/FullData.csv")


bin_values = np.arange(start=5, stop=55, step=5)

fifa_data['Age'].plot(kind='hist', bins=bin_values, figsize=[12, 6], alpha=.4, legend=True)  # alpha for transparency

age_25 = fifa_data['Age'] <=25

fifa_data_25 = fifa_data[age_25]

print("There are %d players who are equal to or under 25 " %fifa_data_25.shape[0])  # Number of players who are under 25
x_cols = ['Name','Weak_foot', 'Skill_Moves','Ball_Control', 'Dribbling', 'Marking', 'Sliding_Tackle','Standing_Tackle', 'Aggression', 'Reactions', 'Attacking_Position',

       'Interceptions', 'Vision', 'Composure', 'Crossing', 'Short_Pass','Long_Pass', 'Acceleration', 'Speed', 'Stamina', 'Strength', 'Balance',

       'Agility', 'Jumping', 'Heading', 'Shot_Power', 'Finishing','Long_Shots', 'Curve', 'Freekick_Accuracy', 'Penalties', 'Volleys']



from sklearn.metrics.pairwise import euclidean_distances

X= fifa_data_25[x_cols]

print(fifa_data[['Name','Club','Nationality']][fifa_data['Name'].str.contains("Messi")])

X2 = pd.DataFrame(fifa_data.loc[1,x_cols] .values.reshape(1,32)) 

# Substitute fifa_data.loc[row#,x_cols] 
top_10 = euclidean_distances(X.iloc[:,1:], X2.iloc[:,1:])



top_10_df = pd.DataFrame(top_10)

top_10_df = top_10_df.sort_values([0])



print("Top 10 players who have similar attributes as Messi are below")

X.iloc[top_10_df.index.values[1:11],0]