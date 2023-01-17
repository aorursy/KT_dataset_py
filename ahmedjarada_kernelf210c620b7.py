import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex3 import *

print("Setup Complete")
# Path of the file to read

ign_filepath = "../input/ign_scores.csv"



# Fill in the line below to read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath, index_col="Platform")



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the data

ign_data # Your code here
titles=list(ign_data)

print(titles)

pc_score=[]

for title in titles:

    pc_score.append(ign_data[title]['PC'])

print(pc_score)

print(max(pc_score))


playstation_vita_score=[]

for title in titles:

    playstation_vita_score.append(ign_data[title]['PlayStation Vita'])

print(playstation_vita_score)

print(min(playstation_vita_score))

print(playstation_vita_score.index(min(playstation_vita_score)))



print(titles[playstation_vita_score.index(min(playstation_vita_score))])
# Fill in the line below: What is the highest average score received by PC games,

# for any platform

high_score = max(pc_score)



# Fill in the line below: On the Playstation Vita platform, which genre has the 

# lowest average score? Please provide the name of the column, and put your answer 

# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)



worst_genre = worst_genre = titles[playstation_vita_score.index(min(playstation_vita_score))]



print(worst_genre)

# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Bar chart showing average score for racing games by platform

# Your code here

sns.barplot(x=ign_data.index,  y=ign_data['Racing']) 

# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

# step_3.a.hint()

#step_3.a.solution_plot()
step_3.b.hint()

# Your code 

plt.figure(figsize=(40,20))

sns.barplot(x=ign_data.index,  y=ign_data['Racing']) 

print('Wii is not the highest rating','then the best alternative is Xbox One')

step_3.b.solution()
# Heatmap showing average game score by platform and genre

# Your code here

plt.figure(figsize=(40,30))

plt.title("Average game score by platform")



sns.heatmap(data=ign_data ,annot=True) 





# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
#step_4.b.solution()