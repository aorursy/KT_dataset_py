import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

import os

if not os.path.exists("../input/ign_scores.csv"):

    os.symlink("../input/data-for-datavis/ign_scores.csv", "../input/ign_scores.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex3 import *

print("Setup Complete")
# Path of the file to read

ign_filepath = "../input/ign_scores.csv"



# Fill in the line below to read the file into a variable ign_data

ign_data = pd.read_csv(ign_filepath,index_col='Platform')



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the data

# Your code here

print(ign_data)
# Fill in the line below: What is the highest average score received by PC games,

# for any platform?

high_score = 7.759930



# Fill in the line below: On the Playstation Vita platform, which genre has the 

# lowest average score? Please provide the name of the column, and put your answer 

# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)

worst_genre = 'Simulation'



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Bar chart showing average score for racing games by platform

# Your code here

plt.figure(figsize=(10,6))

plt.title('racing games by platform')

sns.barplot(x=ign_data.index,y=ign_data['Racing'])

plt.xlabel('platforms')

plt.ylabel('averages')

# Check your answer

step_3.a.check()

plt.show()

ign_data.index
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
plt.figure(figsize=(10,6))

plt.title('racing games by platform')

sns.barplot(x=ign_data.index,y=ign_data['Racing'],order=['Dreamcast', 'Game Boy Advance', 'Game Boy Color', 'GameCube',

       'Nintendo 3DS', 'Nintendo 64', 'Nintendo DS', 'Nintendo DSi', 'PC',

       'PlayStation', 'PlayStation 2', 'PlayStation 3', 'PlayStation 4',

       'PlayStation Portable', 'PlayStation Vita', 'Wii', 'Wireless', 'Xbox',

       'Xbox 360', 'Xbox One', 'iPhone'])

plt.ylabel('averages')

plt.show()

print('Xbox One')
#step_3.b.hint()
step_3.b.solution()
# Heatmap showing average game score by platform and genre

# Your code here

plt.figure(figsize=(10,6))

plt.title('heatmap')

sns.heatmap(data = ign_data, annot = True)

plt.xlabel('Genre')

plt.ylabel('platform')



# Check your answer

step_4.a.check()

plt.show()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
print('highest average rating : 9.2 PlayStation4 and Simulation')

print('lowest average rating : 4.5 GameBoyColor and Fighting,Shooter ')
#step_4.b.hint()
#step_4.b.solution()