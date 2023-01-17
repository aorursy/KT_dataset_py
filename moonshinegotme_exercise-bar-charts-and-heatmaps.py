import pandas as pd

import numpy as np

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

ign_data = pd.read_csv("../input/ign_scores.csv", index_col = "Platform")



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the data

ign_data # Your code here
# Fill in the line below: What is the highest average score received by PC games,

# for any platform?

high_score = ign_data.iloc[8, :].values.tolist()

high_score = max(high_score)

print(high_score)



def find_max_row(x):

    max_row_x = ign_data.iloc[x, :].values.tolist()

    return max(max_row_x)

    

max_row_8 = find_max_row(8)

print(max_row_8)



max_score = ign_data.iloc[8, :].values.max()

print(max_score)



# returns max value in the whole dataframe after converting it to an array

""" slew_rate_max.values converts the dataframe to a np.ndarray then 

using numpy.ndarray.max which as an argument axis that the default is None, 

this gives the max of the entire ndarray.



You can get the same by calling method .max() twice on the df

fd.max().max() returns a series with the max of each column, 

then taking the max again of that series 

will give you the max of the entire dataframe.



This gives me the entire PC row as series

ign_data.loc['PC']



This takes row 8 (PC) from the converted array, the array is 

a set of lists with each list a row

ign_data.values[8]



This gives me the values of the column 'PC'

ign_data.loc['PC', :]



Same as this but the below uses index

ign_data.iloc[8, :]



Interesting one

from operator import itemgetter

high_score = max(high_score, key=itemgetter(1))[0]



This is pandas syntax for finding max in a coloumn

high_score = ign_data['Action'].max()

"""











# Fill in the line below: On the Playstation Vita platform, which genre has the 

# lowest average score? Please provide the name of the column, and put your answer 

# in single quotes (e.g., 'Action', 'Adventure', 'Fighting', etc.)

worst_genre = ign_data.loc['PlayStation Vita', :].idxmin()

print(worst_genre)





# Check your answers

# step_2.check()
# Lines below will give you a hint or solution code

# step_2.hint()

step_2.solution()
# Bar chart showing average score for racing games by platform

plt.figure(figsize =(14, 8)) # Your code here



sns.barplot(x = ign_data['Racing'], y = ign_data.index)





# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint() 
#step_3.b.solution()
# Heatmap showing average game score by platform and genre

plt.figure(figsize = (14, 10))

sns.heatmap(data = ign_data, annot = True, cmap = "Reds") # Your code here



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
#step_4.b.solution()