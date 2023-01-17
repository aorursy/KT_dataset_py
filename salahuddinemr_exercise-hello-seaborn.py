import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex1 import *

print("Setup Complete")
# Fill in the line below

one = 1



# Check your answer

step_1.check()
step_1.hint()

step_1.solution()
# Path of the file to read

fifa_filepath = "../input/fifa.csv"



# Read the file into a variable fifa_data

fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)



# Check your answer

step_2.check()
#Uncomment the line below to receive a hint

step_2.hint()

# Uncomment the line below to see the solution

#step_2.solution()
# Set the width and height of the figure

plt.figure(figsize=(16,6))



# Line chart showing how FIFA rankings evolved over time

sns.lineplot(data=fifa_data)



# Check your answer

step_3.a.check()
#step_3.b.hint()
#step_3.b.solution()