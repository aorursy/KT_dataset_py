# set up code checking

import os

if not os.path.exists("../input/fifa.csv"):

    os.symlink("../input/data-for-datavis/fifa.csv", "../input/fifa.csv")  

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex1 import *

print("Setup Complete")



import pandas as pd

# for plotting timeseries

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
one = 1



# check your answer

step_1.check()
# step_1.hint()

# step_1.solution()
# path of the file to read

fifa_filepath = "../input/fifa.csv"



# read the file into a variable fifa_data

fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)



# check your answer

step_2.check()
# Uncomment the line below to receive a hint

# step_2.hint()

# Uncomment the line below to see the solution

# step_2.solution()
# set the width and height of the figure

plt.figure(figsize=(16,6))



# line chart showing how FIFA rankings evolved over time

sns.lineplot(data=fifa_data)



# Check your answer

step_3.a.check()
# step_3.b.hint()
# Check your answer (Run this code cell to receive credit!)

step_3.b.solution()