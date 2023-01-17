!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@master
import sys

sys.path.append('/kaggle/working')
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_easy.ex1 import *

print("Setup Complete")
# Fill in the line below

one = 1



# Check your answer

step_1.check()
step_1.hint()

step_1.solution()
# Path of the file to read

fifa_filepath = "../input/fifa.csv"



# Fill in the line below to read the file into a variable fifa_data

fifa_data = ____



# Check your answer

step_2.check()
# Uncomment the line below to receive a hint

#step_2.hint()

# Uncomment the line below to see the solution

#step_2.solution()
# Print the last five rows of the data 

____ # Your code here
# Fill in the line below: What was Brazil's ranking (Code: BRA) on December 23, 1993?

brazil_rank = ____ 



# Check your answer

step_3.check()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution()
# Set the width and height of the figure

plt.figure(figsize=(16,6))



# Fill in the line below: Line chart showing how FIFA rankings evolved over time

____ # Your code here



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
#step_4.b.solution()