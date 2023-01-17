import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

import os

if not os.path.exists("../input/candy.csv"):

    os.symlink("../input/data-for-datavis/candy.csv", "../input/candy.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex4 import *

print("Setup Complete")
# Path of the file to read

candy_filepath = "../input/candy.csv"



# Fill in the line below to read the file into a variable candy_data

candy_data = pd.read_csv(candy_filepath, index_col="id")



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the first five rows of the data

candy_data.head() # Your code here
# Fill in the line below: Which candy was more popular with survey respondents:

# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)

more_popular = '3 Musketeers'



# Fill in the line below: Which candy has higher sugar content: 'Air Heads'

# or 'Baby Ruth'? (Please enclose your answer in single quotes.)

more_sugar = 'Air Heads'



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'

sns.scatterplot(data=candy_data, x="sugarpercent", y="winpercent") # Your code here



# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint()

# No strong indication of correlation between sugar content ["sugarpercent"] and popularity ["winpercent"] of candies.

# Thus, no strong relationship between the two variables amid the weak correlation.
# Check your answer (Run this code cell to receive credit!)

step_3.b.solution()
# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'

sns.regplot(data=candy_data, x="sugarpercent", y="winpercent") # Your code here



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()

# There is a slight positive correlation between the two variables. 

# People have a slight tendency to prefer candies with higher sugar content.
# Check your answer (Run this code cell to receive credit!)

step_4.b.solution()
# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'

sns.scatterplot(data=candy_data, x="pricepercent", y="winpercent", hue="chocolate") # Your code here



# Check your answer

step_5.check()
# Lines below will give you a hint or solution code

#step_5.hint()

#step_5.solution_plot()
# Color-coded scatter plot w/ regression lines

sns.lmplot(data=candy_data, x="pricepercent", y="winpercent", hue="chocolate") # Your code here



# Check your answer

step_6.a.check()
# Lines below will give you a hint or solution code

#step_6.a.hint()

#step_6.a.solution_plot()
#step_6.b.hint()

# Prices ["pricepercent"] for chocolate candies ["candies] have a slight positive correlation with popularity ["winpercent"].

# Thus, premium chocolate candies tend to be more popular amongst candy consumers.

# Meanwhile, prices for non-chocolate candies have a slight negative correlation with popularity, indicating that

# non-chocolate candies tend to command more popularity when they are cheaper.
# Check your answer (Run this code cell to receive credit!)

#step_6.b.solution()
# Scatter plot showing the relationship between 'chocolate' and 'winpercent'

sns.swarmplot(data=candy_data, x="chocolate", y="winpercent")

# Your code here



# Check your answer

step_7.a.check()
# Lines below will give you a hint or solution code

#step_7.a.hint()

#step_7.a.solution_plot()
#step_7.b.hint()

# Step 7 as the chart conveys overall popularity of chocolate and non-chocolate candies and its simple to interpret.
# Check your answer (Run this code cell to receive credit!)

step_7.b.solution()