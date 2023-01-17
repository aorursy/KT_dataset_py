import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex4 import *

print("Setup Complete")
# Path of the file to read

candy_filepath = "../input/candy.csv"



# Fill in the line below to read the file into a variable candy_data

candy_data = pd.read_csv(candy_filepath,index_col='id')



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the first five rows of the data

____ # Your code here

candy_data.head(5)
# Fill in the line below: Which candy was more popular with survey respondents:

# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)



temp = candy_data.loc[(candy_data['competitorname']=='3 Musketeers') | (candy_data['competitorname']=='Almond Joy')]

more_popular = temp.loc[temp['winpercent'] == temp['winpercent'].max(),'competitorname'].values[0]



# Fill in the line below: Which candy has higher sugar content: 'Air Heads'

# or 'Baby Ruth'? (Please enclose your answer in single quotes.)



temp = candy_data.loc[(candy_data['competitorname']=='Air Heads') | (candy_data['competitorname']=='Baby Ruth')]

more_sugar = temp.loc[temp['sugarpercent'] == temp['sugarpercent'].max(),'competitorname'].values[0]



# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'

____ # Your code here



plt.figure(figsize=(6,6))

sns.scatterplot(x=candy_data['sugarpercent'],y=candy_data['winpercent'])

# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

# step_3.a.solution_plot()
#step_3.b.hint()
#step_3.b.solution()
# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'

____ # Your code here

plt.figure(figsize=(6,6))

sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

# step_4.a.hint()

#step_4.a.solution_plot()
# step_4.b.hint()
step_4.b.solution()
# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'

____ # Your code here

sns.scatterplot(x=candy_data['pricepercent'],y=candy_data['winpercent'],hue=candy_data['chocolate'])



# Check your answer

step_5.check()
# Lines below will give you a hint or solution code

# step_5.hint()

#step_5.solution_plot()
# Color-coded scatter plot w/ regression lines

____ # Your code here

sns.lmplot(x='pricepercent',y='winpercent',data=candy_data,hue='chocolate')



# Check your answer

step_6.a.check()
# Lines below will give you a hint or solution code

# step_6.a.hint()

# step_6.a.solution_plot()
#step_6.b.hint()
#step_6.b.solution()
# Scatter plot showing the relationship between 'chocolate' and 'winpercent'

____ # Your code here

sns.swarmplot(x='chocolate',y='winpercent',data=candy_data)

# Check your answer

step_7.a.check()
# Lines below will give you a hint or solution code

#step_7.a.hint()

#step_7.a.solution_plot()
# step_7.b.hint()
step_7.b.solution()