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

candy_data = pd.read_csv(candy_filepath,index_col="id")



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the first five rows of the data

print(candy_data.head())# Your code here

# Fill in the line below: Which candy was more popular with survey respondents:

# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)

more_popular = candy_data.loc[candy_data['winpercent']==max(candy_data.loc[1]['winpercent'], candy_data.loc[3]['winpercent'])].competitorname

print(more_popular)

# Fill in the line below: Which candy has higher sugar content: 'Air Heads'

# or 'Baby Ruth'? (Please enclose your answer in single quotes.)

#& (ca'Air Heads' | candy_data.competitorname=='Baby Ruth') 





# Check your answers





# Check your answers

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'

sns.scatterplot(x='sugarpercent',y='winpercent',data=candy_data) # Your code here



# Check your answer

step_3.a.check()
# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'

sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])# Your code here



# Check your answer

step_4.a.check()
sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'

sns.scatterplot(x=candy_data['pricepercent'], y=candy_data['winpercent'],hue=candy_data['chocolate'])# Your code here



# Check your answer

step_5.check()
# Color-coded scatter plot w/ regression lines

sns.lmplot(x='pricepercent', y='winpercent',hue='chocolate',data=candy_data)# Your code here



# Check your answer

step_6.a.check()
# Scatter plot showing the relationship between 'chocolate' and 'winpercent'

sns.swarmplot(x=candy_data['chocolate'],y=candy_data['winpercent'])# Your code here



# Check your answer

step_7.a.check()