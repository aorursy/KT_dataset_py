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

plt.figure(figsize=(16,5))

sns.scatterplot(x=candy_data['sugarpercent'],y=candy_data['winpercent']) # Your code here

plt.title("Sugar Content Vs Popularity")

# Check your answer

step_3.a.check()



#There is kind of like an increasing trend wherein more the sugar more popular it is.The most popular candy has

#around 80% sugar in it.Something like a small mango/strawberry candy 
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
plt.figure(figsize=(16,5))

sns.regplot(x=candy_data['sugarpercent'],y=candy_data['winpercent']) # Your code here

plt.title("Sugar Content Vs Popularity with regression")

plt.show()



'''There is some ammount of coorelation between the fact that people prefer candies with more sugar in them.

SInce our regression line has positive slope(Can compute slope by hand to verify this between 2 points)'''
#step_3.b.hint()
#step_3.b.solution()
# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'

sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent']) # Your code here



# Check your answer

step_4.a.check()
sns.regplot(x=candy_data['sugarpercent'], y=candy_data['winpercent'])

# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
#step_4.b.solution()
# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'

plt.figure(figsize=(16,5))

sns.scatterplot(x=candy_data['pricepercent'],y=candy_data['winpercent'],hue=candy_data['chocolate']) # Your code here

plt.title("Sugar Content Vs Popularity Color Code with Containing Chocolate(Yes/No)")





# Check your answer

step_5.check()

plt.show()
# Lines below will give you a hint or solution code

#step_5.hint()

#step_5.solution_plot()
# Color-coded scatter plot w/ regression lines

plt.figure(figsize=(16,5))

sns.lmplot(x="pricepercent", y="winpercent", hue="chocolate", data=candy_data)

plt.title("Sugar Content Vs Popularity Color Code with Containing Chocolate(Yes/No) and Regression")



# Check your answer

step_6.a.check()

plt.show()
# Lines below will give you a hint or solution code

#step_6.a.hint()

#step_6.a.solution_plot()
#step_6.b.hint()



'''At lower Price Points,people prefer candies with no chocolate content and at higher price points

people prefer candies with chocolate in them'''
#step_6.b.solution()
# Scatter plot showing the relationship between 'chocolate' and 'winpercent'

sns.swarmplot(x=candy_data["chocolate"], y=candy_data["winpercent"]) # Your code here



# Check your answer

step_7.a.check()
# Lines below will give you a hint or solution code

#step_7.a.hint()

#step_7.a.solution_plot()
#step_7.b.hint()



'''Plot no 7 Would be more appropriate in this case as the difference between non chocolate and chocolate 

candies is very clear'''
#step_7.b.solution()