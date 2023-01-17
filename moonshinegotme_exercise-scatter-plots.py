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

candy_data = pd.read_csv(candy_filepath, index_col = "id")



# Run the line below with no changes to check that you've loaded the data correctly

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

#step_1.solution()
# Print the first five rows of the data

candy_data[0:10] # Your code here

# Fill in the line below: Which candy was more popular with survey respondents:

# '3 Musketeers' or 'Almond Joy'?  (Please enclose your answer in single quotes.)



# Below gives target coloumn value (competitorname) based on the max of column winpercent

# df.loc[df['B'] == 3, 'A']

"""    A  B

    0  p1  1

    1  p1  2

    2  p3  3

    3  p2  4

""" # df.loc[df['B'] == 3, 'A'] gives "2 p3"

more_popular = candy_data.loc[candy_data['winpercent'] == candy_data['winpercent'].max(), 'competitorname']

print(more_popular)



# Adding item() method gives just the value

more_popular = candy_data.loc[candy_data['winpercent'] == candy_data['winpercent'].max(), 'competitorname'].item()

print(more_popular)



# Now I want a function that finds the max winpercent value among two arbitrary rows

"""The code compares the winpercent value for input rows, finds the max, then uses the 

max value to find the value in competitorname corresponding to the winpercent max value"""

def find_more_popular(row1, row2):

    more_popular_row1 = candy_data.loc[candy_data['competitorname'] == row1, 'winpercent'].item()

    more_popular_row2 = candy_data.loc[candy_data['competitorname'] == row2, 'winpercent'].item()

    more_popular_row1_row2 = max(more_popular_row1, more_popular_row2)

    answer = candy_data.loc[candy_data['winpercent'] == more_popular_row1_row2, 'competitorname'].item()

    return answer



more_popular_3M_AJ = find_more_popular('3 Musketeers', 'Almond Joy')

print(more_popular_3M_AJ)





# Now I want a function that finds the max winpercent value among two arbitrary rows

"""The code compares the sugarpercent value for input rows, finds the max, then uses the 

max value to find the value in competitorname corresponding to the sugarpercent max value"""

def find_more_popular_any_col(row1, row2, column):

    more_popular_row1 = candy_data.loc[candy_data['competitorname'] == row1, column]

    more_popular_row2 = candy_data.loc[candy_data['competitorname'] == row2, column]

    max_row1 = more_popular_row1.max()

    max_row2 = more_popular_row2.max()

    maxmax = max(max_row1, max_row2)

    matches = candy_data.loc[candy_data[column] == maxmax, 'competitorname'].tolist() # returns all the values that match maxmax

    if row1 and row2 in matches:

        print("These competitors have the same value for " + column)

        return row1, row2

    elif row1 in matches:

        return row1

    elif row2 in matches:

        return row2

    else:

        print("Error")



more_popular_AH_BR_sugar = find_more_popular_any_col('Air Heads', 'Baby Ruth', 'sugarpercent')

print(more_popular_AH_BR_sugar)



# This should return both competitors as they have the same sugarpercent value

more_popular_same_sugar_example = find_more_popular_any_col('Caramel Apple Pops', 'Charleston Chew', 'sugarpercent')

print(more_popular_same_sugar_example)





"""

***Ignore this stuff, various trial and error***



str.contains('USA')



df_1[:5783][(df_1['Location'].str.contains('USA')) & (df_1['Date at Murder'] >= 20) & (df_1['Date at Murder'] <= 29)][['Characteristics', 'Date at Murder', 'Location', 'Name']][:5783]



You dont need the first [:5783] since you already limit it to that many rows at the end."""



"""

def find_more_popular(row1, row2):

    more_popular_row1 = candy_data.iloc[candy_data[row1], 12].max()

    more_popular_row2 = candy_data.iloc[candy_data[row2], 12].max()

    return max(more_popular_row1, more_popular_row2)

# this gives me the max value but not corresponding to the competitorname coloumn



more_popular_3M_AJ = find_more_popular(1, 3)

print(more_popular_3M_AJ) 



    more_popular_row1 = candy_data.loc[candy_data[row1], candy_data[12] == candy_data[12].max()]

    more_popular_row2 = candy_data.iloc[candy_data[row2], candy_data[12] == candy_data[12].max()]

    return max(more_popular_row1, more_popular_row2)

    

    more_popular_row1 = candy_data.loc[row1, candy_data['winpercent'] == candy_data['winpercent'].max()]

    more_popular_row2 = candy_data.loc[row2, candy_data['winpercent'] == candy_data['winpercent'].max()]

    return max(more_popular_row1, more_popular_row2)

    

    more_popular_row2 = candy_data.loc[row2, candy_data['winpercent'] == candy_data['winpercent'].max()]

    

    more_popular_row1 = candy_data.loc[[row1, row2], candy_data['winpercent'] == candy_data['winpercent'].max()]

    return max(more_popular_row1, more_popular_row2)

    

more_popular_row1 = candy_data.loc[candy_data['competitorname'] == row1, candy_data['winpercent'] == candy_data['winpercent'].max()]

more_popular_row2 = candy_data.loc[candy_data['competitorname'] == row2, candy_data['winpercent'] == candy_data['winpercent'].max()]

    

    more_popular_row1_row2 = candy_data.loc[[row1, row2], candy_data['winpercent'] == candy_data['winpercent'].max()]



more_popular_row1 = candy_data.loc[candy_data.loc[:, 'competitorname'] == row1, candy_data['winpercent'] == candy_data['winpercent'].max()]

    more_popular_row2 = candy_data.loc[candy_data.loc[:, 'competitorname'] == row2, candy_data['winpercent'] == candy_data['winpercent'].max()]





more_popular_row1 = candy_data.loc[candy_data['competitorname'].values[1] == row1, candy_data['winpercent'] == candy_data['winpercent'].max()]

    more_popular_row2 = candy_data.loc[candy_data['competitorname'].values[3] == row2, candy_data['winpercent'] == candy_data['winpercent'].max()]



https://stackoverflow.com/questions/44352271/label-not-in-list-and-keyerror



def find_more_popular(row1, row2):

    more_popular_row1 = candy_data.loc[candy_data['competitorname'] == row1, 'winpercent'].item()

    more_popular_row2 = candy_data.loc[candy_data['competitorname'] == row2, 'winpercent'].item()

    return more_popular_row1, more_popular_row2



"""



"""

def find_max_col(x):

    max_col_x = candy_data.iloc[:, x].values

    return max(max_col_x)



more_popular = find_max_col(12)

print(more_popular)



loc[:,['competitorname','winpercent']]

"""



"""

Now I want a function that finds the max value for a coloumn among two arbitrary rows

The code compares an input column value for input rows, finds the max, then uses the 

max value to find the value in competitorname corresponding to the the input column max value

def find_more_popular_any_col(row1, row2, column):

    more_popular_row1 = candy_data.loc[candy_data['competitorname'] == row1, column]

    more_popular_row2 = candy_data.loc[candy_data['competitorname'] == row2, column]

    more_popular_row1_row2 = max(more_popular_row1, more_popular_row2)

    answer = candy_data.loc[candy_data[column] == more_popular_row1_row2, 'competitorname']

    return answer



more_popular_AH_BR_sugar = find_more_popular_any_col('Air Heads', 'Baby Ruth', 'sugarpercent')

print(more_popular_AH_BR_sugar)



The below clears error of unable to compare series with different label



def find_more_popular_any_col(row1, row2, column):

    more_popular_row1 = candy_data.loc[candy_data['competitorname'] == row1, column]

    more_popular_row2 = candy_data.loc[candy_data['competitorname'] == row2, column]

    max_row1 = more_popular_row1.max()

    max_row2 = more_popular_row2.max()

    maxmax = max(max_row1, max_row2)

    return more_popular_row1, more_popular_row2, maxmax



With the below i return a list of the values in competitorname that match the max for my column of choice    

    

    def find_more_popular_any_col(row1, row2, column):

    more_popular_row1 = candy_data.loc[candy_data['competitorname'] == row1, column]

    more_popular_row2 = candy_data.loc[candy_data['competitorname'] == row2, column]

    max_row1 = more_popular_row1.max()

    max_row2 = more_popular_row2.max()

    maxmax = max(max_row1, max_row2)

    answer = candy_data.loc[candy_data[column] == maxmax, 'competitorname'].tolist() # returns all the values that match maxmax

    return answer

    

I can check if the value of the 2 rows I input initially are in the list, and then return just that value as proper answer



# what if they are both in matches? If they are both in matches I first check as basecase 



"""



# Fill in the line below: Which candy has higher sugar content: 'Air Heads'

# or 'Baby Ruth'? (Please enclose your answer in single quotes.)

more_sugar = 'Air Heads'



more_sugar = find_more_popular_any_col('Air Heads', 'Baby Ruth', 'sugarpercent')

print(more_sugar)





# Check your answers

# step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'

plt.figure(figsize = [10, 5])

sns.scatterplot(x = candy_data['sugarpercent'], y=candy_data['winpercent']) # Your code here



# Check your answer

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution_plot()
#step_3.b.hint()
step_3.b.solution()
# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'

plt.figure(figsize = [10, 5])

sns.regplot(x = candy_data['sugarpercent'], y=candy_data['winpercent']) # Your code here



# Check your answer

step_4.a.check()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution_plot()
#step_4.b.hint()
#step_4.b.solution()
# Scatter plot showing the relationship between 'pricepercent', 'winpercent', and 'chocolate'

plt.figure(figsize = [10, 5])

# sns.scatterplot(x = candy_data['sugarpercent'], y = candy_data['winpercent'], hue = candy_data['chocolate']) # Your code here



sns.scatterplot(x = 'sugarpercent', y = 'winpercent', hue = 'chocolate', data = candy_data)



# Check your answer

step_5.check()
# Lines below will give you a hint or solution code

#step_5.hint()

#step_5.solution_plot()
# Color-coded scatter plot w/ regression lines

# cant use this here plt.figure(figsize = [10, 5])

sns.lmplot(x = "sugarpercent", y = "winpercent", hue = "chocolate", data = candy_data) # Your code here





# Check your answer

step_6.a.check()
# Lines below will give you a hint or solution code

#step_6.a.hint()

#step_6.a.solution_plot()
#step_6.b.hint()
step_6.b.solution()
# Scatter plot showing the relationship between 'chocolate' and 'winpercent'

plt.figure(figsize = [10, 5]) 

sns.swarmplot(x = 'chocolate', y = 'winpercent', data = candy_data) # Your code here



# Check your answer

step_7.a.check()
# Lines below will give you a hint or solution code

#step_7.a.hint()

#step_7.a.solution_plot()
#step_7.b.hint()
step_7.b.solution()