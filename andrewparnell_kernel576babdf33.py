# Import the Pandas Python library for use in working with data

import pandas



# Import our data from CSV file into a Pandas dataframe for us to play with

battles = pandas.read_csv('/kaggle/input/final-fantasy-tactics-battles/fft_red-battles.csv')
# Call the 'head' (the first 5 rows of the data) to get a feel for what it looks like

battles.head()
# I would like to see a chart of how many guests join you in battles as the game goes on

battles.plot.bar('chronological','guests',figsize=(20,8))
# I would like to see how the number of enemies you face breaks down as the game goes on



# Since we dont have a column for the count of enemies, we need to build one

# Start by getting a subset of all columns that end with "_job" and setting it to enemies

enemies = [col for col in battles if col.endswith('_job')]

# Use this subset of columns that contain the enemy data to count the number of nonzero enemies (axis=1 means to look across columns rather than rows)

battles['num_of_enemies'] = battles[enemies].count(axis=1)

# Plot the bar chart

battles.plot.bar('chronological','num_of_enemies',figsize=(20,8))
# Which battles feature more than 6 enemies?

battles[battles['num_of_enemies'] > 6][['chronological','chapter','location','num_of_enemies']]
# No wonder that Execution Site battle was so hard!

# Let's see if we can get a scatterplot of the average enemy level per battle

# Start by creating a subset of columns that contain the levels of all enemies

enemy_levels = [col for col in battles if col.endswith('_level')]

# Get the average of that subset of columns (axis=1 means average the columns rather than the rows)

battles['average_enemy_level'] = battles[enemy_levels].mean(axis=1)



battles[['location', 'average_enemy_level']]
# Now lets try that scatterplot!

battles.plot.scatter('chronological','average_enemy_level', figsize=(20,8))
# Now what would be REALLY sweet is if we could see the distribution of unit types among enemies we battle against. 

# Any guesses as to the top 10? Im guessing "Knight" will probably be the most common?



# Lets subset only the columns containing enemy jobs, then "stack" the results, meaning shove them all into a single column,

# then lets use value_counts to get a count of each unique result

battles[enemies].stack().value_counts()
# I was able to add the full skills tree to the dataset as well. Let's look at the head 

# maybe there is some fun things we can do with this?

skills = pandas.read_csv('/kaggle/input/final-fantasy-tactics-battles/fft_skills.csv')

skills.head()
# Can we see a clean breakdown of JP required by job?

skills.boxplot('jp','job',figsize=(20,8))