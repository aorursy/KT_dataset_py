# Run this code block to set up Python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

print("Setup Complete")
# This cell will load the data for the exercise - just run it, you don't need to change it.

# Line chart data
museum_filepath = "../input/data-for-datavis/museum_visitors.csv"
museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)

# Bar chart data
ign_filepath = "../input/data-for-datavis/ign_scores.csv"
ign_data = pd.read_csv(ign_filepath, index_col='Platform')

# Scatter plot data
candy_filepath = "../input/data-for-datavis/candy.csv"
candy_data = pd.read_csv(candy_filepath, index_col='id')

# Histogram data (Text)
iris_filepath = "../input/data-for-datavis/iris.csv"
iris_data = pd.read_csv(iris_filepath, index_col="Id")

# Histogram data (Exercise)
cancer_b_filepath = "../input/data-for-datavis/cancer_b.csv"
cancer_m_filepath = "../input/data-for-datavis/cancer_m.csv"
cancer_b_data = pd.read_csv(cancer_b_filepath, index_col='Id')
cancer_m_data = pd.read_csv(cancer_m_filepath, index_col='Id')

print("Data Loaded")
# When I'm trying to answer questions like this, I'll sometimes write some lines of code exploring the data. This might include commands like head(), describe(), columns, or 
# just slicing and dicing the data in ways that I need. 

# I've left this cell here with a head method, so you can explore the data, if you'd like.
museum_data.head()
# The next two code blocks are from Step 2 in the line charts lesson 
# Hint: I used .loc when answering these questions. Also remember that you can make Python do math for you!

# Fill in the line below: How many visitors did the Chinese American Museum 
# receive in July 2018?
ca_museum_jul18 = _____
print(ca_museum_jul18) # This should print 2620, without hardcoding the value
assert(ca_museum_jul18 == 2620) # This is a handy testing syntax to use in Python - it will throw an exception if the assertion (the value in the parentheses) is false
# Fill in the line below: In October 2018, how many more visitors did Avila 
# Adobe receive than the Firehouse Museum?
avila_oct18 = ______
print(avila_oct18) # This should print 14658, without hardcoding the value
assert(avila_oct18 == 14658)
# The next three code blocks are from Step 4B in the line charts lesson 
# Hint: I used Series slices when answering these questions. Recall that Series slices don't have to be integers, they can also be index values.

# Fill in the line below to list how many visitors Avila Adobe got from September 2014 - Feburary 2015? 
# (Hint: Use a slice, but don't use integers) 
winter_2015_visitors_to_avila = _____
print(winter_2015_visitors_to_avila) # Should print 136929, without hardcoding the value
assert(winter_2015_visitors_to_avila == 136929)
# Fill in the line below to list how many visitors Avila Adobe got from March 2015 - August 2015?  
summer_2015_visitors_to_avila = _____ 
print(summer_2015_visitors_to_avila) # Should print 184861, without hardcoding the value
assert(summer_2015_visitors_to_avila == 184861)
# Use the winter_2015_visitors_to_avila and summer_2015_visitors_to_avila results to set is_more_visitors_in_summer to True if there are more visitors in summer, to False if not. 
# Hint: Create a boolean expression
is_more_visitors_in_summer = ______ 
print(is_more_visitors_in_summer) #Should print True, without hardcoding the value
assert(is_more_visitors_in_summer == True)
# As a reminder, the data frame called ign_data stores this information. 
# I've left this cell here with a head method, so you can explore the data, if you'd like.
ign_data.head()
# This cell block is from Step 2 in the bar chart module. 
# Hint: In addition to the min and max command on Series, look up the idxmin and idxmax commands.

# Fill in the line below: What is the highest average score received by PC games, for any platform?
high_score = _____
print(high_score) # This should print 7.759930313588848, without hardcoding the value
assert(high_score == 7.759930313588848)

# Fill in the line below: On the PlayStation Vita platform, which genre has the 
# lowest average score? 
worst_genre = _____
print(worst_genre) # This should print Simulation, without hardcoding the value
assert(worst_genre == 'Simulation')
# This cell block is from Step 4B in the bar char module. This will allow you to practice combinations of min, max, idxmin, and idxmax.
# Remember, no hardcoding!

# What is the name of the genre with the absolute highest rating?
highest_genre = _____ 
assert(highest_genre == 'Simulation')

# What is the name of the genre with the absolute lowest rating?
lowest_genre = _____ 
assert(lowest_genre == 'Fighting')

# What is the largest score in the highest_genre series?
highest_score = ____ 
assert(highest_score == 9.25)

# What is the lowest score in the lowest_genre series?
lowest_score = _____ 
assert(lowest_score == 4.5)

# Which platform in the highest genre has the highest score?
highest_platform = _____ 
assert(highest_platform == 'PlayStation 4')

# Which platform in the lowest genre has the lowest score?
lowest_platform = _____ 
assert(lowest_platform == 'Game Boy Color')

# This should print...
# Highest Score ==  9.25  Platform ==  PlayStation 4  Genre ==  Simulation
# Lowest Score ==  4.5  Platform ==  Game Boy Color  Genre ==  Fighting
print("Highest Score == ", highest_score, " Platform == ", highest_platform, " Genre == ", highest_genre)
print("Lowest Score == ", lowest_score, " Platform == ", lowest_platform, " Genre == ", lowest_genre)

# Note that this doesn't come up with all possible combinations of lowest score (Ex: the lowest score is both Fighting *AND* Shooting on the Game Boy Color)
# For extra credit, can you come up with a way to print out all of the combinations of highest and lowest genre+platform+score?

# As a reminder, the data frame called candy_data stores this information. You can use this cell to explore the data as you need
candy_data.head()
# Fill in the line below: Which candy was more popular with survey respondents: '3 Musketeers' or 'Almond Joy'? 
# Hint: Create two filters. One for the 3 Musketeers candies in the data frame and one for the Almond Joy candies in the data frame. 
# Need a filters refresher? https://youtu.be/Lw2rlcxScZY
musketeers_filter = _____
almondjoy_filter = _____

# Now use those filters in either a ternary operation or in an if/else block to identify which candy has the highest winning percentage. 
# Hint: Be careful of the type of data you have after using the filter - depending how you did it, you may need to get numeric data instead!
more_popular = _____
print(more_popular)
assert(more_popular == '3 Musketeers')

# Fill in the line below: Which candy has higher sugar content: 'Air Heads' or 'Baby Ruth'? (Hint: This should be the same process as the previous code block)
more_sugar = ____ 
print(more_sugar)
assert(more_sugar == 'Air Heads')
# As a reminder, the data frame called iris_data stores this information. You can use this cell to explore the data as you need.
iris_data.head()
# In the explanation, the author manually creates and loads three new files for each of the different species of iris. Why let a human do what a computer can do better?

# Create a filter and dataframe that only stores Iris-setosa examples
iris_setosa_filter = _____
iris_setosa_df = _____

# Now do the same for Iris-versicolor

# And for Iris-virginica

print(iris_setosa_df.Species.value_counts()) # This should print Iris-setosa    50 (and the series name and data type)
print(iris_versicolor_df.Species.value_counts()) # This should print Iris-versicolor    50 (and the series name and data type)
print(iris_virginica_df.Species.value_counts()) # This should print Iris-virginica    50 (and the series name and data type)

# As a reminder, there are two data frames for this data. One with benign cancers (cancer_b_data) and one with malignant cancers (cancer_m_data)
# You can use this cell to explore the data as you need
cancer_b_data.head()
cancer_m_data.head()

# Note: Because these are not print statement, it will only print the *last* head command.
# Create a slice of the first five rows of the benign cancers
benign_slice = cancer_b_data[_____]
print(list(benign_slice.index)) # This will print the "Id" of each benign cancer as a list [8510426, 8510653, 8510824, 854941, 85713702]

# Create a slice of the first five rows of the malignant cancers
malignant_slice = _____
print(list(malignant_slice.index)) # This will print the "Id" of each malignant cancer as a list [842302, 842517, 84300903, 84348301, 84358402]

# Put the two data frames into the same data frame (You won't be using a join or merge, we'll be using something else. Hint: Notice that both frames have the same column names)
both_slices = _____
print(list(both_slices.index)) # This will print the "Id" of both sets of cancers as a list 
# Specifically it will print either
# [8510426, 8510653, 8510824, 854941, 85713702, 842302, 842517, 84300903, 84348301, 84358402]
# OR
# [842302, 842517, 84300903, 84348301, 84358402, 8510426, 8510653, 8510824, 854941, 85713702]

# Now, let's use those slices.

# In the first five rows of the data for benign tumors, what is the largest value for 'Perimeter (mean)'?
max_perim = _____
print(max_perim) # Should print 87.46
assert(max_perim == 87.46)

# Fill in the line below: What is the value for 'Radius (mean)' for the tumor with Id 842517? (Make sure you use the both_slices data frame)
mean_radius = _____
print(mean_radius) # Should print 20.57
assert(mean_radius == 20.57)
