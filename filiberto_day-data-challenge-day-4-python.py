import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # for plotting



# read in our data

dataframe = pd.read_csv("../input/anonymous-survey-responses.csv")

# look at the first few rows

dataframe.head()



# plt.bar(left, data=dataframe)
# Matplotlib is a little bit involved, so we need to do quite  abit of

# data manipulation before we can make our bar chart 



## data preparation



# count how often each pet preference is observed 

petFreqTable = dataframe["Just for fun, do you prefer dogs or cat?"].value_counts()



# to check the data structure 

preFreqTable.__dict__



# This will get us a list of the names

list(petFreqTable.index)



# This will get us a list of the counts

petFreqTable.values



# get all the names from our frequency plot & save them for later

labels = list(petFreqTable.index)



# generate a list of number as long as our number of labels

# len() outputs an integer, you need list() function to slice

positionForBars = list(range(len(labels)))



## actual plotting



# pass the names and counts to the bar function

plt.bar(positionForBars, petFreqTable.values) # plot our bars

plt.xticks(positionForBars, labels) # add labels

plt.title("Pet Preferences")
# We can also use the seaborn package to plot



# import seaborn and alias it as an sns

import seaborn as sns



# make a barplot from a column in our dataframe

sns.countplot(dataframe["Just for fun, do you prefer dogs or cat?"])