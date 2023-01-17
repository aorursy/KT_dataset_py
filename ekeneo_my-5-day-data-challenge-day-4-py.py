import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for plotting



# read in our data

dataframe = pd.read_csv("../input/anonymous-survey-responses.csv")



# look at the first few rows

dataframe.head()
# Matplotlib is a little bit involved, so we need to do quite a bit of 

# data manipulation before we can make our bar chart



## data preperation



# count how often each pet preference is observed 

petFreqTable = dataframe["Just for fun, do you prefer dogs or cat?"].value_counts()



# just FYI: this will get us a list of the names

list(petFreqTable.index)



# just FYI: this will get us a list of the counts

petFreqTable.values



# get all the name from our frequency plot & save them for later

labels = list(petFreqTable.index)



# generate a list of numbers as long as our number of labels

positionsForBars = list(range(len(labels)))

## or positionsForBars = list(range(0,len(labels)))





## actual plotting



# pass the names and counts to the bar function

plt.bar(positionsForBars, petFreqTable.values) # plot our bars

plt.xticks(positionsForBars, labels) # add lables

plt.title("Pet Preferences")

plt.show()
# We can also use the seaborn package to plot



# import seaborn and alias it as sns

import seaborn as sns



# make a barplot from a column in our dataframe

sns.countplot(dataframe["Just for fun, do you prefer dogs or cat?"],palette="cool")