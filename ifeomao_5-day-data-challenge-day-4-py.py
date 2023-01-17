import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt # for plotting

import seaborn as sns



# read in our data

dataframe = pd.read_csv('../input/anonymous-survey-responses.csv')

# look at the first few rows

dataframe.head().transpose()
# count how often each pet preference is observed 

petFreqTable = dataframe['Just for fun, do you prefer dogs or cat?'].value_counts()
# just FYI: this will get us a list of the names

print(petFreqTable.index)

# just FYI: this will get us a list of the counts

print(petFreqTable.values)



# pass the names and counts to the bar function

plt.bar(petFreqTable.index, petFreqTable.values)

plt.title('Pet Preferences')
# make a barplot from a column in our dataframe

sns.countplot(x='Just for fun, do you prefer dogs or cat?', data=dataframe).set_title('Pet Preferences')
# make a barplot from a column in our dataframe

sns.countplot(dataframe["Just for fun, do you prefer dogs or cat?"]).set_title("Dogs vs. Cats")