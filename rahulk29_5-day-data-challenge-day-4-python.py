import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting

# read in our data
dataframe = pd.read_csv("../input/5day-data-challenge-signup-survey-responses/anonymous-survey-responses.csv")
dm = pd.read_csv("../input/digidb/DigiDB_digimonlist.csv")
# look at the first few rows
dataframe.head()
dm.head(20)
s = dm.groupby("Stage")
d1 = dm['Type'].value_counts()
ss = dm.Stage
labels = list(d1.index)
bars = list(d1.values)
#print(labels, bars)
#bars2 =  
plt.bar(range(len(labels)),bars)
#plt.bars()
plt.xticks(range(len(labels)),labels)
plt.title("Digimon Types")
plt.show()
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

## actual plotting

# pass the names and counts to the bar function
plt.bar(positionsForBars, petFreqTable.values) # plot our bars
plt.xticks(positionsForBars, labels) # add lables
plt.title("Pet Preferences")
print(positionsForBars)
import seaborn as sns
sns.countplot(dm["Stage"]).set_title("Digimon Stages")
# We can also use the seaborn package to plot

# import seaborn and alias it as sns
import seaborn as sns

# make a barplot from a column in our dataframe
sns.countplot(dataframe["Just for fun, do you prefer dogs or cat?"]).set_title("Dogs vs. Cats")