import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting

# read in our data
survey = pd.read_csv("../input/anonymous-survey-responses.csv")
# look at the first few rows
survey.head()
# look at the dimension of the data 
print(survey.shape)

# look at the columns
print(survey.columns)

# look at a summary of the numeric columns
survey.describe()
# Matplotlib is a little bit involved, so we need to do quite a bit of 
# data manipulation before we can make our bar chart

## data preparation

# count how often each pet preference is observed 
petFreqTable = survey["Just for fun, do you prefer dogs or cat?"].value_counts()

# just FYI: this will get us a list of the names
list(petFreqTable.index)
# just FYI: this will get us a list of the counts
petFreqTable.values

# get all the name from our frequency plot & save them for later
labels = list(petFreqTable.index)

# generate a list of numbers as long as our number of labels
positionsForBars = list(range(len(labels)))

# actual plotting

# pass the names and counts to the bar function
plt.bar(positionsForBars, petFreqTable.values) # plot our bars
plt.xticks(positionsForBars, labels) # add lables
plt.title("Pet Preferences")
# Another option for plotting is the seaborn package, which is much more streamlined, as you can see.

# We can also use the seaborn package to plot

# import seaborn and alias it as sns
import seaborn as sns

# make a barplot from a column in our dataframe
sns.countplot(survey["Just for fun, do you prefer dogs or cat?"]).set_title("Dogs vs. Cats")
# we can also create multiple plots in a single cell
fig, ax = plt.subplots(figsize=(14,8), ncols=2, nrows=2)
plt.subplots_adjust(wspace = .4, hspace = 0.5)
plt.suptitle("Visualization of Survey Results", y = 1, fontsize=20)
sns.countplot(survey["Just for fun, do you prefer dogs or cat?"], ax=ax[0][0])
sns.countplot(survey["Have you ever taken a course in statistics?"], ax=ax[0][1])
sns.countplot(survey["What's your interest in data science?"], ax=ax[1][0])
sns.countplot(survey["Do you have any previous experience with programming?"], ax=ax[1][1])