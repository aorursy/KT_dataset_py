import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# reading the file

data = pd.read_csv('../input/20160307hundehalter.csv')
order = data.ALTER.value_counts()  # Counting how many of each type there are.

order = order.sort_values()  # Ordering types ascendingly.



plt.figure(figsize=(15 , 8))

sns.countplot(data.ALTER, saturation=1, palette="cool", order=order.index)
# count how often each pet preference is observed 

petFreqTable = data.ALTER.value_counts()



# just FYI: this will get us a list of the names

list(petFreqTable.index)

# just FYI: this will get us a list of the counts

petFreqTable.values



# get all the name from our frequency plot & save them for later

labels = list(petFreqTable.index)



# generate a list of numbers as long as our number of labels

positionsForBars = list(range(len(labels)))



# pass the names and counts to the bar function

plt.figure(figsize=(15 , 8))

plt.bar(positionsForBars, petFreqTable.values, color='blue', edgecolor='white') # plot our bars

plt.xticks(positionsForBars, labels) # add lables

plt.title("Dog's age distribution")

plt.xlabel('Dogs Age')

plt.ylabel('Most frequent')
plt.figure(figsize=(15 , 8))

data.ALTER.value_counts().plot(

    kind="bar",

    title="Dog's age distribution"

)