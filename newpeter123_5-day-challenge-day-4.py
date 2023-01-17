import pandas as pd

import matplotlib.pyplot as plt



dataframe = pd.read_csv("../input/anonymous-survey-responses.csv")

dataframe.head()

petFreqTable = dataframe["Just for fun, do you prefer dogs or cat?"].value_counts()



# petFreqTable.__dict__



petFreqTable = dataframe["Just for fun, do you prefer dogs or cat?"].value_counts()



print(petFreqTable.index)

print(petFreqTable.values)



labels = list(petFreqTable.index)



positionsForBars = list(range(len(labels)))



plt.bar(positionsForBars, petFreqTable.values)

plt.xticks(positionsForBars, labels)

plt.title("Pet Pref")



# get all the name from our frequency plt

labels = list(petFreqTable.index)





print(list(range(len(labels))))

print(list(labels))
import seaborn as sns



sns.countplot(dataframe["Just for fun, do you prefer dogs or cat?"])