import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

#Read the csv using Pandas and create a dataframe

df = pd.read_csv("../input/anonymous-survey-responses.csv")
#See few values of dataframe

df.head()
freqtable = df['Just for fun, do you prefer dogs or cat?'].value_counts()

#labels= list(freqtable.indexes)
labels= list(freqtable.index)

listofnumbers = list(range(len(freqtable.index)))

plt.bar(listofnumbers,freqtable.values)

plt.xticks(listofnumbers,labels)
# Bar chart using seaborn 

import seaborn as sns
sns.countplot(df['Just for fun, do you prefer dogs or cat?'])