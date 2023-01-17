# libraries

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
# relevant data

googleplaystore = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
# data overview

googleplaystore.head(5)
# keep only usefull columns

googleplaystore = googleplaystore[['Category', 'Installs']]



# clean "Category" names

googleplaystore.Category = googleplaystore.Category[googleplaystore.Category != '1.9'].apply(lambda x: x.replace('_', ' ').capitalize())



# clean and convert "Installs" to float64

googleplaystore.Installs = googleplaystore.Installs[googleplaystore.Installs != 'Free'].apply(lambda x: int(x.replace('+', '').replace(',', '')))
most_popular = googleplaystore.groupby('Category').sum().sort_values(by=['Installs'], ascending=False)
sns.set(style="darkgrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(13, 10))



# Plot the total

sns.barplot(x="Installs", y=most_popular.index, data=most_popular, label="Installs", color="blue")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(ylabel="Mobile Application Categories", xlabel="Total number of installations (in billions)")

sns.despine(left=True, bottom=True)