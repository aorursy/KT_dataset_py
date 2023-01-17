# Import libaries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Import dataset

spotify_dataset = pd.read_csv('../input/top2018.csv')

spotify_dataset.head(5)
# print dataset shape

spotify_dataset.shape
# Lets check the artists we have here in top 100 tracks and the number of their songs made it to the top 100

plt.figure(figsize=(18,12))

spotify_dataset['artists'].value_counts().plot.bar()
# Lets see the correlation between features in our dataset

plt.figure(figsize=(15,10))

sns.heatmap(spotify_dataset.corr(), 

            xticklabels=spotify_dataset.corr().columns.values,

            yticklabels=spotify_dataset.corr().columns.values)
# Plot a scatter mattrix

sns.set(style="whitegrid")

sns.pairplot(spotify_dataset)