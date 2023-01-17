# Import packages
import os

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

%matplotlib inline
# Read data 
files = os.listdir('../input')
df = pd.read_csv('../input/' + files[0], encoding="ISO-8859-1")
# Select song
song = df.loc[df['Song'] == 'bad romance'].iloc[0]
print(song)
# Populate matrix
words = song['Lyrics'].split()
matrix_one, matrix_two = np.zeros((len(words), len(words))), np.zeros((len(words), len(words)))
for i in range(len(words)):
    for j in range(len(words)):
        if words[i] == words[j]:
            matrix_one[i, j] = words.count(words[i]) / len(words)
            matrix_two[i, j] = list(set(words)).index(words[i])
# Plot it
fig, axarr = plt.subplots(1, 2, figsize=(20,10))
_ = axarr[0].imshow(matrix_one, aspect='equal', cmap='Blues')
_ = axarr[0].axis('off')
_ = axarr[0].set_title('Word occurance')
_ = axarr[1].imshow(matrix_two, aspect='equal')
_ = axarr[1].axis('off')
_ = axarr[1].set_title('Unique identifier')
_ = fig.suptitle(song['Artist'] + " - " + song['Song'])
# Define function to get matrix of a song
def get_score(lyrics):
    try:
        words = lyrics.split()
    except AttributeError: 
        return 0
    if len(words) == 0:
        return 0
    else:
        matrix = np.zeros((len(words), len(words)))
        for i in range(len(words)):
            for j in range(len(words)):
                if words[i] == words[j] and i > j:
                    matrix[i, j] = 2 / (len(words)**2 - len(words))
        return matrix.sum()
# Calculate the means for a subset of the df 
df['score'] = df['Lyrics'].map(get_score)
df = df.loc[df['score'] != 1]
# Plot
fig, axarr = plt.subplots(2, 1, figsize=(15, 15))
_ = axarr[0].plot(df.groupby('Year')['score'].median())
_ = df.boxplot(column='score', by='Year', ax=axarr[1], showfliers=False, grid=False, showmeans=True)
_ = axarr[0].set_ylim(bottom=0)
# Lyrics of most repetitive song
df.loc[df['score'].idxmax()]['Lyrics']
# Top 5 repetative songs
df.nlargest(10, 'score')
# Distinguis top 10 
df['Top 10'] = df['Rank'].map(lambda x: x <= 10)
# Plot repetitiveness for top 10 and non-top 10 songs. 
fig, ax = plt.subplots()
ax = df.boxplot(column='score', by='Top 10', showfliers=False, ax=ax, showmeans=True)
title = fig.suptitle('')
# Plot repetitiveness versus rank
fig, ax = plt.subplots()
ax = df.boxplot(column='score', by='Rank', showfliers=False, ax=ax, showmeans=True, grid=False)
title = fig.suptitle('')
# Add wordcount column 
df = df.dropna()
df['wordcount'] = df['Lyrics'].map(lambda x: len(x.split()))
# Plot wordcount per popularity group, and per year
fig, ax = plt.subplots()
_ = ax.plot(df.loc[df['Top 10'] == True].groupby('Year')['wordcount'].mean(), 'r')
_ = ax.plot(df.loc[df['Top 10'] == False].groupby('Year')['wordcount'].mean(), 'b')
_ = ax.set_ylim(bottom=0)