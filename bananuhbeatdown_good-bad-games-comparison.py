import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import cm



# load the dataset

path = '../input/ign.csv'

data = pd.read_csv(path, index_col=0)



# delete url column

data = data.drop('url', 1)



# delete any row where the release_year is less than 1995

data = data.drop(data[data.release_year < 1995].index)

data.head()
# rounded game scores total count from 1996-2016

round_scores = data.score.round(0)

bins = np.arange(0, 12) - 0.5

plt.figure(figsize=(12,6))

plt.hist(round_scores, bins=bins, width=0.8)

plt.xticks(np.arange(0,11))

plt.autoscale()

plt.ylabel('Game Score Count')

plt.xlabel('Score')

plt.title('Rounded Game Scores from 1996-2016')
# pie chart of rounded scores from 1996-2016

year_score = data.score.round().value_counts().sort_index()

explode = [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0]

colors = cm.Set1(np.arange(11)/11.)

plt.figure(figsize=(12, 12))

plt.pie(year_score, colors=colors, explode=explode, shadow=True, autopct='%1.1f%%')

plt.title("Percentage Breakdown of Rounded Game Scores from 1996-2016")

plt.legend(labels=np.arange(0, 11.0), bbox_to_anchor=(1.075, 0.9), frameon=True, shadow=True, title='SCORE')

plt.axis('equal')
# create a column where any score less than 5.0 is a 1 and otherwise a 0

data['bad_game'] = np.where(data.score < 5.0, 1, 0)

(data.bad_game == 1).value_counts()
# histogram of bad games released by year

%matplotlib inline

bins = np.arange(1996, 2018) - 0.5

plt.figure(figsize=(12, 6))

plt.hist(data.release_year, weights=data.bad_game, bins=bins, width=0.9, color='darkred')

plt.xlabel("Release Year")

plt.ylabel("Bad Game Count")

plt.title("Bad Games Released by Year (Bad = Score < 0.5)")

labels=np.arange(1996, 2017)

plt.xticks(labels, rotation=45)

plt.autoscale()

plt.grid(True)
# pie chart of bad games released by year

bad_years = pd.DataFrame(data.release_year[data.bad_game ==1].value_counts()).sort_index()

colors = cm.Set1(np.arange(21)/21.)

explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]

plt.figure(figsize=(12, 12))

plt.pie(bad_years, shadow=True, colors=colors, explode=explode, autopct='%1.1f%%')

plt.axis('equal')

plt.title("Percentage of Bad Games Released by Year")

plt.legend(labels=bad_years.index, bbox_to_anchor=(1.15, 0.9), frameon=True, shadow=True, title='YEAR')
# create a column where any score 8.0 or above is a 1 and otherwise a 0

data['good_game'] = np.where(data.score > 7.9, 1, 0)

(data.good_game == 1).value_counts()
# histogram of good games released by year

bins = np.arange(1996, 2018) - 0.5

plt.figure(figsize=(12, 6))

plt.hist(data.release_year, weights=data.good_game, bins=bins, color='green', width=0.9)

plt.title("Good Games Released by Year (Good = Score > 7.9)")

plt.xlabel("Release Year")

plt.ylabel("Score")

labels = np.arange(1996, 2017)

plt.xticks(labels, rotation=45)

plt.autoscale()

plt.grid(True)
# pie chart of good games released by year

good_years = pd.DataFrame(data.release_year[data.good_game ==1].value_counts()).sort_index()

colors = cm.Set1(np.arange(21)/21.)

explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0]

plt.figure(figsize=(12,12))

plt.pie(good_years, shadow=True, colors=colors, explode=explode, autopct='%1.1f%%')

plt.axis('equal')

plt.title("Percentage of Good Games Released by Year")

plt.legend(labels=good_years.index, bbox_to_anchor=(1.15, 0.9), frameon=True, shadow=True, title='YEAR')