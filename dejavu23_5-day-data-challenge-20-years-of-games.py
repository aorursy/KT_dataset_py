import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

#warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
games = pd.read_csv("../input/ign.csv")

games.head()
games.describe()
games.describe(include = ['O'])
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

plt.hist(games['score'], bins=19)

plt.xlim([0,11])   

plt.xlabel('Score')

plt.ylabel('Counts')

plt.title('Distribution of score for all games')

plt.grid(linestyle='dotted')

plt.show()
from scipy.stats import probplot # for a qqplot

import pylab

probplot(games["score"], dist="norm", plot=pylab)  

plt.show()
from scipy.stats import ttest_ind



scores_PS2 = games["score"][games["platform"] == "PlayStation 2"]

scores_Xbox360 = games["score"][games["platform"] == "Xbox 360"]



ttest_ind(scores_PS2, scores_Xbox360, equal_var=False)
print(scores_PS2.mean())

print(scores_PS2.std())

print(scores_Xbox360.mean())

print(scores_Xbox360.std())
plt.figure(figsize=(7,5))

plt.hist(scores_PS2, alpha=0.6, bins=19, label="PS2")

plt.hist(scores_Xbox360, alpha=0.6, bins=19, label="Xbox360")

plt.xlim([0,11])   

plt.xlabel('Score')

plt.ylabel('Counts')

plt.title('Distribution of score for PS2 and Xbox360 games')

plt.grid(linestyle='dotted')

plt.legend()

plt.show()
probplot(scores_PS2 , dist="norm", plot=pylab)

plt.show()
probplot(scores_Xbox360 , dist="norm", plot=pylab)

plt.show()
games['score_phrase'].value_counts()
import seaborn as sns

fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

sns.countplot(games['score_phrase'],ax=ax)

plt.xticks(rotation=45)

plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(10, 5)

sns.barplot(x='release_year', y='score', data=games, ax=ax)

plt.xticks(rotation=45)

plt.show()
games['genre'].value_counts().head(10)
advt_racg = games[(games.genre == 'Adventure') | (games.genre == 'Racing')]

contingencyTable = pd.crosstab(advt_racg.score_phrase, advt_racg.genre, margins=True)

contingencyTable
from scipy import stats

chi2, p, dof, expctd = stats.chi2_contingency(contingencyTable)

print("chi2 :", chi2)

print("p value :", p)
advt_strg = games[(games.genre == 'Adventure') | (games.genre == 'Strategy')]

contingencyTable_2 = pd.crosstab(advt_strg.score_phrase, advt_strg.genre, margins=True)

contingencyTable_2
chi2, p, dof, expctd = stats.chi2_contingency(contingencyTable_2)

print("chi2 :", chi2)

print("p value :", p)