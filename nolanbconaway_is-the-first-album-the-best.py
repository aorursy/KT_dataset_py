import sqlite3, datetime

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import linregress

from scipy.stats import ttest_rel



pd.set_option('precision', 2)

np.set_printoptions(precision=2)



con = sqlite3.connect('../input/database.sqlite')

scores = pd.read_sql('SELECT reviewid, score FROM reviews', con)

artists = pd.read_sql('SELECT * FROM artists', con)

years = pd.read_sql('SELECT * FROM years', con)

con.close()



# combine into score-artist mapping

reviews = pd.merge(scores, artists, on = 'reviewid')



# remove various artists

reviews = reviews[reviews.artist != 'various artists']



# remove multi-year reviews [re-releases]

year_counts = years.groupby('reviewid').count().reset_index()

keepers = year_counts.loc[year_counts.year == 1, 'reviewid']

reviews = reviews.loc[reviews.reviewid.isin(keepers)]
numreviews = reviews.groupby('artist').size()



num_artists = len(numreviews)

most_reviews = numreviews.max()

most_reviewed = numreviews[numreviews >= most_reviews-2].index.values



#plot histogram

sns.distplot(numreviews, range(1, numreviews.max()), kde = False)

plt.xlabel('Number of Reviews')

plt.ylabel('Number of Artists')





# create descriptive box

S = (

    "Unique Artists = " + str(num_artists) + '\n' + 

    "Max Reviews = " + str(most_reviews) + '\n\n' + 

    "--Most Reviewed--" 

    )



for i in most_reviewed:

    S+= '\n' + i



plt.text(24.5, 4800, S, backgroundcolor = 'w', va = 'top', ha = 'right')



plt.title('Histogram of Number of Reviews')

plt.show()
for x, rows in numreviews.groupby(numreviews):

    group_scores = reviews.loc[reviews.artist.isin(rows.index.values),'score']

    plt.boxplot(group_scores.as_matrix(), positions = [x], widths = 0.5,

               manage_xticks = False)

plt.xlim([0,23])

plt.ylim([-0.2,10.2])

plt.ylabel('Score')

plt.xlabel('Number of Reviews')

plt.title('Average Score by Number of Reviews')



plt.show()
# remove artists with less than 2 reviews

keep_artists = numreviews[numreviews > 1].index.values

twoplus = reviews.loc[reviews.artist.isin(keep_artists)]



# add review number per artist

twoplus = twoplus.assign(number = pd.Series(index=twoplus.index))

for a, rows in twoplus.groupby('artist'):

    values = list(reversed(range(rows.shape[0])))

    twoplus.set_value(rows.index, 'number', values)

    

twoplus['number'] = twoplus['number'].astype(int)
# plot boxes

sns.boxplot(x = 'number', y = 'score',  data = twoplus)



# add regression

reg = linregress(twoplus.number, twoplus.score)

print(reg)



x = np.arange(-1, 23)

y = reg.intercept + x*reg.slope

plt.plot(x,y,'--', color ='black', linewidth = 3,

    label = 'Regression, p = ' + str(round(reg.pvalue, 3)))

plt.legend(loc = 'lower right')



plt.xlabel('Review Number')

plt.title('Score by Review Number')

plt.ylim([-0.2, 10.2])

plt.show()
finalsplit = twoplus.copy()



# add final release indicator

finalsplit = finalsplit.assign(final = pd.Series(index=finalsplit.index))

for a, rows in finalsplit.groupby('artist'):

    

    values = rows.number == max(rows.number)

    values[values == False] = 'Not Final'

    values[values == True ] = 'Final Release'

    finalsplit.set_value(rows.index, 'final', values)
for i, rows in finalsplit.groupby('final'):

    sns.kdeplot(rows.score, shade = True, label = i)

plt.xlabel('Score')

plt.ylabel('Density')

plt.show()
groups = finalsplit.groupby(['final','artist'])

groups = groups['score'].agg('mean')



t = ttest_rel(groups['Not Final'], groups['Final Release'])

print('T Test, p = ' + str(t.pvalue) + '\n')



for i, rows in finalsplit.groupby('final'):

    S  = i + '\t' + 'M = ' + str(round(groups[i].mean(),4))

    S += '\tSD = ' + str(round(groups[i].std(),4))

    print(S)