import os

import pandas as pd

import numpy as np

import numpy.ma as ma

import seaborn as sns

import matplotlib.pyplot as plt
#matches_df = pd.read_csv(os.path.join('data', 'game_stats', 'matches.csv'), low_memory=False)

matches_df = pd.read_csv("../input/major-league-soccer-dataset/matches.csv", low_memory=False)

matches_df.columns.tolist()[:14]
# Only grab certain columns

matches_df = matches_df[['id', 'home', 'away', 'date', 'year', 'part_of_competition', 'game_status', 'home_score', 'away_score']]

matches_df.head()
# Only want regular season games

matches_df = matches_df[ matches_df['part_of_competition'].str.contains('Regular Season')]

matches_df.part_of_competition.unique()
matches_df.game_status.unique()
# Check out what games are 'Abandoned'

matches_df[matches_df['game_status']=='Abandoned'] # remove this row
matches_df = matches_df.drop(index=2306) # matches_df = matches_df[ matches_df['game_status'] != 'Abandoned' ]

matches_df[matches_df['game_status']=='Abandoned'] # should be 0 rows
# Find the max score (for size of 2D array)

max_score = max(matches_df['home_score'].max(), matches_df['away_score'].max())

ms = max_score + 1

max_score
# Convert scores to matrix

scores_np = np.zeros((ms,ms), dtype=np.int32)



for i, row in matches_df.iterrows():

    if row['home_score'] > row['away_score']: # home won

        scores_np[row['home_score']][row['away_score']] += 1

    else: # draw or away won

        scores_np[row['away_score']][row['home_score']] += 1

        

scores_df = pd.DataFrame(scores_np).transpose()

scores_df
# Reverse the rows so the heatmap looks better (hopefully)

scores_df = scores_df.iloc[::-1]

scores_df
# Make the mask so the empty half displays differently on the heatmap

mask = np.zeros_like(scores_df)

mask[np.triu_indices_from(mask, k=1)] = True

mask = np.array(mask, dtype=np.float32)

mask = np.flip(mask, axis=1)

mask
# Make the annotations (replace '0's with a '-')

annots = pd.DataFrame(scores_df, dtype=str)

for i in range(ms):

    for j in range(ms):

        if annots[i][j] == "0":

            annots[i][j] = '-'

annots
# Plot it as a heatmap

# https://seaborn.pydata.org/generated/seaborn.heatmap.html



with sns.axes_style("darkgrid"):

    f, ax = plt.subplots(figsize=(6, 6))

    ax = sns.heatmap(scores_df.astype(float), linewidths=0.5, annot=annots, 

                     fmt='s', mask=mask, cbar=False, cmap='Blues', annot_kws={'fontweight':'demi'})

    

plt.yticks(rotation=0)

plt.xlabel('Winner', size=20)

plt.ylabel('Loser', size=20)

plt.suptitle('MLS Final Score Appearances', size=25)

plt.savefig("./winner_loser_number.png", bbox_inches='tight')

plt.show()
# Make the annotations and store the percents

annots = pd.DataFrame(scores_df, dtype=str)

percents = pd.DataFrame(scores_df, dtype=float)

total_games = len(matches_df)



for i in range(ms):

    for j in range(ms):

        if annots[i][j] == "0":

            annots[i][j] = '-'

            percents[i][j] = 0.0

        else:

            annots[i][j] = str('{:.2f}'.format((float(annots[i][j]) / total_games)*100))

            percents[i][j] = 100 * float(percents[i][j] / total_games)

annots
# Plot it as a heatmap



with sns.axes_style("darkgrid"):

    f, ax = plt.subplots(figsize=(6, 6))

    ax = sns.heatmap(scores_df.astype(float), linewidths=0.5, annot=annots, 

                     fmt='s', mask=mask, cbar=False, cmap='Blues', annot_kws={'fontweight':'demi'})

    

plt.yticks(rotation=0)

plt.xlabel('Winner', size=20)

plt.ylabel('Loser', size=20)

plt.suptitle('Final Score Frequencies', size=25, y=1)

plt.title('As % of All Games')

plt.savefig("./winner_loser_freq.png", bbox_inches='tight')

plt.show()
# Do it again but with home and away instead of winner and loser

scores_ha = np.zeros((ms,ms), dtype=np.int32)



for i, row in matches_df.iterrows(): 

    scores_ha[row['home_score']][row['away_score']] += 1

scores_df = pd.DataFrame(scores_ha).transpose()

scores_df = scores_df.iloc[::-1]

scores_df
# Make the annotations

annots = pd.DataFrame(scores_df, dtype=str)

for i in range(ms):

    for j in range(ms):

        if annots[i][j] == "0":

            annots[i][j] = '-'

annots
# Note: We don't need a mask this time since we're not forcing the data to 

#       half of the 2D array



with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(6, 6))

    ax = sns.heatmap(scores_df, linewidths=0.5, annot=annots, 

                     fmt='s', cbar=False, cmap='Greens', annot_kws={'fontweight':'demi'})

    

plt.yticks(rotation=0)

plt.xlabel('Home', size=20)

plt.ylabel('Away', size=20)

plt.suptitle('Final Score Appearances', size=25)

plt.savefig("./home_away_number.png", bbox_inches='tight')

plt.show()
# Make the annotations

annots = pd.DataFrame(scores_df, dtype=str)

total_games = len(matches_df)



for i in range(ms):

    for j in range(ms):

        if annots[i][j] == "0":

            annots[i][j] = '-'

        else:

            annots[i][j] = str('{:.2f}'.format((float(annots[i][j]) / total_games)*100))



annots
with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(6, 6))

    ax = sns.heatmap(scores_df, linewidths=0.5, annot=annots, 

                     fmt='s', cbar=False, cmap='Greens', annot_kws={'fontweight':'demi'})

    

plt.yticks(rotation=0)

plt.xlabel('Home', size=20)

plt.ylabel('Away', size=20)

plt.suptitle('Final Score Frequencies', size=25, y=1)

plt.title('As % of All Games')

plt.savefig("./home_away_freq.png", bbox_inches='tight')

plt.show()
# Percents for games in which a certain number of goals isn't exceeded by either team

for under in range(1, ms): # 1 through 8

    per_un = 0.0

    for i in range(under+1): # 0 through under (want it to include under as well)

        for j in range(under+1): # 0 through under

            per_un += percents.loc[i, j]

    print('{:.4f}% of games feature neither team scoring over {} goals'.format(per_un, under))
# Percents for total goals in a game

total_goals = {}

total_goals_p = {}

for i in range(len(percents)):

    for j in range(len(percents)):

        if percents.loc[i,j] == 0.0: continue

        if i + j not in total_goals:

            total_goals[i+j] = 0

            total_goals_p[i+j] = 0.0

        total_goals_p[i+j] += percents.loc[i,j]

        total_goals[i+j] += scores_df.loc[i,j]



tg_s = pd.Series(total_goals_p)

fig, ax = plt.subplots(figsize=(7,5))

rects = plt.bar(tg_s.index, height=tg_s, color='orange')

plt.xlabel('Total Goals', size=18)

plt.ylabel('Percent of Games', size=18)

plt.suptitle('Total Goals in Game by Percent', size=24, y=1)

plt.title('Number of Occurrences Above Bar')

plt.xticks(range(0,12),range(0,12))

plt.yticks(range(0,26,5), range(0,26,5))



# https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

# ^ helpful for getting labels on a bar graph

for i, rect in enumerate(rects):

    ax.annotate(total_goals[i], xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()), 

                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    #ax.annotate('{:.2f}'.format(total_goals_p[i]), xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()), 

     #           xytext=(0, 14), textcoords="offset points", ha='center', va='bottom')



plt.savefig("./total_goals_bar.png", bbox_inches='tight')

plt.show()
# Next Up: Do a similar procedure for the Premier League and La Liga

# Compare the three leagues