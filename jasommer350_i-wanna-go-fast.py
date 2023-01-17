import numpy as np

import pandas as pd

from scipy.stats import norm, expon, cumfreq

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#df = pd.read_csv('../input/FullData.csv')



#df_league = pd.read_csv('ClubLeagueNames.csv',encoding = "ISO-8859-1")
#dfMerged = df.merge(df_league, on='Club', how='left')

#On my computer I had a file for league to club names ignoring that here

dfMerged = pd.read_csv('../input/FullData.csv')
sns.distplot(dfMerged['Speed'], fit=norm);
resCuml = cumfreq(dfMerged['Speed'])

xCuml = resCuml.lowerlimit + np.linspace(0, resCuml.binsize*resCuml.cumcount.size, resCuml.cumcount.size)

ax_barCuml = sns.pointplot(x=xCuml, y=resCuml.cumcount)

ax_barCuml.set_xticklabels(ax_barCuml.xaxis.get_majorticklabels(), rotation=90)

plt.show()
grpSorted = dfMerged.groupby('Club_Position')['Speed'].mean().sort_values(ascending=False)

grpSortedValues = grpSorted.index.values
f, ax = plt.subplots(figsize=(12, 10))

ax = sns.boxplot(y='Speed', x='Club_Position', data=dfMerged.sort_values(by='Speed'), order=grpSortedValues)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)

plt.show()
dfMergedforCorr = dfMerged.loc[:,~dfMerged.columns.isin(['National_Kit', 'Club_Kit', 'Contract_Expiry', 'Age', 'Weak_foot', 'GK_Positioning',

                                      'GK_Diving', 'GK_Kicking', 'GK_Handling', 'GK_Reflexes'])]
corrmat = dfMergedforCorr.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 10))

sns.barplot(data=corrmat['Speed'].reset_index().sort_values(by='Speed'), x='Speed', y='index',orient='h')
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Speed')['Speed'].index
f, ax = plt.subplots(figsize=(12, 10))

#sns.heatmap(corrmat, vmax=.8, square=True);

sns.heatmap(corrmat.loc[cols,cols], cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10});
fastestOverallClubs = dfMerged.groupby(['Club'])[['Speed', 'Acceleration', 'Agility']].mean().sort_values(by='Speed', ascending=False).head(25)

fastestOverallClubsMask = dfMerged['Club'].isin(fastestOverallClubs.index.get_level_values(0).values)



f, ax = plt.subplots(figsize=(12, 15))

#sns.barplot(x='Speed', y=fastestOverallClubs.index.get_level_values(0), data=fastestOverallClubs)

sns.stripplot(x='Speed', y='Club', data=dfMerged.loc[fastestOverallClubsMask,:], jitter=True)
offenseMask = dfMerged['Club_Position'].isin(['LW','RW','ST','RS','LS','CF','RF','LF'])

defenseMask = dfMerged['Club_Position'].isin(['RWB','LWB','RB','LB','CB','RCB','LCB'])

midfieldMask = dfMerged['Club_Position'].isin(['LM','RM','LAM','RAM','CAM','CM','LCM', 'RCM', 'RDM', 'LDM', 'CDM'])
#Now pick certain positions and filter on players in just those spots 

#to see if the top 10 teams change when you factor out certain players

fastestOffensiveClubsMed = dfMerged.loc[offenseMask,:].groupby(['Club'])[['Speed', 'Rating']].median().sort_values(by='Speed', ascending=False).head(25)

# Make the PairGrid

g = sns.PairGrid(fastestOffensiveClubsMed.reset_index().sort_values("Speed", ascending=False),

                 x_vars=fastestOffensiveClubsMed.columns[:2], y_vars=['Club'],

                 size=10, aspect=.5)



# Draw a dot plot using the stripplot function

g.map(sns.stripplot, size=10, orient="h",

      palette="GnBu_d", edgecolor="gray")



# Use semantically meaningful titles for the columns

titles = ["Speed (Median)", "Overall Rating (Median)"]



for ax, title in zip(g.axes.flat, titles):



    # Set a different title for each axes

    ax.set(title=title)



    # Make the grid horizontal instead of vertical

    ax.xaxis.grid(True)

    ax.yaxis.grid(True)



sns.despine(left=True, bottom=True)
f = dfMerged.loc[dfMerged['Club'].isin(['FC Basel', 'PSG']), ['Club','Speed']]

g = sns.FacetGrid(f, col="Club",  col_wrap=2, sharex=False, sharey=False)

g = g.map(sns.distplot, "Speed")
#Now pick certain positions and filter on players in just those spots 

#to see if the top 10 teams change when you factor out certain players

fastestOffensiveClubs = dfMerged.loc[offenseMask,:].groupby(['Club'])[['Speed', 'Rating']].mean().sort_values(by='Speed', ascending=False).head(25)
# Make the PairGrid

g = sns.PairGrid(fastestOffensiveClubs.reset_index().sort_values("Speed", ascending=False),

                 x_vars=fastestOffensiveClubs.columns[:2], y_vars=['Club'],

                 size=10, aspect=.5)



# Draw a dot plot using the stripplot function

g.map(sns.stripplot, size=10, orient="h",

      palette="GnBu_d", edgecolor="gray")



# Use semantically meaningful titles for the columns

titles = ["Speed (Mean)", "Overall Rating (Mean)"]



for ax, title in zip(g.axes.flat, titles):



    # Set a different title for each axes

    ax.set(title=title)



    # Make the grid horizontal instead of vertical

    ax.xaxis.grid(True)

    ax.yaxis.grid(True)



sns.despine(left=True, bottom=True)
fastestOffensivePlayers = dfMerged.loc[offenseMask,:].groupby(['Name', 'Club'])[['Speed', 'Rating']].mean().sort_values(by='Speed', ascending=False).head(25)

fastestOffensivePlayers['Name'] = fastestOffensivePlayers.index.get_level_values(0) + ' - ' + fastestOffensivePlayers.index.get_level_values(1)
# Make the PairGrid

g = sns.PairGrid(fastestOffensivePlayers.sort_values("Speed", ascending=False),

                 x_vars=fastestOffensivePlayers.columns[:2], y_vars=["Name"],

                 size=10, aspect=.5)



# Draw a dot plot using the stripplot function

g.map(sns.stripplot, size=10, orient="h",

      palette="GnBu_d", edgecolor="gray")



# Use semantically meaningful titles for the columns

titles = ["Speed", "Overall Rating"]



for ax, title in zip(g.axes.flat, titles):



    # Set a different title for each axes

    ax.set(title=title)



    # Make the grid horizontal instead of vertical

    ax.xaxis.grid(True)

    ax.yaxis.grid(True)



sns.despine(left=True, bottom=True)
fastVsRatingClubs = dfMerged.groupby(['Club'])[['Speed', 'Rating']].mean()
sns.jointplot(fastVsRatingClubs['Speed'], fastVsRatingClubs['Rating'], kind="hex", color="#4CB391", size=10)
#To Do - Take Offense, Mid, and Defense speeds and compare them together by club.  Look for big differences.

#To Do - Weight the values so you can find teams that are faster overall, with offense, mid or D

fastestOffensiveClubs = dfMerged.loc[offenseMask,:].groupby(['Club'])[['Speed', 'Rating']].mean().sort_values(by='Speed', ascending=False).reset_index()

fastestDefensiveClubs = dfMerged.loc[defenseMask,:].groupby(['Club'])[['Speed', 'Rating']].mean().sort_values(by='Speed', ascending=False).reset_index()

fastestMidfieldClubs = dfMerged.loc[midfieldMask,:].groupby(['Club'])[['Speed', 'Rating']].mean().sort_values(by='Speed', ascending=False).reset_index()
merged = pd.merge(fastestOffensiveClubs, fastestDefensiveClubs, on=['Club'], suffixes=('_off', '_def'))
merged = pd.merge(merged, fastestMidfieldClubs, on=['Club'])
#Melt the dataframe for speed then do a stripplot with hue as the speed_off, speed_def, etc.. colunmns and each row the team

mergedMeltSpeed = pd.melt(merged, id_vars=['Club'], value_vars=['Speed_off', 'Speed_def', 'Speed'])
def test(x):

    return x.sum()
weightedSpeed = merged[['Speed_off', 'Speed_def', 'Speed']]*(1/3)

merged['WeightedSpeed'] = weightedSpeed.apply(test, axis=1)

mergedTopFactoredSpeed = merged.sort_values(by='WeightedSpeed', ascending=False).head(25)
#Melt the dataframe for speed then do a stripplot with hue as the speed_off, speed_def, etc.. colunmns and each row the team

mergedMeltTopFactoredSpeed = pd.melt(mergedTopFactoredSpeed, id_vars=['Club'], value_vars=['Speed_off', 'Speed_def', 'Speed'])
f, ax = plt.subplots(figsize=(12, 25))

ax = sns.barplot(x='value', y='Club', data=mergedMeltTopFactoredSpeed, hue='variable', orient='h')