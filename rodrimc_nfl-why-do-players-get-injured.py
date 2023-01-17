import numpy as np

import pandas as pd

import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.stats.multicomp import MultiComparison

from statsmodels.stats.weightstats import ttest_ind

import scipy.stats as stats

import itertools
df_injuries = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')

df_playlist = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')

df_player_tracks = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')
print('Number of injuries: %d' % len(df_injuries))

print('Number of plays: %d' % len(df_playlist))

print('Number of player tracks: %d' % len(df_player_tracks))
df_injuries.info(verbose=True)

print('\n----------------------------------------------\n')

df_playlist.info(verbose=True)

print('\n----------------------------------------------\n')

df_player_tracks.info(verbose=True)
# Are there any duplicated IDs?

df_playlist['PlayKey'].duplicated().value_counts()
# Are there any null values that need to be dropped?

df_playlist[['PlayerKey', 'GameID', 'PlayKey']].isna().sum()
# Join playlist dataset and injury dataset using the keys from the playlist dataset

df_playlist_ext = pd.merge(df_playlist, df_injuries, how='left', on=['PlayerKey', 'GameID', 'PlayKey'])

#df_playlist_ext.info()
# Create a new Boolean column indicating whether the player got injuried in that play.

df_playlist_ext['IsInjured'] = (df_playlist_ext['DM_M1']==1) | (df_playlist_ext['DM_M7']==1) | (df_playlist_ext['DM_M28']==1) | (df_playlist_ext['DM_M42']==1)

df_playlist_ext['IsInjured'].value_counts()
# compute the injury probability for synthetic and natural turf

p_injury = df_playlist_ext[['FieldType', 'IsInjured']].groupby('FieldType').mean()['IsInjured']

p_injury
p_null = df_playlist_ext[['FieldType', 'IsInjured']][df_playlist_ext['FieldType'] == 'Natural'].mean()[0]



n_plays_natural = df_playlist_ext.groupby('FieldType').size()[0]

n_plays_synthetic = df_playlist_ext.groupby('FieldType').size()[0]



div = np.sqrt(p_null * (1-p_null) * (1/n_plays_natural + 1/n_plays_synthetic))



#  compute z-score and p-value

z = (p_injury[1] - p_injury[0]) / div



print('The z-score is: {}'.format(z))

print('The p-value is: {}'.format(1-stats.norm.cdf(z)))
df_injuries_ext = pd.merge(df_injuries, df_playlist, how='inner', on=['PlayerKey', 'GameID', 'PlayKey'])



# Create a numerical variable from the duration

df_injuries_ext['Duration1'] = np.where(df_injuries_ext['DM_M1']>=1, 1, 0)

df_injuries_ext['Duration7'] = np.where(df_injuries_ext['DM_M7']>=1, 7, 0)

df_injuries_ext['Duration28'] = np.where(df_injuries_ext['DM_M28']>=1, 28, 0)

df_injuries_ext['Duration42'] = np.where(df_injuries_ext['DM_M42']>=1, 42, 0)

df_injuries_ext['Duration'] = df_injuries_ext[['Duration1', 'Duration7', 'Duration28', 'Duration42']].max(axis=1)

df_injuries_ext.drop(columns=['Duration1', 'Duration7', 'Duration28', 'Duration42'], inplace=True)

df_injuries_ext.head()
df_corr = df_injuries_ext[['PlayerDay', 'PlayerGame', 'Temperature', 'PlayerGamePlay', 'Duration']].corr()



fig = plt.figure(figsize=(10,7))

ax = sns.heatmap(df_corr, annot=True, cmap=sns.diverging_palette(240, 10, n=9))

ax.set_title("Pearson Correlation between Variables");

plt.show()
# The following function was originally seen here: https://www.kaggle.com/phaethonprime/eda-and-logistic-regression



def cramers_corrected_stat(confusion_matrix):

    """ calculate Cramers V statistic for categorical-categorical association.

        uses correction from Bergsma and Wicher, 

        Journal of the Korean Statistical Society 42 (2013): 323-328

    """

    chi2 = stats.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
cols = ['Surface', 'BodyPart', 'RosterPosition', 'StadiumType', 'FieldType', 'Weather', 'PlayType', 'Position', 'PositionGroup']

corr = np.zeros((len(cols),len(cols)))



# Apply previous function to calculate Cramer's V for each pair

for col1, col2 in itertools.combinations(cols, 2):

    idx1, idx2 = cols.index(col1), cols.index(col2)

    corr[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df_injuries_ext[col1], df_injuries_ext[col2]))

    corr[idx2, idx1] = corr[idx1, idx2]
df_corr = pd.DataFrame(corr, index=cols, columns=cols)



# Plot the correlation matrix

fig = plt.figure(figsize=(10,7))

ax = sns.heatmap(df_corr, annot=True, cmap=sns.diverging_palette(240, 10, n=9))

ax.set_title("Cramer V Correlation between Variables");

plt.show()
# Plot the distribution of injuries per surface

sns.set(style="darkgrid")

ax = sns.countplot(x="Surface", data=df_injuries)

ax.set_title("Injury count on both surfaces");

plt.show()
sns.set(style="darkgrid")

ax = sns.countplot(x="Duration", data=df_injuries_ext)

plt.show()
crosstab = pd.crosstab(df_injuries_ext['Surface'], df_injuries_ext['Duration'])

crosstab
ret = stats.chi2_contingency(crosstab)

print('Pearson Chi-square = %.3f' % ret[0])

print('p-value = %.3f > 0.05' % ret[1])

print('We cannot reject H0. There is no relationship between Surface and Duration')
# Show each observation with a scatterplot

sns.set(style="darkgrid")

sns.stripplot(x="Duration", y="Surface", order=['Natural','Synthetic'], data=df_injuries_ext, alpha=.50)

sns.pointplot(x="Duration", y="Surface", order=['Natural','Synthetic'], data=df_injuries_ext, palette="dark", markers="d")

plt.show()
# Get each position independently

d_natural = df_injuries_ext['Duration'][df_injuries_ext['Surface']=='Natural']

d_synthetic = df_injuries_ext['Duration'][df_injuries_ext['Surface']=='Synthetic']



# Example of how it would be done for only two groups

ret = stats.ttest_ind(d_natural, d_synthetic)

print('Independent t-test = %.3f' % ret[0])

print('p-value = %.3f > 0.05' % ret[1])

print('We cannot reject H0. There is no relationship between Surface and Duration')
df_injuries_ext = df_injuries_ext.replace('Mostly sunny', 'Sunny')

df_injuries_ext = df_injuries_ext.replace('Mostly Sunny', 'Sunny')

df_injuries_ext = df_injuries_ext.replace('Clear Skies', 'Clear')

df_injuries_ext = df_injuries_ext.replace('Controlled Climate', 'Clear')

df_injuries_ext = df_injuries_ext.replace('Clear and warm', 'Clear')

df_injuries_ext = df_injuries_ext.replace('Fair', 'Clear')

df_injuries_ext = df_injuries_ext.replace('Clear skies', 'Clear')

df_injuries_ext = df_injuries_ext.replace('Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.', 'Rain')

df_injuries_ext = df_injuries_ext.replace('Coudy', 'Cloudy')

df_injuries_ext = df_injuries_ext.replace('Mostly cloudy', 'Cloudy')

df_injuries_ext = df_injuries_ext.replace('Cloudy and Cool', 'Cloudy')

df_injuries_ext = df_injuries_ext.replace('Sun & clouds', 'Partly Cloudy')

df_injuries_ext = df_injuries_ext.replace('Light Rain', 'Rain')

df_injuries_ext = df_injuries_ext.replace('Rain shower', 'Rain')

df_injuries_ext = df_injuries_ext.replace('Cloudy, 50% change of rain', 'Rain')

df_injuries_ext = df_injuries_ext.replace('Indoors', 'Indoor')
sns.set(style="darkgrid")

ax = sns.countplot(x="Weather", data=df_injuries_ext)

ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

plt.show()
crosstab = pd.crosstab(df_injuries_ext['Surface'], df_injuries_ext['Weather'])

crosstab
ret = stats.chi2_contingency(crosstab)

print('Pearson Chi-square = %.3f' % ret[0])

print('p-value = %.3f < 0.05' % ret[1])

print('We can reject H0. There is a relationship between Surface and Weather')
df_aux = df_injuries_ext[df_injuries_ext['Weather'] != 'Indoor']

crosstab = pd.crosstab(df_aux['Surface'], df_aux['Weather'])

ret = stats.chi2_contingency(crosstab)

print('Pearson Chi-square = %.3f' % ret[0])

print('p-value = %.3f < 0.05' % ret[1])

print('We cannot reject H0.')
sns.set(style="darkgrid")

ax = sns.countplot(x="BodyPart", data=df_injuries_ext)

plt.show()
crosstab = pd.crosstab(df_injuries_ext['Surface'], df_injuries_ext['BodyPart'])

crosstab
ret = stats.chi2_contingency(crosstab)

print('Pearson Chi-square = %.3f' % ret[0])

print('p-value = %.3f > 0.05' % ret[1])

print('We cannot reject H0. There is no relationship between Surface and BodyPart')
crosstab_heel_toe = crosstab[['Ankle', 'Foot']]

ret = stats.fisher_exact(crosstab_heel_toe)

print('Pearson Chi-square = %.3f' % ret[0])

print('p-value = %.3f > 0.05' % ret[1])

print('We cannot reject H0. There is no relationship between Surface and BodyPart for the specific case of heels and toes')
sns.set(style="darkgrid")

ax = sns.countplot(x="PositionGroup", data=df_injuries_ext)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.show()
crosstab = pd.crosstab(df_injuries_ext['Surface'], df_injuries_ext['PositionGroup'])

crosstab
ret = stats.chi2_contingency(crosstab)

print('Pearson Chi-square = %.3f' % ret[0])

print('p-value = %.3f > 0.05' % ret[1])

print('We cannot reject H0. There is no relationship between Surface and BodyPart')
# We have some values -999 which are clear NaNs. Replace them

df_aux = df_injuries_ext[df_injuries_ext['Temperature'] != -999]
sns.set(style="darkgrid")

sns.stripplot(x="Temperature", y="Surface", data=df_aux, alpha=.70)

sns.pointplot(x="Temperature", y="Surface", data=df_aux, palette="dark", markers="d")

plt.show()
temp_natural = df_aux['Temperature'][df_aux['Surface']=='Natural']

temp_synth = df_aux['Temperature'][df_aux['Surface']=='Synthetic']



ret = stats.ttest_ind(temp_natural, temp_synth)

print('Independent t-test = %.3f' % ret[0])

print('p-value = %.3f > 0.05' % ret[1])

print('We cannot reject H0. There is no relationship between Surface and Temperature')
df_tracks_ext = pd.merge(df_player_tracks, df_injuries, on='PlayKey', how='inner')

df_tracks_ext.head()
# Scatter plot for one example play (using 39678-2-1, 47813-8-19 and 31070-3-7 as examples)

df_track = df_tracks_ext[df_tracks_ext['PlayKey']=='47813-8-19']

turf_img = plt.imread('../input/customimgs/nfl-turf.jpg')



fig = plt.figure(figsize=(10,6))

implot = plt.imshow(turf_img, extent=[-10, 120, -10, 54])

sns.scatterplot(x='x', y='y', hue='PlayKey', data=df_track)

plt.show()
# And scatter plot for every play that resulted into an injury

fig = plt.figure(figsize=(10,6))

implot = plt.imshow(turf_img, extent=[-10, 120, -10, 54])

sns.scatterplot(x='x', y='y', hue='PlayKey', data=df_tracks_ext, legend=False)

plt.show()
# Obtain average speed for each play

df_speed = df_tracks_ext.groupby('PlayKey')['s'].mean()

df_speed_ext = pd.merge(df_speed, df_tracks_ext[['PlayKey', 'Surface']], on='PlayKey', how='inner')

df_speed_ext = df_speed_ext.drop_duplicates().reset_index()
# Sort the dataframe by surface

df_natural = df_speed_ext.loc[df_speed_ext['Surface'] == 'Natural']

df_synthetic = df_speed_ext.loc[df_speed_ext['Surface'] == 'Synthetic']



# Plot distribution of velocities for both groups

sns.distplot(df_natural['s'], bins=10)

sns.distplot(df_synthetic['s'], bins=10)

plt.show()
sns.set(style="darkgrid")

sns.stripplot(x="s", y="Surface", data=df_speed_ext, alpha=.70)

sns.pointplot(x="s", y="Surface", data=df_speed_ext, palette="dark", markers="d")

plt.show()
df_aux = pd.concat([df_speed_ext, df_injuries_ext['Duration']], axis=1)

corr = df_aux['s'].corr(df_aux['Duration'])

print('Pearson correlation: ', corr)

print('Degrees of freedom: ', (len(df_speed_ext.index)-2) )
area = 4 * df_track['s']**2



fig = plt.figure(figsize=(8, 20))



# Plotting direction and orientation in polar coordinates, for one specific play

ax = plt.subplot(1, 2, 1, projection='polar')

plt.scatter(df_track['dir'], df_track['s'], s=area, alpha=0.50, color='orange')

ax.set_title('Speed and direction.')

ax = plt.subplot(1, 2, 2, projection='polar')

plt.scatter(df_track['o'], df_track['s'], s=area, alpha=0.50, color='green')

ax.set_title('Speed and orientation.')

plt.show()
# To get the amount of rotation from orientation we substract the current row with the previous row

df_tracks_ext['o_diff'] = df_tracks_ext['o'].diff()

df_orientation = df_tracks_ext.groupby('PlayKey')['o_diff'].sum().abs()
# To get the amount of rotation from direction we substract the current row with the previous row

df_tracks_ext['dir_diff'] = df_tracks_ext['dir'].diff()

df_direction = df_tracks_ext.groupby('PlayKey')['dir_diff'].sum().abs()
# Merge the dataframes

df_orientation_ext = pd.merge(df_orientation, df_tracks_ext[['PlayKey', 'Surface']], on='PlayKey', how='inner')

df_orientation_ext = df_orientation_ext.drop_duplicates().reset_index()

df_direction_ext = pd.merge(df_direction, df_tracks_ext[['PlayKey', 'Surface']], on='PlayKey', how='inner')

df_direction_ext = df_direction_ext.drop_duplicates().reset_index()
df_direction_ext['dir_diff'].corr(df_orientation_ext['o_diff'])
# Plot conditional distribution

sns.set(style="darkgrid")

ax = sns.stripplot(x="dir_diff", y="Surface", data=df_direction_ext, alpha=.70)

sns.pointplot(x="dir_diff", y="Surface", data=df_direction_ext, palette="dark", markers="d")

ax.set_title('Conditional distribution, direction')

plt.show()
temp_natural = df_direction_ext['dir_diff'][df_direction_ext['Surface']=='Natural']

temp_synth = df_direction_ext['dir_diff'][df_direction_ext['Surface']=='Synthetic']



# Student's t-test

ret = stats.ttest_ind(temp_natural, temp_synth)

print('Independent t-test = %.3f' % ret[0])

print('p-value = %.3f > 0.05' % ret[1])

print('We cannot reject H0. There is no relationship between Surface and Temperature')
df_aux = pd.concat([df_direction_ext, df_injuries_ext['Duration']], axis=1)

corr = df_aux['dir_diff'].corr(df_aux['Duration'])

print('Pearson correlation: ', corr)

print('Degrees of freedom: ', (len(df_speed_ext.index)-2) )