import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%pylab inline
df = pd.read_csv('../input/aegypti_albopictus.csv')

df.info()
# I see X and Y, let's plot them and see what they look like

plt.plot(df['X'], df['Y'], '.')
# Looks like the world map, a bit skewed though

# I want to see what STATUS looks like on this plot.

T, E = df.loc[df.STATUS == 'T'], df.loc[df.STATUS == 'E']

plt.plot(T['X'], T['Y'], '.', color='r')

plt.plot(E['X'], E['Y'], '.', color='b')
# This is very different from the world map, meaning most of the STATUS values

# are missing, perhaps

# Let's check

df.STATUS.replace(np.nan, 'NA').value_counts().plot(kind='bar')


# WOW!, the measured values actually are disappearingly small in quantity

# Let's see what the other columns look like

df.head()
# Are there vectors other than Aedes aegypti?

df.VECTOR.value_counts().plot(kind='bar')
# OCCURANCE_ID seems to be an index column

(df.OCCURRENCE_ID - df.OCCURRENCE_ID.shift()).unique()
# yes, it's a index value, so we can safely ignore that (perhaps?)

df.SOURCE_TYPE.value_counts().plot(kind='bar')
# we need to see where the data comes from, so we plot two things

df.LOCATION_TYPE.value_counts().plot(kind='bar')
#plt.figure(figsize=(20, 5))

df.COUNTRY.value_counts().plot(kind='bar', figsize=(20, 5))
# A lot of this is from Taiwan. I want to take a look at the X, Y world map again

# but this time with instance densities

sns.jointplot('X', 'Y', data=df, kind='kde')
# WOW. Looks like the only place the data is from is Taiwan

# This is bound to be a pretty important feature, though it will have little

# meaning if we try to predict a case's status. This is what bias in the data looks like.

# How about the distribution in years? That might show us something interesting

df.YEAR.value_counts().plot(kind='bar', figsize=(10, 5))
# So this is mostly recent data. Fair enough, back then data collection would

# have been incredibly expensive too.



# I still cannot make heads or tails of the GAUL_AD0 column. Let's see

# what it actually looks like

df.GAUL_AD0.hist()
df.GAUL_AD0.head()
df.GAUL_AD0.unique()
# Still no idea what this is. I must have missed something in the description.

# Another column which we have not yet touched is POLYGON_ADMIN

df.POLYGON_ADMIN.unique()
# This looks a lot like the LOCATION_TYPE. I'm leaving this out for someone else to make

# sense of. Now let's see the correlations.

# The -999 looks like a NA value, but I'm going to leave that in to make it easier for

# plotting later on

sns.heatmap(df.corr(), annot=True)
# This shows a lot of correlation, but I suppose that's simply because of the imbalance

# in data collection towards Taiwan, so it's probably not refelctive of anything.

# How about we visualize the XY again with different things as hue?

sns.lmplot('X', 'Y', hue='LOCATION_TYPE', data=df, fit_reg=False, aspect=1.5)


sns.lmplot('X', 'Y', hue='SOURCE_TYPE', data=df, fit_reg=False, aspect=1.5)
sns.lmplot('X', 'Y', hue='STATUS', data=df, fit_reg=False,aspect=1.5)
sns.lmplot('X', 'Y', hue='VECTOR', data=df, fit_reg=False, aspect=1.5)
sns.lmplot('X', 'Y', hue='POLYGON_ADMIN', data=df, fit_reg=False, aspect=1.5)
# A lot of this data is from very different places in the world. For instance I cannot find

# Russia out there. Also, India has moved up in the world, and Africa is quiet literally

# the dark continent.



# We'll be leaving it at that sicne this is a Scout Script and names must be obeyed.

# Feedback is welcome.