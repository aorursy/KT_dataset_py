import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting parameters
%matplotlib inline
sns.set(style='whitegrid')   # Use seaborn default syling
df = pd.read_csv('../input/train.csv')
df['IsTrain'] = True

df_test = pd.read_csv('../input/test.csv')
df_test['IsTrain'] = False
df_test.insert(1, 'Survived', np.nan) #  Add a 'Survived' column with all values set to NaN

print('DataFrame Shape: ' + str(df.shape) + '\n')
# First get a preview of the data
df.head(10)
# Examine column labels
print(df.columns)
print(df.info())
# The following function was obtained from here
# http://stackoverflow.com/questions/21925114/is-there-an-implementation-of-missingmaps-in-pythons-ecosystem
# All credit to Tom Augspurger
# Minimal changes were made to make it Python 3 compatible

from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import collections as collections
from matplotlib.patches import Rectangle


def missmap(df, ax=None, colors=None, aspect=4, sort='descending',
            title=None, **kwargs):
    """
    Plot the missing values of df.

    Parameters
    ----------
    df : pandas DataFrame
    ax : matplotlib axes
        if None then a new figure and axes will be created
    colors : dict
        dict with {True: c1, False: c2} where the values are
        matplotlib colors.
    aspect : int
        the width to height ratio for each rectangle.
    sort : one of {'descending', 'ascending', None}
    title : str
    kwargs : dict
        matplotlib.axes.bar kwargs

    Returns
    -------
    ax : matplotlib axes

    """

    if ax is None:
        fig, ax = plt.subplots()

    # setup the axes
    dfn = pd.isnull(df)

    if sort in ('ascending', 'descending'):
        counts = dfn.sum()
        sort_dict = {'ascending': True, 'descending': False}
        counts = counts.sort_values(ascending=sort_dict[sort])
        dfn = dfn[counts.index]

    # Up to here
    ny = len(df)
    nx = len(df.columns)
    # each column is a stacked bar made up of ny patches.
    xgrid = np.tile(np.arange(nx), (ny, 1)).T
    ygrid = np.tile(np.arange(ny), (nx, 1))
    # xys is the lower left corner of each patch
    xys = (zip(x, y) for x, y in zip(xgrid, ygrid))

    if colors is None:
        colors = {True: '#EAF205', False: 'k'}

    widths = cycle([aspect])
    heights = cycle([1])

    for xy, width, height, col in zip(xys, widths, heights, dfn.columns):
        color_array = dfn[col].map(colors)

        rects = [Rectangle(xyc, width, height, **kwargs)
                 for xyc, c in zip(xy, color_array)]

        p_coll = collections.PatchCollection(rects, color=color_array,
                                             edgecolor=color_array, **kwargs)
        ax.add_collection(p_coll, autolim=False)

    # post plot aesthetics
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    ax.set_xticks(.5 + np.arange(nx))  # center the ticks
    ax.set_xticklabels(dfn.columns)
    for t in ax.get_xticklabels():
        t.set_rotation(90)

    # remove tick lines
    ax.tick_params(axis='both', which='both', bottom=False, left=False,labelleft=False)
    ax.grid(False)

    if title:
        ax.set_title(title)
    return ax

# Examine the missing values
fig = plt.figure(figsize=(12.0, 8.0))
ax = fig.subplots()
ax = missmap(df, title='Missing Values', ax=ax)
plt.show(ax)
# Find passengers with missing Embarked details
df[df.Embarked.isnull()]
# Find duplicate ticket numbers
duplicate_index = df.duplicated(subset='Ticket', keep=False)
duplicates = df[duplicate_index].sort_values('Ticket')
print("Total number of duplicate ticket values: " + str(len(duplicates)))
duplicates.head(30)
# Quality check categorical variables to look for missing values or categories out of the ordinary
# There are too many possibilitities for Ticket and Cabin to look at these sensibly

print('Gender values: ')
print(df.Sex.value_counts())

print('\nEmabarked Values: ')
print(df.Embarked.value_counts())

print('\nPclass: ')
print(df.Pclass.value_counts())

print('\nSibSp: ')
print(df.SibSp.value_counts())

print('\nParch: ')
print(df.Parch.value_counts())
# First look at summary statistics for numerical data
df.describe()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
sns.distplot(df.Age[df.Age.isnull() == False], ax=ax1)
ax1.set_title("Distribution of age values")
sns.distplot(df.Fare[df.Fare.isnull() == False], ax=ax2)
ax2.set_title("Distribution of fare values")
df[df.Fare > 300]
# Calculate correlation values between our numerical values
# To include gender details we need to include a new column, assigning 1 for male, 0 for female
# Necessary as pandas correlattion method does not accomodate categorical variables
# We'll also remove PassengerId and IsTrain from our correlation.

df.Sex = df.Sex.astype('category')
df['SexN'] = df.Sex.cat.codes

df_corr = df.corr()
df_corr.drop(labels=['PassengerId', 'IsTrain'], axis=0, inplace=True) # Drop from rows
df_corr.drop(labels=['PassengerId', 'IsTrain'], axis=1, inplace=True) # Drop from columns
df_corr
# To make it easier to see the relationships we'll make it into a heatmap

# Generate a mask for the upper triangle so we don't see repeated correlations
mask = np.zeros_like(df_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio. Fix the max and min values for even comparison.
sns.heatmap(df_corr, vmin=-1, vmax=1, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
fig = sns.factorplot(x="Sex", y="Age", col="Pclass", kind="swarm", ci=None, data=df)
fig = sns.factorplot(x="Sex", y="Survived", col="Pclass", kind="bar", ci=None, data=df)
# First some data supplementation - create age brackets for easier visualisation
df['AgeF'] = np.nan
df.loc[df.Age >= 0, "AgeF"] = "Infant"
df.loc[df.Age >= 2, "AgeF"] = "Young child"
df.loc[df.Age >=5, 'AgeF'] = 'Child'
df.loc[df.Age >= 16, 'AgeF'] = 'Adult'
df.loc[df.Age >= 50, 'AgeF'] = 'Elderly'

# Set the ordering of columns and then plot
col_order = ['Infant', 'Young child', 'Child', 'Adult', 'Elderly']
fig = sns.factorplot(x="Sex", y="Age", col="AgeF", col_order=col_order, kind="swarm", ci=None, data=df)
fig = sns.factorplot(x="Sex", y="Survived", col="AgeF", col_order=col_order, ci=None, kind="bar", data=df)
bins = np.arange(0, 6, 1)
g = sns.FacetGrid(df, col="AgeF", row="Sex", hue="Sex", col_order=col_order, legend_out=True, ylim=(0, 500))
g = g.map(plt.hist, "Parch", bins=bins)
fig = sns.factorplot(x="Parch", y="Survived", col="AgeF", hue="Sex", col_order=col_order, ci=None, kind="bar", data=df)
# Find passengers under the age of 16 travelling unaccompanied by adults
df.loc[(df.Age < 16) & (df.Parch == 0)]