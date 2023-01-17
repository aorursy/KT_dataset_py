import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import math
%matplotlib inline
# Read the datasets and merge together
# Use NaN values to store values for Survived in the case of test data

titanic_train = pd.read_csv('../input/train.csv')
titanic_train['IsTrain'] = True
titanic_test = pd.read_csv('../input/test.csv')
titanic_test.insert(1, 'Survived', np.nan)
titanic_test['IsTrain'] = False
titanic = titanic_train.append(titanic_test)
del titanic_train; del titanic_test
titanic.info()
titanic.head(5)
# I could not find an equivalent of the missmap function in Python.
# The following function was obtained from here
# http://stackoverflow.com/questions/21925114/is-there-an-implementation-of-missingmaps-in-pythons-ecosystem
# All credit to Tom Augspurger
# Minimal changes were made to make it Python 3 compatible

from matplotlib import collections as collections
from matplotlib.patches import Rectangle
from itertools import cycle

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
    ax.tick_params(axis='both', which='both', bottom='off', left='off',
                   labelleft='off')
    ax.grid(False)

    if title:
        ax.set_title(title)
    return ax
colours = {True: "#CC3399", False: "#6699FF"}
ax = missmap(titanic, colors = colours)
plt.show(ax)
age_hist = titanic['Age'].plot(kind = "hist", grid = True, title = 'Age Distribution on Titanic', bins=30, facecolor = '#6699FF', edgecolor = 'b')
age_hist.set(title = "Age Distribution on Titanic", ylabel = 'Count', xlabel = 'Age')
age_hist.set_yticks(np.arange(0, 120, 30))
sum(titanic['Age'].isnull())
mean_age = np.mean(titanic['Age'])
print(mean_age)
titanic['Age'].fillna(mean_age, inplace=True)
age_hist = titanic['Age'].plot(kind = "hist", grid = True, title = 'Age Distribution on Titanic after filling NaN values', bins=30, facecolor = '#6699FF', edgecolor = 'b')
titanic['Survived'].value_counts()
colours = ["#CC3399", "#6699FF"]
survived_sex_plot = pd.crosstab(titanic['Survived'], titanic['Sex']).plot(kind='bar', stacked=True, color=colours)
survived_sex_plot.set(title = "Survived by Sex", ylabel = 'Count', xlabel = 'Survived')
survived_sex_plot.set_xticklabels(["No", "Yes"])
pd.crosstab(titanic['Survived'], titanic['Pclass'])
print(pd.crosstab(titanic['Survived'], titanic['Pclass']))

colours = ["#CC3399", "#6699FF", "#66CC33"]
survived_class_plot = pd.crosstab(titanic['Survived'], titanic['Pclass']).plot(kind='bar', stacked=True, color=colours)
survived_class_plot.set(title = "Survived by Class", ylabel = 'Count', xlabel = 'Survived')
survived_class_plot.set_xticklabels(["No", "Yes"])
titanic['AgeF'] = "Infant"
titanic.loc[titanic['Age'] >= 2, 'AgeF'] = 'Child'
titanic.loc[titanic['Age'] >= 18, 'AgeF'] = 'Adult'
titanic.loc[titanic['Age'] >= 50, 'AgeF'] = 'Elderly'

colours = ["#CC3399", "#66CCCC", "#CC99FF", "#66CC33"]
survived_agef_plot = pd.crosstab(titanic['Survived'], titanic['AgeF']).plot(kind='bar', stacked=True, color=colours)
survived_agef_plot.set(title = "Survived by Age Factor", ylabel = 'Count', xlabel = 'Survived')
survived_agef_plot.set_xticklabels(["No", "Yes"])
age_hist = titanic['Fare'].plot(kind = "hist", grid = True, title = 'Fare Distribution on Titanic', bins=40, facecolor = '#6699FF', edgecolor = 'b')
print("Number passengers with no fare information: " + str(sum(titanic['Fare'].isnull())))
titanic['Fare'].fillna(np.mean(titanic['Fare']), inplace=True)
# Define a new column for sex assigning 1 for female, 0 for male
titanic['SexN'] = 1
titanic.loc[titanic['Sex'] == 'male', 'SexN'] = 0

M = titanic.corr()
del M['PassengerId']; del M['IsTrain'] # Delete columns
M = M.drop('PassengerId'); M = M.drop('IsTrain') # Delete rows
print(M)
# From here: http://seaborn.pydata.org/examples/many_pairwise_correlations.html
# Generate a mask for the upper triangle
mask = np.zeros_like(M, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(M, mask=mask, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
X = titanic.loc[titanic['IsTrain'], ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'SexN']]
X = sm.add_constant(X)     # By default statsmodels does not include a contant
Y = titanic.loc[titanic['IsTrain'], ['Survived']]

# Note the swap of X and Y (statsmodels uses endog (Y/dependent) and exog (X/independent))
model = sm.GLM(Y, X, family = sm.families.Binomial())
results = model.fit()
# Statsmodels gives R-like statistical output
print(results.summary())
# Get a random selection of indices for the training set
smp_size = math.floor(0.75 * 891)
np.random.seed(123)
train_ind = np.sort(np.random.choice(891, size=smp_size, replace=False))
print(train_ind.shape)
titanic_numeric = titanic.loc[titanic['IsTrain'], ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'SexN']].dropna()
training = titanic_numeric.iloc[train_ind, :]
test = titanic_numeric.drop(train_ind)
print(X.shape, train_ind.shape, training.shape, test.shape)
#training.loc[:, ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'SexN']].head(20)
#training.loc[:, ['Survived']].head()
# Note the swap of X and Y (statsmodels uses endog (Y/dependent) and exog (X/independent))
training_x = training.drop('Survived', axis=1)
training_y = training.loc[:, ['Survived']]
logit_training = sm.GLM(training_y, sm.add_constant(training_x), family = sm.families.Binomial())   # <-----using a constant adds an extra co-efficient we need to add here
result = logit_training.fit()
print(result.summary())
print(train_ind.shape, training.shape, test.shape)
test.head(20)
# Generate predicted values using the logistic model
test_y = test.loc[:, ['Survived']]
test_x = test.drop('Survived', axis=1)
y_pred = result.predict(sm.add_constant(test_x))
y_pred = y_pred.apply(lambda x: 1 if x > 0.5 else 0)
# Look at the output
actual_vs_pred = test_y
actual_vs_pred['Predicted'] = y_pred
actual_vs_pred['Match'] = True
actual_vs_pred.loc[actual_vs_pred['Survived'] != actual_vs_pred['Predicted'], 'Match'] = False
actual_vs_pred.head(20)
print('Accuracy: ' + str(np.mean(actual_vs_pred['Match'])))