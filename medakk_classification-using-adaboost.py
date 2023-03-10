from string import ascii_uppercase



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier
# Load the CSV file into a pandas DataFrame

adult_df = pd.read_csv('../input/adult.csv')
adult_df.describe()
adult_df.columns
# We need to replace the incomes with an integer that denotes its class

adult_df['income'] = np.where(adult_df['income'] == '>50K', 1, 0)
# Define a function to visualize how the different columns correlate with the income

def hist_by(df, column):

    X, y = [], []

    for value in sorted(df[column].unique()):

        X.append(value)

        y.append(df[df[column] == value]['income'].mean() * 100.0)



    index = np.arange(len(X))

    width = 0.35

    plt.bar(index, y, width)

    plt.xticks(index+width/2, X, rotation=70)

    plt.yticks(np.arange(0, 100, 10))

    plt.ylim(0, 100)

    plt.xlabel(column)

    plt.ylabel('Percentage of people who\'s income is above $50K')

    plt.tight_layout()

    plt.show()
# Visualize how the sex correlates with the income

hist_by(adult_df, 'sex')
# Visualize how an educational degree correlates with the income

hist_by(adult_df, 'education')
# Visualize how marital status is related to income

hist_by(adult_df, 'marital.status')
# Visualize how occupation correlates with income

hist_by(adult_df, 'occupation')
# Visualize how workclass correlates with income

hist_by(adult_df, 'workclass')
# Get rid of the entries where the occupation or workclass is unknown

adult_df = adult_df[adult_df['occupation'] != '?']

adult_df = adult_df[adult_df['workclass'] != '?']
# Get the dummies for the columns. This is the same as LabelBinarizer in sklearn

education_dummies = pd.get_dummies(adult_df['education'])

marital_dummies = pd.get_dummies(adult_df['marital.status'])

relationship_dummies = pd.get_dummies(adult_df['relationship'])

sex_dummies = pd.get_dummies(adult_df['sex'])

occupation_dummies = pd.get_dummies(adult_df['occupation'])

native_dummies = pd.get_dummies(adult_df['native.country'])

race_dummies = pd.get_dummies(adult_df['race'])

workclass_dummies = pd.get_dummies(adult_df['workclass'])
# Example: for marital status

marital_dummies.head()
# Define a function to put the continuous values in bins

def into_bins(column, bins):

    group_names = list(ascii_uppercase[:len(bins)-1])

    binned = pd.cut(column, bins, labels=group_names)

    return binned
# Let's see how the capital loss varies

adult_df['capital.loss'].describe()
# Create a scatter plot of all the unique values in capital.loss.

# This will be helpful in visualizing how to assign bins to this feature

unique = sorted(adult_df['capital.loss'].unique())

plt.scatter(range(len(unique)), unique)

plt.ylabel('Capital Loss')

plt.tick_params(axis='x', which='both', labelbottom='off', bottom='off') # disable x ticks

plt.show()
# Create bins from -1 to 4500, with 500 values in each bin

loss_bins = into_bins(adult_df['capital.loss'], list(range(-1, 4500, 500)))

loss_dummies = pd.get_dummies(loss_bins)
# Let's see how the capital gain varies

adult_df['capital.gain'].describe()
# Create a scatter plot of all the unique values in capital.gain.

# This will be helpful in visualizing how to assign bins to this feature

unique = sorted(adult_df['capital.gain'].unique())

plt.scatter(range(len(unique)), unique)

plt.ylabel('Capital Gain')

plt.tick_params(axis='x', which='both', labelbottom='off', bottom='off') # disable x ticks

plt.show()
# Create bins from -1 to 42000, with 5000 values in each bin. And an extra one for the outlier

gain_bins = into_bins(adult_df['capital.gain'], list(range(-1, 42000, 5000)) + [100000])

gain_dummies = pd.get_dummies(gain_bins)
# Concatenate all the columns we need and the ones we generated by binning and creating dummies

X = pd.concat([adult_df[['age', 'hours.per.week']], gain_dummies, occupation_dummies, workclass_dummies, education_dummies, marital_dummies, race_dummies, sex_dummies], axis=1)

y = adult_df['income']



# Create test and train sets

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=1)
# Create a classifier and fit the data

clf = AdaBoostClassifier(random_state=1)

clf.fit(X_train, y_train)
# Find accuracy using the test set

y_pred = clf.predict(X_test)

print('Accuracy: {}'.format(accuracy_score(y_pred, y_test)))