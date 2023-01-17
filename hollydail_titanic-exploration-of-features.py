import numpy as np

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns
# Read in both datasets

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



# Print out some basic information about the datasets before we focus

# entirely on the training dataset

print(f'Training data has {train_data.shape[0]} rows, {train_data.shape[1]} columns')

print(f'Test data has {test_data.shape[0]} rows, {test_data.shape[1]} columns')



cols_both = [col for col in train_data.columns if col in test_data.columns]

cols_train_only = [col for col in train_data.columns if col not in test_data.columns]

cols_test_only = [col for col in test_data.columns if col not in train_data.columns]



print(f'Both datasets have columns {cols_both}') if len(cols_both) else None

print(f'Training has columns {cols_train_only}, test does not') if len(cols_train_only) else None

print(f'Test has columns {cols_test_only}, train does not') if len(cols_test_only) else None
train_data.profile_report(title='Training data: initial profiling report')
train_data.head()
def explore_generalsurvival(df_train, min_samples=10):

    """Show mean survival for data grouped by gender, embarkation point,

    and Pclass.  Only show subsets with at least <min_samples> samples.

    """

    

    fig, axes = plt.subplots(1, 2, figsize=[14, 5])

    for gender, ax in zip(['male', 'female'], axes):

        ind = (train_data['Sex'] == gender)

        mean_surv = df_train[['Embarked', 'Pclass', 'Survived', 'Sex']][ind]

        mean_surv_vals = (mean_surv.groupby(['Embarked', 'Pclass'])['Survived']

                                   .mean()

                                   .unstack('Pclass'))

        mask = (mean_surv.groupby(['Embarked', 'Pclass'])['Survived']

                         .count()

                         .unstack('Pclass'))

        mask = mask < min_samples

        sns.heatmap(mean_surv_vals, ax=ax, annot=mean_surv_vals, mask=mask,

                    cbar_kws={'label': 'Mean survival'})

        ax.set_title(gender)

        

explore_generalsurvival(train_data)
def explore_familysamplesize(df, sibsp='SibSp', parch='Parch'):

    """Show the sample size once we group by gender, Parch, and SibSp.  No cases are

    masked in this plot.  sibsp and parch variable names are passed in as we use the

    same function before and after we collapse classes.

    """

    fig, axes = plt.subplots(1, 2, figsize=[15, 5])

    for gender, ax in zip(['male', 'female'], axes):

        mean_surv = df[[sibsp, parch, 'Survived', 'Sex']][train_data['Sex'] == gender]

        mean_surv_cnts = (mean_surv.groupby([sibsp, parch])['Survived']

                                   .count()

                                   .unstack(sibsp))

        sns.heatmap(mean_surv_cnts, ax=ax, annot=mean_surv_cnts,

                    cbar_kws={'label': 'Number of cases'})

        ax.set_title(gender)

        

explore_familysamplesize(train_data)
def explore_familysurvival(df, min_samples=6, sibsp='SibSp', parch='Parch'):

    """Show mean survival for different parent child and sibling classes, masking out

    any classes with less than <min_samples> samples.  The variable name is passed in

    for sibsp and parch because we use the same function to plot before and after we

    collapse classes."""

    

    fig, axes = plt.subplots(1, 2, figsize=[15, 5])

    for gender, ax in zip(['male', 'female'], axes):

        mean_surv = df[[sibsp, parch, 'Survived', 'Sex']][train_data['Sex'] == gender]

        mean_surv_vals = (mean_surv.groupby([sibsp, parch])['Survived']

                                   .mean()

                                   .unstack(sibsp))

        mask = (mean_surv.groupby([sibsp, parch])['Survived']

                         .count()

                         .unstack(sibsp))

        mask = mask < min_samples

        sns.heatmap(mean_surv_vals, ax=ax, annot=mean_surv_vals, mask=mask,

                    cbar_kws={'label': 'Mean survival'})

        ax.set_title(gender)

        

explore_familysurvival(train_data)
def add_binary_family_features(df_train):

    """Create new variables has_parch and has_sibsp where values of Parch and SibSp

    greater than 1 are collapsed to 1.

    """



    new_names = {'Parch': 'has_parch',

                 'SibSp': 'has_sibsp'}

    for var in ['Parch', 'SibSp']:

        new_var = df_train[var].copy(deep=True)

        new_var[new_var > 1] = 1

        df_train[new_names[var]] = new_var

    return df_train



train_data = add_binary_family_features(train_data)
explore_familysamplesize(train_data, sibsp='has_sibsp', parch='has_parch')
explore_familysurvival(train_data, min_samples=6, sibsp='has_sibsp', parch='has_parch')
g = sns.catplot(x="Pclass", y="Age", hue="Survived", col="Sex",

                data=train_data[['Pclass', 'Survived', 'Sex', 'Age']],

                kind="violin", split=True, cut=0, scale='count')
train_data[train_data['Fare'] == 0]
train_data[train_data['Fare'] == 0].groupby('Pclass')['Fare'].count()
# First we look at the distribution in fares for all three classes

for pclass in [1, 2, 3]:

    df_ind = (train_data['Pclass'] == pclass)

    sns.distplot(train_data['Fare'][df_ind], hist=False, label=pclass)
# Next we see if removing the passengers who paid zero and looking only at

# the low end of the distribution clarifies

df_sub = train_data[train_data['Fare'] != 0]

fig, axes = plt.subplots(nrows=3, figsize=(7, 7), sharex=True)

for ind, pclass in enumerate([1, 2, 3]):

    df_ind = (df_sub['Pclass'] == pclass)

    sns.distplot(df_sub['Fare'][df_ind], hist=True, 

                 bins=np.arange(0, 110, 2), label=pclass,

                 ax=axes[ind], )

    _ = axes[ind].set_xlim([-10, 100])

    axes[ind].set_ylabel(f'pclass={pclass}')
g = sns.catplot(x="Pclass", y="Fare", hue="Survived", col="Sex",

                data=train_data[['Fare', 'Pclass', 'Survived', 'Sex']],

                kind="violin", split=True)