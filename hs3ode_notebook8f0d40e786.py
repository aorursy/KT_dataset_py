import pandas as pd

import seaborn as sns

import string
def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if string.find(big_string, substring) != -1:

            return substring

    print(big_string)

    return np.nan
# get train & test csv files as DataFrame

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



titanic = train.append(test, ignore_index=True)
titanic['Title'] = titanic['Name'].map(lambda x: substrings_in_string(x, title_list))
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 

            'Countess', 'Don', 'Jonkheer']
titanic.Cabin.values
sns.heatmap(titanic.corr(), annot=True)
titanic.Age.hist()
sns.kdeplot(titanic.Age, shade=True)
sns.boxplot(titanic.Age)
titanic.Age.describe()