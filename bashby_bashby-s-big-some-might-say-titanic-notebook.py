# Libs to work with the data

import pandas as pd

import numpy as np

import random as rnd

import seaborn as sns

sns.set(style="darkgrid")

import matplotlib.pyplot as plt



# Print out what files are available to work with

from subprocess import check_output

print("Files:\n", check_output(["ls", "../input"]).decode("utf8"))



# Load the data

train_raw = pd.read_csv("../input/train.csv")

test_raw = pd.read_csv("../input/test.csv")

combined_raw = pd.concat([train_raw, train_raw])



# Let's look at the possible data to even play with

print("Columns:\n", combined_raw.columns.values)



# Let's look at the data

# Categorical Features

print("\nCategorical:\n", combined_raw.describe(include=['O']))

# Numerical Features

combined_raw.describe()
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: (x.split(',')[1]).split('.')[0])

    return df    

    

def drop_features(df):

    return df.drop(['Ticket', 'Embarked'], axis=1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = drop_features(df)

    return df



train_formatted = transform_features(train_raw)

test_formatted = transform_features(test_raw)

combined_formatted = pd.concat([train_formatted, test_formatted])

combined_formatted.head()



plt.figure(figsize=(8,8))

sns.countplot(y="NamePrefix", data=combined_formatted)
combined_formatted.loc[combined_formatted['NamePrefix'] == "y"]