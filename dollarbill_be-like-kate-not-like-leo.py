# Import required modules



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd

import seaborn as sns # data visualisation

import matplotlib.pyplot as plt

%matplotlib inline
# Import training and test data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# View first five rows of dataset

train_df.head(10)
# Get aggregate data on numerical features

train_df.describe()
# Get aggregate data on non-numerical features

train_df.describe(include=['O'])
# Visualise distributions over key features and correlation with survival.

survived_df = train_df[train_df['Survived']==1]



def histogram_for_feature(feature):

    bars = train_df[feature].value_counts()

    survived_bars = survived_df[feature].value_counts()

    plt.bar(bars.index, bars)

    plt.bar(survived_bars.index, survived_bars)

    plt.ylabel('# of passengers')

    plt.xlabel(feature)

    plt.show()

histogram_for_feature('Sex')

histogram_for_feature('Age')
# Break age into 10-year blocks

train_df['Decade'] = np.floor(train_df['Age']/10)

survived_df['Decade'] = np.floor(survived_df['Age']/10)

histogram_for_feature('Decade')
histogram_for_feature('Pclass')

histogram_for_feature('Fare')
histogram_for_feature('Embarked')
histogram_for_feature('Parch')

histogram_for_feature('SibSp')
def combine_features(feature_1, feature_2, index):

    all_passengers = train_df.groupby([feature_1, feature_2]).size()

    survivors = survived_df.groupby([feature_1, feature_2]).size()

    all_passengers.index = survivors.index = index

    plt.bar(all_passengers.index, all_passengers)

    plt.bar(survivors.index, survivors)

    plt.show()
combine_features('Pclass', 'Sex', ['1f', '1m', '2f', '2m', '3f', '3m'])
combine_features('Decade', 'Sex', ['0f', '0m', '1f', '1m', '2f', '2m', '3f', '3m', '4f', '4m', '5f', '5m', '6f', '6m', '7m', '8m'])
combine_features('Embarked', 'Sex', ['Cf', 'Cm', 'Qf', 'Qm', 'Sf', 'Sm'])
combine_features('Embarked', 'Pclass', ['C1', 'C2', 'C3', 'Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3'])