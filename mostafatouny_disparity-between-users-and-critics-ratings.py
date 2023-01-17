# 3rd-party libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# read input from kaggle into a dataframe



# a map from each platform to its corresponding dataframe

platform_df = {}

# platform names and their corresonding data file names

platformsNames = ['ps4', 'xbox', 'switch', 'pc']

filesNames = ['ps4.csv', 'xbox.csv', 'switch.csv', 'pc.csv']



# for each platform, then 

for name in platformsNames:

    # parse it as pandas dataframe, then map platform name to it

    platform_df[name] = pd.read_csv('/kaggle/input/metacritic-best-2019-video-games/' + name + '.csv')
# take a look at a dataframe

platform_df['ps4']
platform_df['ps4'].dtypes
# drop unneeded columns and re-organize them 

for name in platformsNames:

    platform_df[name] = platform_df[name][['title', 'user_rating', 'critic_rating']]
# take a look at a dataframe, again

platform_df['xbox']
# for each platform

for name in platform_df:

    # get dataframe of the platform

    df = platform_df[name]

    # for each record, compute distance between user and critic ratings, then set result to a new column

    df['userCritic_difference'] = df.apply(lambda x: abs(x['user_rating']-x['critic_rating']), axis=1)

    # assign updates back to our dataframe

    platform_df[name] = df
platform_df['pc']
# define categories and their intervals

def numToCat(row):

    # equal or greater than 30

    if row['userCritic_difference'] >= 30:

        return 'high'

    # equal or greater than 20 and less than 30

    elif row['userCritic_difference'] >= 20:

        return 'moderate'

    # less than 20

    else:

        return 'low'
# compute categories as defined earlier



# loop on platforms

for platformName in platform_df:

    # get dataframe of the platform

    df = platform_df[platformName]

    # add category based on difference just defined

    df['difference_category'] = df.apply(lambda x: numToCat(x), axis=1)

    # let categories be recognized by pandas

    df['difference_category'] = df['difference_category'].astype("category")

    # re-order categories

    df['difference_category'] = df['difference_category'].cat.set_categories(["low", "moderate", "high"])

    # assign back to our dataframe

    platform_df[platformName] = df
# take a look after our new columns added

platform_df['switch']
# for each platform

for platformName in platform_df:

    # get platform dataframe

    df = platform_df[platformName]

    # sort it by userCritic_difference

    df = df.sort_values(axis=0, by='userCritic_difference', ascending=False)

    # assign sorted dataframe back to our dataframe

    platform_df[platformName] = df
platform_df['ps4'].head(20)
platform_df['xbox'].head(20)
platform_df['pc'].head(20)
platform_df['switch'].head(20)
platform_df['ps4'].tail(20)
# filter only records whose user ratings is greater than critics ratings

def higherUserRatings(platform_in):

    return platform_df[platform_in][platform_df[platform_in]['user_rating'] > platform_df[platform_in]['critic_rating']].head(10)
higherUserRatings('pc')
higherUserRatings('ps4')
higherUserRatings('xbox')
higherUserRatings('switch')
plt.close('all')

# for each platform dataframe

for platformName in platform_df:

    print("\non platform ", platformName)

    # count categories among all records

    categories_count = platform_df[platformName].groupby('difference_category').size()

    # construct a series based on it

    pie_series = pd.Series(categories_count, name='categories percentages')

    # plot a pie chart

    pie_series.plot.pie(figsize=(6,6))

    plt.show()
# for each platform

for platformName in platform_df:

    # print platform name

    print("\n", "on ", platformName)

    # show basic stat

    print(platform_df[platformName]['userCritic_difference'].describe())