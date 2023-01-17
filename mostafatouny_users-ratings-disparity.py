# 3rd-party libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# local file (as kaggle utility script)

import switch_functions as func
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
# drop unneeded columns and re-organize them 

for name in platformsNames:

    platform_df[name] = platform_df[name][['title', 'user_rating', 'critic_rating']]
# for each platform

for name in platform_df:

    # get dataframe of the platform

    df = platform_df[name]

    # for each record, compute distance between user and critic ratings, then set result to a new column

    df['userCritic_difference'] = df.apply(lambda x: abs(x['user_rating']-x['critic_rating']), axis=1)

    # assign updates back to our dataframe

    platform_df[name] = df
# categories names and their corresponding intervals

# category at location x corresponds to interval equal or greater than intervals location x and less than location x + 1

# except for last category, has no end

categories = pd.Series(["low", "moderate", "high", "very high", "extremely high"], dtype="category")

intervals_categories = [0, 20, 30, 40, 50]
# map a value to its interval

def numToCat(row):

    row_catValu = row

    

    # check if value is in between two boundaries

    for idx in range(len(intervals_categories)-1):

        if row_catValu >= intervals_categories[idx] and row_catValu < intervals_categories[idx+1]:

            return categories.iloc[idx]

    # if not, then check if it is greater than latest boundary

    lastIndex = len(categories)-1

    if row_catValu >= intervals_categories[lastIndex]:

        return categories.iloc[lastIndex]

    # if not either, raise error

    raise ValueError("unexpected value within supposed ranges")
# compute categories as defined earlier



# loop on platforms

for platformName in platform_df:

    # get dataframe of the platform

    df = platform_df[platformName]

    # add category based on difference just defined

    df['difference_category'] = df.apply(lambda x: numToCat(x['userCritic_difference']), axis=1)

    

    # let categories be recognized by pandas

    df['difference_category'] = df['difference_category'].astype("category")

    

    # re-order categories

    df['difference_category'] = df['difference_category'].cat.set_categories(categories)

    

    # assign back to our dataframe

    platform_df[platformName] = df
# construct a map from a platform to its categories count

platform_categoriesCount = func.map_columnCount(platform_df, 'difference_category')
# construct a map from a category to its sizes among platforms

categories_size = func.map_categoriesSize(categories, platformsNames, platform_categoriesCount)
func.showGroupedBars(categories_size, platformsNames, "disparity", "users and critics disparity among platforms")
# construct categories sizes as a 2d list

categoriesSize_2dList = func.ConstCategoriesSize_2dList(categories_size)
func.showCategoricalHeatmap(8, 8, categoriesSize_2dList, platformsNames, categories_size, "users and critics disparity among platforms")
# Comparing pairs of platforms



# construct a map of platforms pairs to the two platforms distance dataframe

platformsPairs_df = func.constPlatformsPairs(platform_df)
# Compute difference of each pair of platforms



# a map of platforms pairs to dataframe showing computed results

platformsPairs_diff = func.constPlatformsPairs_diff(platformsNames, platformsPairs_df, platform_df)
# map a value to its interval

def numToCat(row):

    row_userCriticDiff = row['user_diff']

    

    # check if value is in between two boundaries

    for idx in range(len(intervals_categories)-1):

        if row_userCriticDiff >= intervals_categories[idx] and row_userCriticDiff < intervals_categories[idx+1]:

            return categories.iloc[idx]

    # if not, then check if it is greater than latest boundary

    lastIndex = len(categories)-1

    if row_userCriticDiff >= intervals_categories[lastIndex]:

        return categories.iloc[lastIndex]

    # if not either, raise error

    raise ValueError("unexpected value within supposed ranges")
# compute categories as defined earlier



# loop on platforms

for pair_diff in platformsPairs_diff:

    # get dataframe of the platform

    df = platformsPairs_diff[pair_diff]

    # add category based on difference just defined

    df['difference_category'] = df.apply(lambda x: numToCat(x), axis=1)

    

    # let categories be recognized by pandas

    df['difference_category'] = df['difference_category'].astype("category")

    

    # re-order categories

    df['difference_category'] = df['difference_category'].cat.set_categories(categories)

    

    # assign back to our dataframe

    platformsPairs_diff[pair_diff] = df
# construct a map from platforms pairs to categories count

pair_categoriesCount = func.map_columnCount(platformsPairs_diff, 'difference_category')
# construct a map from a category to its sizes among platforms

categories_size = func.map_categoriesSize(categories, pair_categoriesCount, pair_categoriesCount)
func.showGroupedBars(categories_size, platformsPairs_diff, 'dispariy count', 'Platforms Pairs Disparity in Users Ratings')
# cosntruct categories sizes in a 2d-list format

categoriesSize_2dList = func.ConstCategoriesSize_2dList(categories_size)
func.showCategoricalHeatmap(8, 8, categoriesSize_2dList, platformsPairs_diff, categories_size, "Platforms Pairs Disparity in Users Ratings")
platformsPairs_diff['ps4_switch'].sort_values(by='user_diff', ascending=False).head(20)
platformsPairs_diff['xbox_switch'].sort_values(by='user_diff', ascending=False).head(20)
platformsPairs_diff['switch_pc'].sort_values(by='user_diff', ascending=False).head(20)