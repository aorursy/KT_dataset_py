# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import time



import numpy as np

import matplotlib

import matplotlib.cm as cm

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
sns.set_context(context='talk')
# Import raw data

print("Started at ", time.ctime())

data_path = "../input/8anu-climbing-logbook-csv-files/"

df_ascent = pd.read_csv("{path}ASCENT.csv".format(path=data_path), low_memory=False, index_col=0)

df_grade = pd.read_csv("{path}GRADE.csv".format(path=data_path), index_col=0)

df_user = pd.read_csv("{path}USER.csv".format(path=data_path), low_memory=False, index_col=0)

print("Finished at ", time.ctime())
def standardize_usa_boulder_ratings(row):

    """Group and standardize V-scale ratings"""

    rating = row.usa_boulders

    if rating == 'VB':

        rating = 'V0-'

    elif rating == 'V3/4':

        rating = 'V3'

    elif rating == 'V4/V5':

        rating = 'V4'

    elif rating == 'V5/V6':

        rating = 'V5'

    elif rating == 'V8/9':

        rating = 'V8'

    row.usa_boulders = rating

    

    return row





# Pre-process raw data

df_grade_processed = (

    df_grade

    .loc[df_grade.usa_boulders != '', :] # filter for climbs with V-scale ratings 

    .apply(standardize_usa_boulder_ratings, axis=1)  # group and standardize V-scale ratings

)

df_ascent_processed = df_ascent.loc[df_ascent.climb_type == 1, :] # filter for bouldering climbs
# Merge and filter data to create interim dataset

df_interim = (

    df_ascent_processed

    .merge(df_grade_processed, how='inner', left_on='grade_id', right_on='id', suffixes=('_ascent', '_grade'))

    .loc[:, ['id_ascent', 'id_grade', 'user_id', 'date', 'year', 'usa_boulders', 'name']]  # select relevant columns for project

    .sort_values(by=['user_id', 'date'])

    .reset_index(drop=True)   

)



display(df_interim.head())
# Pivot interim data to setup for analysis

df_interim_pivot = (

    df_interim

    .groupby(['user_id', 'usa_boulders'])

    .nth(0, dropna=None)  # select first ascent at each level for each climber

    .reset_index()

    .pivot_table(index='user_id', columns='usa_boulders', values='date')

    .loc[:, ['V0-', 'V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']]  # limit analysis to V0- to V10 problems

    .reset_index()

)



display(df_interim_pivot.head())
df_clean = (

    df_interim_pivot

    .drop('user_id', axis=1)

)
# Visualize avaialble data

plt.figure(figsize=(10, 5))

sns.heatmap(df_clean.notnull().applymap(lambda x: int(x)), cbar=False, cmap='Blues')

plt.title("Missing Value Heatmap")

plt.xlabel("Boulder Grade")

plt.ylabel("Climbers")

plt.show()



print("Blue indicates available data")

print("Number of Climbs: {}".format(df_clean.notnull().sum().sum()))

print("Number Climbers: {}".format(len(df_clean)))
# Compute time spent at each grade

df_clean_diff = (

    df_clean

    .diff(axis=1)

    .applymap(lambda x: np.nan if x < 0 else x)  # remove negative time intervals

    .applymap(lambda x: x / (3600*24*30))  # convert timestamps to months

    .drop('V0-', axis=1)

    .rename(index=str, columns={

        "V0": "V0-", "V1": "V0", "V2": "V1",

        "V3": "V2", "V4": "V3", "V5": "V4",

        "V6": "V5", "V7": "V6","V8": "V7",

        "V9": "V8", "V10": "V9",

    })  # shift column labels down a grade to account for diff command

)



# Remove outliers

thresh = df_clean_diff.quantile(q=0.99, axis='index')

mask = df_clean_diff.apply(lambda row: row < thresh.values, axis=1)

df_clean_diff_thresh = df_clean_diff[mask]



display(df_clean_diff_thresh.head())
vals = (

    df_clean_diff_thresh

    .rename(index=str, columns={'V0-': '-V0'}) # rename column for plotting purposes

    .mean()

    .cumsum()

)

errs = (

    df_clean_diff_thresh

    .std()

)
# Define colors for plot

cmap = cm.Pastel1

colors = []

colors.extend(cmap(np.linspace(0, 0.1, 2)))

colors.extend(cmap(np.linspace(0.15, 0.2, 3)))

colors.extend(cmap(np.linspace(0.25, 0.3, 2)))

colors.extend(cmap(np.linspace(0.35, 0.4, 4)))



# Create figure

plt.figure(figsize=(15, 10))

ax = plt.gca()



# Creat  Plot

ax.set_facecolor('ghostwhite')

for i, c in enumerate(colors):    

    plt.errorbar(

        vals[i], 

        i, 

        xerr=errs[i], 

        color=c,

        fmt='o', 

        markersize=25,

        markeredgecolor='black',

        markeredgewidth=2.0,

        linewidth=5,

        ecolor=c,

        capsize=7

    )



# Alter y-axis

plt.yticks(

    np.arange(11),

    vals.index

)

    

# Alter x-axis

plt.xticks(

    np.arange(0, 85, 6),

    ['Start', '6 mos', '1yr', '1.5yrs', '2yrs', '', '3yrs', '', '4yrs', '', '5yrs', '', '6yrs', '', '7yrs'])

plt.xlim((-0.1, 78.1))



# Grid

plt.grid(False, axis='y', which='both')

plt.grid(True, axis='x', which='both', linestyle='--', linewidth=1)



# Titles

plt.title("Expected Bouldering Grade\nBased on Time Climbing", fontsize=35)

plt.ylabel("Outdoor Bouldering Grade", fontsize=35)

plt.xlabel("Time Bouldering Outdoors", fontsize=35)



# Font size

plt.rc('xtick', labelsize=20)

plt.rc('ytick', labelsize=20)



plt.show()



print("Plot Values:")

display(vals)