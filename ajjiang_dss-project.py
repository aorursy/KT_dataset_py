# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/oakland-crime-911-calls-gun-incidents/prr-20055-2010.csv', index_col = ['Agency'])

df.head()
to_drop = ["Area Id", "Incident Type Id", "Event Number", "Closed Time", "Zip Codes"]

df.drop(to_drop, inplace = True, axis = 1)

df.head()
new_names = {"Create Time" : "Date", "Location 1" : "Location", "Incident Type Description" : "Type"}

df.rename(columns = new_names, inplace = True)

df.head()
df2 = pd.read_csv('../input/oakland-crime-911-calls-gun-incidents/prr-10437.csv', index_col = ['EVENT NUMBER'])

df2.head()
to_drop = ["INCIDENT TYPE"]

df2.drop(to_drop, inplace = True, axis = 1)

df2.head()
df.shape
df2.shape
new_names = {"DATE/TIME" : "Date", "ADDRESS ROUNDED TO BLOCK NUMBER OR INTERSECTION" : "Location", "PATROL BEAT" : "Beat", "INCIDENT TYPE DESCRIPTION": "Type", "PRIORITY" : "Priority"}

df2.rename(columns = new_names, inplace = True)

df2.head()
df = pd.concat([df, df2])

df
NUM = 10000

df = df.sample(frac=1).reset_index(drop=True)

df = df.head(NUM)
ratings_file = open('/kaggle/input/crime-ratings/ratings.json')

ratings = ratings_file.read()

ratings_file.close()



from ast import literal_eval

ratings = literal_eval(ratings)
ratings_column = []

unindexed = set()

for index, row in df.iterrows():

    row_type = row['Type']

    if not isinstance(row_type, str):

        ratings_column.append(float('nan'))

    elif row_type in ratings:

        ratings_column.append(ratings[row_type])

    else:

        print(row_type)
df['Danger Rating'] = ratings_column
to_drop = ["Priority"]

df.drop(to_drop, inplace = True, axis = 1)

df.head(10)
import matplotlib.pyplot as plt

from math import isnan



def get_dangers_dict(df, column_name='Beat'):

    """Get a dictionary of BEATS and DANGERS of those beats based on a dataframe"""

    dangers = dict()

    for index, row in df.iterrows():

        beat = row[column_name]

        if isinstance(beat, str) and beat != 'PDT2':

            if beat not in dangers:

                dangers[beat] = 0

            value = row['Danger Rating']

            if isnan(value):

                value = 0

            value = int(value)

            dangers[beat] += value

    return dangers



dangers = get_dangers_dict(df)
def plot_danger(dangers_dict):

    """Plot DANGER ratings against BEATS where those dangers occur based on a dictionary with BEATS as keys and DANGERS as values"""

    x_vals, y_vals = [], []

    for key, value in dangers.items():

        x_vals.append(key)

        y_vals.append(value)

    plt.bar(x_vals, y_vals, align='center')

    plt.xticks(rotation=80)

    plt.show()

    

plot_danger(dangers)
safest_beat, danger_beat = None, None

least_danger, most_danger = float('inf'), 0

for key, value in dangers.items():

    if value < least_danger:

        least_danger = value

        safest_beat = key

    if value > most_danger:

        most_danger = value

        danger_beat = key

print("Most dangerous beat is", danger_beat)

values = np.array(list(dangers.values()))

average_danger = np.mean(values)

relative_difference = (most_danger - average_danger) / average_danger * 100

print("This beats danger exceeds the average danger by {0} %".format(relative_difference))
print("Is there some reason that makes this district the most dangerous or is it merely a statistical fluctuation? Let's test it!")

print("Null hypothesis: There is no external factor which makes that district the most dangerous")

print("Alternative hypothesis: There is some factor that makes that district the most dangerous")

print("Test statistic: Sum of danger ratings of all crimes that happened in a given beat")
BEATS_ARR = np.array(list(dangers.keys()))

import random



def randomize_beats(amount=len(df.index)):

    random.seed()

    randomized = list()

    for i in range(amount):

        randomized.append(BEATS_ARR[random.randrange(BEATS_ARR.size)])

    return randomized



def add_randomized_column():

    df['Randomized'] = np.array(randomize_beats())



add_randomized_column()

df.head(10)
import seaborn as sns



repetitions = 1000

differences = np.array(list())

for i in range(repetitions):

    add_randomized_column()

    new_dangers = get_dangers_dict(df, column_name='Randomized')

    new_values = np.array(list(new_dangers.values()))

    differences = np.append(differences, new_dangers[danger_beat] - average_danger)



sns.distplot(differences)
observed_difference = most_danger - average_danger

p_value = np.count_nonzero(differences >= observed_difference) / repetitions

p_value