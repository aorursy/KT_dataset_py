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
import matplotlib.pyplot as plt

from collections import Counter
df = pd.read_csv('/kaggle/input/crime.csv', encoding = "ISO-8859-1")
df.head()
crime_counts = Counter(np.array(df['OFFENSE_CODE_GROUP']))

count_dict = dict(crime_counts)

pop_crimes = dict(sorted(count_dict.items(), key=lambda kv: kv[1])[::-1][:8])

counts_df = pd.DataFrame.from_dict(pop_crimes, orient='index', columns=['Number of crimes in 2015-2018'])

counts_df.plot(figsize=(15, 15), kind='bar', title='Most often crimes', )

harr_dict = {}

for year in range(2015, 2019):

    harr_dict[year] = df[(df['OFFENSE_CODE_GROUP'] == 'Harassment') & (df['YEAR'] == year)].shape[0]

counts_df = pd.DataFrame.from_dict(harr_dict, orient='index', columns=['Number of harassment crimes'])

counts_df.plot(figsize=(7, 7), kind='bar', title='Harassment crimes', color='orange')
def bar_plot(column, values, by, title=None, columns=[], figsize=(15, 5), color=None):

    count_dict = {}

    try:

        for i in sorted(df[by].unique()):

            if str(i) != 'nan':

                count_dict[i] = [df[(df[column] == value) & (df[by] == i)].shape[0] for value in values if value != 'Other']

    except Exception:

        for i in df[by].unique():

            if str(i) != 'nan':

                count_dict[i] = [df[(df[column] == value) & (df[by] == i)].shape[0] for value in values if value != 'Other']

    counts_df = pd.DataFrame.from_dict(count_dict, orient='index', columns=[i for i in columns if i != 'Other'])

    counts_df.plot(figsize=figsize, kind='bar', title=title, color=color)
bar_plot('OFFENSE_CODE_GROUP', pop_crimes, 'DISTRICT', columns=pop_crimes, title='Popular crimes by districts')
bar_plot('OFFENSE_CODE_GROUP', pop_crimes, 'HOUR', columns=pop_crimes, figsize=(20, 10), title='Popular crimes by hours')
bar_plot('OFFENSE_CODE_GROUP', pop_crimes, 'DAY_OF_WEEK', columns=pop_crimes, title='Popular crimes by week days')
shoot_counts = Counter(np.array(df[df['SHOOTING'] == 'Y']['DAY_OF_WEEK']))

counts_df = pd.DataFrame.from_dict(shoot_counts, orient='index', columns=['Number of shootings in 2015-2018'])

counts_df.plot(figsize=(8, 6), kind='bar', title='Shootings in 2015-2018 by week days', color='red')
bar_plot('OFFENSE_CODE_GROUP', ['Homicide'], 'YEAR', columns=['Homicide crimes'], title='Homicide crimes by years', color='gray')
bar_plot('OFFENSE_CODE_GROUP', ['Homicide'], 'DAY_OF_WEEK', columns=['Homicide crimes'], title='Homicide crimes by week days', color='gray')
bar_plot('OFFENSE_CODE_GROUP', ['Homicide'], 'HOUR', columns=['Homicide crimes'], title='Homicide crimes by hours', color='gray')