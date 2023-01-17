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
# read data

df = pd.read_csv('/kaggle/input/ufcdata/data.csv')



# set index of the dataframe to 'date' column to easily filter the last 5 years

df.set_index('date', inplace=True)



# select fights of the last 5 years 

df = df.loc['2019-06-08':'2015-01-01']



# get rid of all not needed columns (we just need the two fighters and the information who won)

important_data = df[['R_fighter', 'B_fighter', 'Winner']]

important_data.head()
# helper function to extract the winner's name

def get_winner(row):

    if row['Winner'] == 'Red':

        return row['R_fighter']

    elif row['Winner'] == 'Blue':

        return row['B_fighter']

    else:

        return None

    

# we don't care about chained assignments in this use case

pd.options.mode.chained_assignment = None 



# we save the winner's name in a new column

important_data['Winner_name'] = important_data.apply(get_winner, axis=1)



# we count the occurences of every name

winner = important_data['Winner_name']

winner.value_counts().head()