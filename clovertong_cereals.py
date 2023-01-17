# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

cereal = pd.read_csv("../input/80-cereals/cereal.csv")

# Take a glance of the dataframe

cereal = cereal.reset_index(drop=True).set_index('name')

cereal.head()
# How many kinds of cereals are there?

cereal.shape[0]
# What about we only want cereal that has sugar less than 5 

cereal[cereal['sugars'] <= 5]

cereal = cereal.sort_values('calories')

cereal.head()
# What are the 10 cereals that have the least calories, and what is the median carbo of those cereals?

least_ten_cal = cereal.iloc[:10]

least_ten_cal['carbo'].median()
# What are the ratings for the cereals with the bottom half of the carbo?

less_carbo = least_ten_cal[least_ten_cal['carbo'] <= 14]

less_carbo['rating'].plot('barh')
# What is the most optimal cereal that satisfy all of the criterias?

most_optimal = least_ten_cal[least_ten_cal['rating'] == least_ten_cal['rating'].max()]

most_optimal
# How many kinds of cereals in each type? 

cereal.groupby('type').size()
# What is the average rating of each manufacturer's cereals

cereal.groupby('mfr').mean().round(2).iloc[:,-1]