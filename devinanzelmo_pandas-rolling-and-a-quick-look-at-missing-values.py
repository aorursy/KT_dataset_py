import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import kagglegym

import missingno # very useful for quickly visualizing nan values

import matplotlib.pyplot as plt
# set up the kagglegym environment

env = kagglegym.make()

observation = env.reset()
id_groups = observation.train.groupby(['id'])

example_group = id_groups.get_group(22)

example_group.head(8)
example_group.shape
# take a look at the missing values for the first 55 columns

missingno.matrix(example_group.iloc[:,:55], figsize=(8,6))
# the missing values for the remaining columns

missingno.matrix(example_group.iloc[:,55:], figsize=(8,6))
# take rolling mean, for each group. 

# Id and timestamp should be excluded from this calculation as there is no meaning to smoothed values of these

id_groups = observation.train.groupby(['id'])

smoothed = id_groups.apply(lambda group:group.rolling(window=5).mean()).reset_index(0, drop=True) # found with form with internet search

smoothed = smoothed.sort_index()

smoothed