# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import the Academy best pictures data into the data frame
adb = pd.read_csv('../input/AA_best_pictures.csv')
# Perform a quick profiling on the numerical columns
adb.describe()
# Look at the first five rows
adb.head()
# What is the average durration for the best movies?
adb.duration.mean()
# Which genre of movies do well at Academy awards?
adb.genre1.value_counts().plot(kind='bar');
# What months are the best pictures got released?
adb.release.value_counts()
adb.release.value_counts().plot(kind='pie', autopct='%1.0f%%',figsize=(10,10));
# We concluded that Dramas perform well at Oscars, when these movies released?
dm=adb[adb.genre1 == 'Drama'].release.value_counts()
dm
dm = dm.reset_index()
dm.columns=['Month', 'count']
# Plot the month released and number of movies in drama genre
plt.scatter(x="Month", y="count", data=dm, alpha=0.5,marker='o');
plt.gcf().set_size_inches((10, 5))  
# Does number of nominations have an impact on deciding the best picture?
# Is there any change in the rating over the years for the best picures?
# Is there an relation between nomiations and ratings?
adb.plot(x='year', y=["nominations", "rating"],figsize=(10,5),grid=True );
adb.rating.mean()
