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
import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

fifa_filepath = "../input/australian-road-accident/ardd_fatal_crashes.csv"



# Read the file into a variable fifa_data

fifa_data = pd.read_csv(fifa_filepath, index_col="Crash ID", parse_dates=True)
# Print the first 5 rows of the data

fifa_data.head()
fifa_data.tail()
list(fifa_data.columns)
# create a figure and axis

fig, ax = plt.subplots()



# scatter the sepal_length against the sepal_width

ax.scatter(fifa_data['Number Fatalities'], fifa_data['Year'])

# set a title and labels

ax.set_title('Australian road Accident Dataset')

ax.set_xlabel('Number Fatalities')

ax.set_ylabel('Year')
fifa_data.describe()
fifa_data.head(10)
fifa_data.info()
fifa_data
fifa_data['Year'].plot.hist()


fifa_data.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)
fifa_data['State'].value_counts().sort_index().plot.barh()
sns.pairplot(fifa_data)


from pandas.plotting import scatter_matrix



fig, ax = plt.subplots(figsize=(12,12))

scatter_matrix(fifa_data, alpha=1, ax=ax)
g = sns.FacetGrid(fifa_data, col='Crash Type')

g = g.map(sns.kdeplot, 'Year')


sns.heatmap(fifa_data.corr(), annot=True)


# get correlation matrix

corr = fifa_data.corr()

fig, ax = plt.subplots()

# create heatmap

im = ax.imshow(corr.values)



# set labels

ax.set_xticks(np.arange(len(corr.columns)))

ax.set_yticks(np.arange(len(corr.columns)))

ax.set_xticklabels(corr.columns)

ax.set_yticklabels(corr.columns)



# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

         rotation_mode="anchor")



# Loop over data dimensions and create text annotations.

for i in range(len(corr.columns)):

    for j in range(len(corr.columns)):

        text = ax.text(j, i, np.around(corr.iloc[i, j], decimals=2),

                       ha="center", va="center", color="black")


import numpy as np



# get correlation matrix

corr = fifa_data.corr()

fig, ax = plt.subplots()

# create heatmap

im = ax.imshow(corr.values)



# set labels

ax.set_xticks(np.arange(len(corr.columns)))

ax.set_yticks(np.arange(len(corr.columns)))

ax.set_xticklabels(corr.columns)

ax.set_yticklabels(corr.columns)



# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

         rotation_mode="anchor")


sns.scatterplot(x='State', y='Year', hue='Crash Type', data=fifa_data)

sns.distplot(fifa_data['Year'], bins=10, kde=False)
sns.distplot(fifa_data['Year'], bins=10, kde=True)


fifa_data['State'].value_counts().sort_index().plot.bar()