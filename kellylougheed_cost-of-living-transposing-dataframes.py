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

cost_of_living = pd.read_csv("../input/cost-of-living/cost-of-living.csv")
cost_of_living = cost_of_living.T

cost_of_living.head()
import seaborn as sns



# Get the zeroeth column ("Meal, Inexpensive Restaurant")

# Then skip the zeroeth row (which contains the name) and start with the first row

# Convert everything to a float (from the original string)

# And make it a box plot!



sns.boxplot(y=cost_of_living[0][1:].astype(float))
# All the cities

print(cost_of_living[0].keys()[1:])
import matplotlib.pyplot as plt

 

# Get data

# Restaurant costs

height = cost_of_living[0][1:].astype(float)

# Cities

bars = cost_of_living[0].keys()[1:]

y_pos = np.arange(len(bars))



# Set size

plt.figure(figsize=(20,10))

 

# Create bars

plt.bar(y_pos, height)

 

# Create names on the x-axis & rotate them 90 degrees

plt.xticks(y_pos, bars, rotation=90)

 

# Show graphic

plt.show()