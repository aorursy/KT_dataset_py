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
rawData = pd.read_csv('../input/restaurants-in-jaipur/Raw.csv')

rawData.head()
cleanedData = pd.read_csv('../input/restaurants-in-jaipur/Cleaned.csv')

cleanedData.head()
import seaborn as sns

import matplotlib.pyplot as plt
neighborhoodGrouped = cleanedData.groupby('Locality')['RestaurantName'].count().sort_values(ascending=False)

neighborhoodGrouped
sns.set(rc={'figure.figsize':(15,8)})
ax = sns.barplot(neighborhoodGrouped.index, neighborhoodGrouped)



rects = ax.patches

labels = list(neighborhoodGrouped)

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x()+rect.get_width()/2, height, label, ha='center')



plt.xticks(rotation=90)

ax.set_ylabel("Number of Restaurants", fontsize=16)

ax.set_xlabel("Neighborhood", fontsize=16)

ax.set_title("Number of Restaurants in each Neighborhood.", fontsize=20)

plt.show()