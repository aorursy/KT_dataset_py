# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # For visualization

import seaborn as sns # For Visualization

import operator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Reading the csv file and storing the data to a dataframe.



csv_file = "/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv"

df = pd.read_csv(csv_file)

df.head()
df.columns
df.describe(include="all")
dev_user_rating = []

for dev, group in df.groupby('Developer'):



    dev_user_rating.append((dev, np.sum(group['Average User Rating'] * group['User Rating Count'])))



dev_user_rating.sort(key = operator.itemgetter(1), reverse = True)

top_10_devs = dev_user_rating[:10]

print (top_10_devs)
dev, total_rating = list(zip(*top_10_devs))

dev , total_rating
# Visualization



plt.figure(figsize=(40, 20))

sns.barplot(x=list(dev), y=list(total_rating), palette="rocket")

plt.show()
supercell = df[df['Developer'] == 'Supercell']

supercell
nam = list(supercell['Name'])

subtitle = list(supercell['Subtitle'])

avg_rating = list(supercell['Average User Rating'])

geners = list(supercell['Genres'])



for i in range(len(nam)):

    print("Name :",nam[i])

    print("Subtitle:",subtitle[i])

    print("Average Ratings:",avg_rating[i])

    print("Genres:",geners[i])

    print("\n\n")