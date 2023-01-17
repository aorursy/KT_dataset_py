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
soccer = pd.read_csv("/kaggle/input/us-major-league-soccer-salaries/mls-salaries-2017.csv")
soccer.head(10)
len(soccer.index) # count of column
soccer["base_salary"]
average = soccer["base_salary"].mean()

print(average)
soccer["base_salary"].max() # max base salary
soccer["guaranteed_compensation"].max()
player = soccer[soccer["guaranteed_compensation"] == soccer["guaranteed_compensation"].max()]
player
player["last_name"].iloc[0]
player2 = soccer[soccer["last_name"] == "Gonzalez Pirez"]
player2["position"].iloc[0]
soccer.groupby("position").mean()
soccer["position"].nunique() # how many position we have
soccer["position"].value_counts()
soccer["club"].value_counts()
def find_word(last_name):

    if "hi" in last_name.lower():

        return True

    return False





soccer[soccer["last_name"].apply(find_word)]
import matplotlib.pyplot as plt
clubCount = soccer["club"].value_counts()

print(clubCount)

plt.hist(clubCount,facecolor="blue",edgecolor="white", normed=True, bins=30)

plt.ylabel('clubCount');

plt.show()