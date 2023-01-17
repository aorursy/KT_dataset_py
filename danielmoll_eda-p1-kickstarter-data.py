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
projects_2018 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

projects_2018.head()
print("There are", projects_2018.shape[0], "rows")

print("There are", projects_2018.shape[1], "columns")
print(projects_2018.columns.values)
# Selecting a subset of our orginal dataframe to work with

projects_short = pd.DataFrame(data=projects_2018,columns=['name','category', 'deadline', 'goal', 'launched', 'pledged', 'state', 'backers', 'country'])

more_than_100 = projects_short[projects_short.backers > 100]

print(more_than_100)
projects_short.sort_values('pledged', ascending=False).iloc[0]
maximum = projects_short.backers.max()

print(maximum)
projects_short.query("pledged > 1000 and backers <=10")
projects_short.groupby('category').mean().sort_values("pledged",ascending = False).iloc[0]
sorted_by_frequency = projects_short.groupby('country').size().sort_values(ascending=False).plot.pie()

projects_short.groupby('country').mean().sort_values("backers",ascending=False)
projects_short.plot.scatter(x="goal",y="pledged")