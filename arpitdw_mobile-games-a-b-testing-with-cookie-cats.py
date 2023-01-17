# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# Reading in the data

df = pd.read_csv('/kaggle/input/cookie_cats.csv')



# Showing the first few rows

df.head()
# Counting the number of players in each AB group.

df.groupby('version')['userid'].count()
# This command makes plots appear in the notebook

%matplotlib inline



# Counting the number of players for each number of game rounds 

plot_df = df.groupby('sum_gamerounds')['userid'].count()



# Plotting the distribution of players that played 0 to 100 game rounds

ax = plot_df.head(n=100).plot(x="sum_gamerounds", y="userid")

ax.set_xlabel("Game Rounds")

ax.set_ylabel("User Count")
# The % of users that came back the day after they installed

df['retention_1'].sum() / df['retention_1'].count()
# Calculating 1-day retention for each AB-group

df.groupby('version')['retention_1'].sum() / df.groupby('version')['userid'].count()
# Creating an list with bootstrapped means for each AB-group

boot_1d = []

for i in range(500):

    boot_mean = df.sample(frac=1, replace=True).groupby('version')['retention_1'].mean()

    boot_1d.append(boot_mean)

    

# Transforming the list to a DataFrame

boot_1d = pd.DataFrame(boot_1d)

    

# A Kernel Density Estimate plot of the bootstrap distributions

boot_1d.plot(kind='kde')
# Adding a column with the % difference between the two AB-groups

boot_1d['diff'] = (boot_1d['gate_30'] - boot_1d['gate_40']) /  boot_1d['gate_40'] * 100



# Ploting the bootstrap % difference

ax = boot_1d['diff'].plot(kind = 'kde')

ax.set_xlabel("% difference in means")
# Calculating the probability that 1-day retention 

# is greater when the gate is at level 30.

prob = (boot_1d['diff'] > 0).sum() / len(boot_1d)



# Pretty printing the probability

'{:.1%}'.format(prob)
# Calculating 7-day retention for both AB-groups

df.groupby('version')['retention_7'].sum() / df.groupby('version')['userid'].count()
# Creating a list with bootstrapped means for each AB-group

boot_7d = []

for i in range(500):

    boot_mean = df.sample(frac=1, replace=True).groupby('version')['retention_7'].mean()

    boot_7d.append(boot_mean)

    

# Transforming the list to a DataFrame

boot_7d = pd.DataFrame(boot_7d)



# Adding a column with the % difference between the two AB-groups

boot_7d['diff'] = (boot_7d['gate_30'] - boot_7d['gate_40']) /  boot_7d['gate_30'] * 100



# Ploting the bootstrap % difference

ax = boot_7d['diff'].plot(kind = 'kde')

ax.set_xlabel("% difference in means")



# Calculating the probability that 7-day retention is greater when the gate is at level 30

prob = (boot_7d['diff'] > 0).sum() / len(boot_7d)



# Pretty printing the probability

'{:.1%}'.format(prob)
# So, given the data and the bootstrap analysis

# Should we move the gate from level 30 to level 40 ?

move_to_level_40 = False # True or False ?