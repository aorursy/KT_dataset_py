# Let's take a look at the files

import os

sorted(os.listdir('../input/collegefootballstatistics'))
# We have the root dir, then a bunch of sub folders by year

# Let's explore a particular year

sorted(os.listdir('../input/collegefootballstatistics/cfbstats-com-2013-1-5-20'))
def remove_csv(inp):

    return inp.replace('.csv', '')
# We want to construct a mapping year -> type -> fname

ROOT_DIR = '../input/collegefootballstatistics/'

dirs = filter(lambda i: '__' not in i, os.listdir(ROOT_DIR))

result = dict()

START_YEAR = 2005

for offset, basename in enumerate(sorted(dirs)): # Ignore MACOSX file

    year = START_YEAR + offset

    # Now, for each subkey, add paths

    current = result[year] = dict()

    sub_dir = os.path.join(ROOT_DIR, basename)

    for file in os.listdir(sub_dir):

        if file.rfind('.csv') > -1:

            current[remove_csv(file)] = os.path.join(sub_dir, file)
print(result[2006]['pass'])
import pandas as pd
# Now we have a way to look up stats and years

# Let's play around

# Let's print the keys we can work with again

result[2005].keys()
# Let's look at all kickoff returns for 2009

ret = pd.read_csv(result[2009]['kickoff-return'])
ret.head()
# Let's plot the distribution of yards here

%matplotlib inline

import seaborn as sns

sns.countplot(x=ret.Yards.name, data=ret)
ret.Yards.hist()
sns.distplot(ret.Yards)
ret.describe()
# Let's look at some more recent player data

players = pd.read_csv(result[2013]['player'])
players.head()
# Let's look at the class distribution

players.Class.hist()
import matplotlib.pyplot as plt
# And their height and weight

for k in ['Height', 'Weight']:

    ax = sns.distplot(players[k].dropna())

    ax.set(xlabel=k, ylabel='Ratio')



    plt.show()
# Let's examine some conferences to finish up

conf = pd.read_csv(result[2010]['conference'])

conf.head()
conf.shape
conf.Subdivision.unique()
# Let's examine distribution in FCS/FBS

conf.Subdivision.hist()