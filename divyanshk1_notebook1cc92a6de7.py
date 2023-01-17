# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/deliveries.csv')

matches = pd.read_csv('../input/matches.csv')

data.columns,matches.columns
col = ["batsman","batsman_runs"]

bats = data[col]

bats.sample
fours = bats[bats.batsman_runs ==4]

fours.columns = ["batsman","fours"]

fours = fours.groupby('batsman').agg({'fours':'count'}).reset_index()

sixes = bats[bats.batsman_runs ==6]

sixes.columns = ["batsman","sixes"]

sixes = sixes.groupby('batsman').agg({'sixes':'count'}).reset_index()
h = bats.batsman.value_counts()

co = {'bowls': h}

c = pd.DataFrame(co).reset_index()

c.columns = ['batsman','bowls']
grouped = bats.groupby('batsman').agg({'batsman_runs':'sum'}).reset_index()

bats = grouped.merge(fours,how = 'left',on='batsman')

bats = bats.merge(sixes,how = 'left',on='batsman')

bats = bats.merge(c,how = 'left',on='batsman')

bats.fours.iloc[bats.fours is NaN] = 0

bats.sixes.iloc[bats.sixes is NaN] = 0

bats.head
x = bats[bats.bowls!= 0]

x.bowls.unique

#sr = pd.DataFrame(sr).reset_index()

#sr.columns = ['batsman','sr']

#bats = bats.merge(sr)

#bats.describe()
ax = bats[['fours','sixes']].plot(kind='bar')