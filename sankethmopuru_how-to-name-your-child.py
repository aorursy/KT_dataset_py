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
nNames = pd.read_csv('../input/NationalNames.csv')
namescount = nNames.groupby(['Name'])['Count'].sum().reindex()
#Most common name

namescount.argmax('Count')
# Least common name

namescount.argmin('Count')
namescount.sort_values(inplace=True, ascending=False)
# Top 10 famous names of all time

namescount[:10]
# Top 10 names by gender.

namecountbygender = nNames.groupby(['Gender', 'Name']).Count.sum().reset_index()
namecountbygender[namecountbygender.Gender == 'F'].sort_values('Count', ascending=False)[:10]
namecountbygender[namecountbygender.Gender == 'M'].sort_values('Count', ascending=False)[:10]
a = nNames.groupby(['Name', 'Gender']).Count.sum().unstack()

b = a[a.F.notnull()] 

c = b[b.M.notnull()]

c['diff'] = pd.DataFrame.abs(c['F']-c['M'])
# Works for a boy or a girl :P

c.sort_values('diff')[:10]
# Rare names

# Kerala ?!#@$

namescount[-20:]
namecountbefore2k = nNames[nNames.Year < 2000].groupby(['Name']).Count.sum().reset_index()

namecountafter2k = nNames[nNames.Year >= 2000].groupby(['Name']).Count.sum().reset_index()

a = namecountafter2k.merge(namecountbefore2k, on='Name', how='left')

a = a.fillna(0)

a['relative'] = a.Count_y/a.Count_x
# Names that were very famous before the year 2000 but became less used after

# Not really surprising once you look at the names

a[a.relative > 1].sort('relative')[-10:]
# On the contrary, names that becames much more famous than they were before 2000

min_non_zero = a[a.relative != 0].relative.min()

a[a.relative.between(min_non_zero, 0.5)].sort('relative')[:10]
# And finally

# My name is a vanishing species

# Save my name

# Name your child after me

# JK, name them Nevaeh or something

a[a.Name=='Sanket']