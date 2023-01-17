# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

t = pd.read_csv('../input/HN_posts_year_to_Sep_26_2016.csv')
t.columns
t['time'] = t['created_at'].apply(lambda i: int(i.split(':')[0].split(' ')[-1]))
g = sb.factorplot(data=t, x='time', y='num_points', kind='bar', size=4, aspect=2)

g.set(ylim=(14,None))
author_group = t.groupby('author')['num_points'].sum()

sorted_author_counts = author_group.sort_values(ascending=False)

rest = sorted_author_counts[10:]

top_10 = sorted_author_counts[0:10].append(pd.Series(rest.mean(), index=['Mean Rest']))

top_10 = pd.DataFrame({'author': top_10.index, 'num_points': top_10.values})
h = sb.factorplot(data=top_10, x='author', y='num_points', kind='bar', size=4, aspect=3)

h.set(yscale='log')