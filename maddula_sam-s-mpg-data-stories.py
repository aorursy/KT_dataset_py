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
%matplotlib inline



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()
!ls '../input/auto-mpg.csv'
# Read the data

df = pd.read_csv("../input/auto-mpg.csv" )
# Lets have a first look at data

df.head()
# Lets stats of data(no.or records, check for nulls, datatypes, memory usage)

df.info()
mask = df.horsepower.map(lambda x: not str(x).isdigit())



df[mask]
# Dropping the 6 records with no horsepower details.

#   we can fix them using various details but for now, let go with this.



df = df[~ mask]

df.horsepower = df.horsepower.map(int)

df = df.reset_index(drop=True)
tmp = plt.subplots(3, 3, figsize=(16, 16))

tmp = tmp[1].flatten().tolist()

for i, col in enumerate(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',

                         'acceleration', 'model year', 'origin', 'car name']):

    try:

        sns.distplot(df[col], kde=False, ax=tmp[i])

    except:

        print(col)
plt.figure(figsize=(16.5, 4))

sns.distplot(df['car name'].value_counts().values, kde=False)
from collections import Counter

Counter(df['car name'].value_counts().values)
df.columns
# Let check the shape

df.shape
# Lets try to drop any duplicate records are there if any and then check the shape

df.drop_duplicates().shape
### Value Counts - Gives you the uniques values in index and their frequencies as values



(

    df['car name'].map(str) + '----' +\

    df['model year'].map(str) + '----' +\

#     df['acceleration'].map(str) + '----' +\

#     df['cylinders'].map(str) + '----' +\

#     df['weight'].map(str) + '----' +\

#     df['origin'].map(str) + '----' +\

    '>'

).value_counts()

# .head() #.unique()
mask = df['car name'].map(lambda x: x in ['ford pinto', 'plymouth reliant'])



df[mask]
cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']

sns.pairplot(df[cols])
# demo of heatmap

sns.heatmap([[0, 1], [1, 0]])
df['mpg'].corr(df['mpg'])
cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']



# Heat Maps of correlations



df.mpg.corr(-1 * df.mpg), df.mpg.corr(-1 * df.mpg)
cols_corr_matrix = [[df[col].corr(df[_]) for _ in cols]   for col in cols]



cols_corr_matrix = pd.DataFrame(cols_corr_matrix, index=cols, columns=cols)
plt.figure(figsize=(9, 5))

sns.heatmap(cols_corr_matrix)

cols_corr_matrix
# Let check those correlations that can have impact.

#  I am taking a benchmark of .8 => it means something like 80% of time a change in `x` effects changes in `y` 



effective_cols_corr_matrix = cols_corr_matrix.applymap(lambda x: x if ((x < -0.7) or (x > .7)) and (x < .999) else 0)



sns.heatmap(effective_cols_corr_matrix)

effective_cols_corr_matrix
import sklearn.cluster
clf = sklearn.cluster.KMeans(n_clusters=8)

clf = clf.fit(df[cols])



df['cluster'] = clf.predict(df[cols])
tmp = plt.subplots(3, 3, figsize=(16, 16))

tmp = tmp[1].flatten().tolist()

for i, col in enumerate(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',

                         'acceleration',

                         'model year', 'origin',

                         'cluster'

                        ]):

    try:

        for cluter_i in range(len(clf.cluster_centers_)):

            sns.distplot(df[df.cluster == cluter_i][col], hist=True, ax=tmp[i])

    except:

        print(col)
clf = sklearn.cluster.KMeans(n_clusters=3)

clf = clf.fit(df[cols])

df['cluster'] = clf.predict(df[cols])



tmp = plt.subplots(3, 3, figsize=(16, 16))

tmp = tmp[1].flatten().tolist()

for i, col in enumerate(['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',

                         'acceleration',

                         'model year', 'origin',

                         'cluster'

                        ]):

    try:

        for cluter_i in range(len(clf.cluster_centers_)):

            sns.distplot(df[df.cluster == cluter_i][col], hist=True, ax=tmp[i], kde=False)

    except:

        print(col)