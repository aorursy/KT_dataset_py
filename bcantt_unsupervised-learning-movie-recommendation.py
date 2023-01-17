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
movies = pd.read_csv('/kaggle/input/movietweetings/movies.dat', delimiter='::', engine='python', header=None, names = ['Movie ID', 'Movie Title', 'Genre'])

users = pd.read_csv('/kaggle/input/movietweetings/users.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Twitter ID'])

ratings = pd.read_csv('/kaggle/input/movietweetings/ratings.dat', delimiter='::', engine='python', header=None, names = ['User ID', 'Movie ID', 'Rating', 'Rating Timestamp'])
movies.head()

users
data =  users.merge(movies.merge(ratings, how = 'inner',on='Movie ID'),how = 'inner',on = 'User ID')
table_1 = data.groupby(['User ID','Movie Title','Genre']).count().sort_values(['Rating','User ID'],ascending=False)
table_1.index.values[0]
table_1['User ID'] = [i[0] for i in table_1.index.values]

table_1['Movie Title'] = [i[1] for i in table_1.index.values]

table_1['Genre'] = [i[2] for i in table_1.index.values]
table_1 = table_1.reset_index(drop=True)
table_1
table_2 = data.groupby(['User ID','Movie Title','Genre']).mean()
table_2['User ID'] = [i[0] for i in table_2.index.values]

table_2['Movie Title'] = [i[1] for i in table_2.index.values]

table_2['Genre'] = [i[2] for i in table_2.index.values]
table_2 = table_2.reset_index(drop=True)
table_2
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for name in table_1.columns:

    if table_1[name].dtypes == "O":

        le.fit(table_1[name])

        table_1[name+'_encoded'] = le.transform(table_1[name])

        table_2[name+'_encoded'] = le.transform(table_2[name])
table_1
table_2
table_1_unsupLabels = table_1[['Rating','Movie Title_encoded','Genre_encoded']]

table_2_unsupLabels = table_2[['Rating','Movie Title_encoded','Genre_encoded']]
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10).fit(table_1_unsupLabels)



table_1['Labels'] = kmeans.labels_
kmeans = KMeans(n_clusters=10).fit(table_2_unsupLabels)



table_2['Labels'] = kmeans.labels_
print(table_1.shape[0])

print(table_2.shape[0])
table_1 = table_1.sort_values(['Movie Title_encoded','User ID'])
table_2 = table_2.sort_values(['Movie Title_encoded','User ID'])
table_1['table_2_Labels'] = table_2.Labels
table_1
table_1.loc[(table_1['Labels'] == 2) & (table_1['table_2_Labels'] == 1)]
for i in table_1['User ID'].unique():

    table_1.loc[table_1['User ID'] == i,'Users_Most_Category_1'] = table_1.loc[table_1['User ID'] == 104]['Labels'].value_counts().index[0]
for i in table_1['User ID'].unique():

    table_1.loc[table_1['User ID'] == i,'Users_Most_Category_2'] = table_1.loc[table_1['User ID'] == 104]['table_2_Labels'].value_counts().index[0]
table_1