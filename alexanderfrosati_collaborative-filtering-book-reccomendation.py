import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from surprise import Dataset

from surprise import Reader



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/goodbooks-10k-updated/ratings.csv')

df1 = pd.read_csv('../input/goodbooks-10k-updated/personalLibrary.csv')



print("Dataset contains {} rows and {} columns".format(df.shape[0], df.shape[1]))

print(df['user_id'].describe())



df = df.append(df1)

print("Dataset contains {} rows and {} columns".format(df.shape[0], df.shape[1]))

df['user_id'].describe()

book_matrix = df.pivot_table(index='book_id', columns='user_id', values='rating')

book_matrix.head()
rosatiRating = book_matrix[53424]

rosatiRating.head()
similar = book_matrix.corrwith(rosatiRating)



corrUser = pd.DataFrame(similar, columns=['Correlation'])

corrUser.dropna(inplace=True)

corrUser.head()
len(corrUser)

corrUser.sort_values(by=['Correlation'], ascending=False).head(1040)
df.loc[df['user_id'] == 15264]
stuff = pd.read_csv('../input/goodbooks-10k-updated/books.csv')



books = stuff['original_title'].tolist()

ids = stuff['book_id'].tolist()
replacements = dict(zip(ids,books))

oneguy = df.loc[df['user_id'] == 15264]

oneguy['book_id'].replace(replacements, inplace=True)

oneguy.head(12)
otherguy = df.loc[df['user_id'] == 402]

otherguy['book_id'].replace(replacements, inplace=True)

otherguy.head(12)