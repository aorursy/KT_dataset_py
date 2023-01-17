import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





train = pd.read_csv('../input/dips-recco-files/train_dips.csv')

test = pd.read_csv('../input/dips-recco-files/recommendations_dips.csv',header=None)

test.columns = ['uid']

train.head()
len(train.user_id.unique())
mat = train.pivot(index='user_id',columns='movie_id',values='rating')

mat.head()
ratings = pd.DataFrame(train.groupby('movie_id').mean().rating)

ratings['no']=pd.DataFrame(train.groupby('movie_id').rating.count())

ratings.sort_values('no',ascending=False,inplace=True)

ratings.head()
mat = train.pivot_table(index='user_id',columns='movie_id',values='rating')

mat = mat.fillna(0)
mat.head()
sim = mat.corr(method='pearson')

sim.head()
def get_similar(mid,r):

    score = sim[mid]*(r-2.5)

    score = score.sort_values(ascending=False)

    return score

mid = train[train['user_id'] == 1][['movie_id','rating','movie title']].reset_index()

mov = pd.DataFrame()

for i in range(len(mid)):

    mov = mov.append(get_similar(mid.loc[i, "movie_id"], mid.loc[i, "rating"]),ignore_index=True)

#mov.sum().sort_values(ascending=False)
final = pd.DataFrame()

for j in range(len(test)):    

    mid = train[train['user_id'] == test.loc[j, "uid"]][['movie_id','rating','movie title']].reset_index()

    mov = pd.DataFrame()

    for i in range(len(mid)):

        mov = mov.append(get_similar(mid.loc[i, "movie_id"], mid.loc[i, "rating"]),ignore_index=True)

    films = mov.sum().sort_values(ascending=False).reset_index()

    films.columns = ['movie_id','rating']

    df = films.merge(mid,on='movie_id',how='left')

    df2 = df[df.rating_y.isna()]

    s = df2.movie_id.head(20).values

    s= np.insert(s,0,test.loc[j, "uid"])

    final = final.append(pd.Series(s),ignore_index=True)

    print(j)

final
final2 = final.astype(int)

final2 = final2.drop([10,11,12,13,14,15,16,17,18,19,20],axis=1)

final2
print(final2.to_csv(index = False))
test.head(1100)