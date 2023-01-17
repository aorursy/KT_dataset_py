import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/insta_train.csv')

test = pd.read_csv('../input/insta_test.csv')
train.head()
def ing(df):

    df['fol'] = df['followers']/(df['followings']+1)

    df['act'] = df['comments']/(df['followers']+1)

    df['pos'] = df['posts']/(df['followers']+1)

    df['hashtag_max'] = df.groupby('user')['hashtag'].transform('max')

    df['hashtag_min'] = df.groupby('user')['hashtag'].transform('min')

    df['hashtag_mean'] = df.groupby('user')['hashtag'].transform('mean')

    df['hashtag_std'] = df.groupby('user')['hashtag'].transform('std')

    df['comments_max'] = df.groupby('user')['comments'].transform('max')

    df['comments_min'] = df.groupby('user')['comments'].transform('min')

    df['comments_mean'] = df.groupby('user')['comments'].transform('mean')

    df['comments_std'] = df.groupby('user')['comments'].transform('std')

    return df
y = train.likes.values
train = ing(train).select_dtypes(exclude=['object']).drop(['likes'], axis=1).fillna(-999).values

test = ing(test).select_dtypes(exclude=['object']).fillna(-999).values
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor
knn = BaggingRegressor(KNeighborsRegressor(n_neighbors = 50, #кількість сусідів

                          metric='hamming', #метод (обираємо цей, бо маємо переважно категорії)

                          n_jobs=-1),random_state=42, max_features=0.9) #кількість ядер для паралелізаціїv
knn.fit(train, y)
prediction = pd.read_csv('../input/sample_submission.csv')



prediction['likes'] = knn.predict(test)



prediction.to_csv('my_submission.csv', index=False)