import pandas as pd

import numpy as np

from tqdm import tqdm_notebook
import warnings

warnings.filterwarnings('ignore')



contacts = pd.read_table('../input/user_contacts-timestamps.dat')

tags = pd.read_table('../input/tags.dat', encoding='koi8-r')

bookmark_tags = pd.read_table('../input/bookmark_tags.dat')

bookmarks = pd.read_table('../input/bookmarks.dat', encoding='koi8-r')

user_taggedbookmarks = pd.read_table('../input/user_taggedbookmarks-timestamps.dat')
contacts.head(3) # отношения между пользователями в базе данных и время
tags.head(3) #  набор тегов 
bookmark_tags.head(3) # теги, назначенные для закладок URL-адресов, а также количество назначений 
bookmarks.head(3) # информация о закладках URL
user_taggedbookmarks.head(3) # назначения тегов URL-адресов с закладками, каждым конкретным пользователем.
user_taggedbookmarks = user_taggedbookmarks.drop('timestamp', axis=1)

user_taggedbookmarks.head(3)
user_taggedbookmarks = user_taggedbookmarks.dropna()
user_taggedbookmarks['act'] = 1

user_taggedbookmarks = user_taggedbookmarks.drop('tagID', axis=1)
user_taggedbookmarks = user_taggedbookmarks[['userID', 'bookmarkID', 'act']].drop_duplicates()

user_taggedbookmarks.shape
user_taggedbookmarks = user_taggedbookmarks.merge(bookmarks[['id', 'title']], left_on='bookmarkID', right_on='id')

user_taggedbookmarks.head(3)
user_taggedbookmarks['user_id'] = user_taggedbookmarks['userID'].astype("category").cat.codes

user_taggedbookmarks['bookmark_id'] = user_taggedbookmarks['bookmarkID'].astype("category").cat.codes

user_taggedbookmarks = user_taggedbookmarks.drop(['userID', 'bookmarkID'], axis=1)
descriptions_table = user_taggedbookmarks[['bookmark_id', 'title']].drop_duplicates().reset_index(drop=True)

descriptions_table.head(3)
dict_recomendation = {}



for  index, value in tqdm_notebook(descriptions_table.iterrows()):

    dict_recomendation[value['bookmark_id']] = value['title']
user_taggedbookmarks = user_taggedbookmarks.drop(['title', 'id'], axis=1)
activity = list(user_taggedbookmarks['act'])

cols = user_taggedbookmarks['bookmark_id'].astype(int)

rows = user_taggedbookmarks['user_id'].astype(int)
shape_0 = len(user_taggedbookmarks['bookmark_id'].unique())

shape_1 = len(user_taggedbookmarks['user_id'].unique())
user_taggedbookmarks.head(3)
user_taggedbookmarks[user_taggedbookmarks['bookmark_id']==1591]
len(rows), len(activity), len(cols)
shape_0, shape_1
from scipy import sparse

data_sparse = sparse.csr_matrix((activity, (rows, cols)), shape=(shape_1, shape_0))
from implicit.als import AlternatingLeastSquares

model = AlternatingLeastSquares(factors=50)

model.fit(data_sparse)
userid = 27



user_items = data_sparse.T.tocsr()

recommendations = model.recommend(userid, user_items)
recommendations
for i in recommendations:

    print(dict_recomendation[int(i[0])])