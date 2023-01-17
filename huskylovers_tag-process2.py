# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/data-preprocessing"))



# Any results you write to the current directory are saved as output.
basic_path = "../input/data-preprocessing"

train_path = "/treemodel_train.csv"

test_path = "/treemodel_test.csv"

train_data = pd.read_csv(basic_path+train_path)

test_data = pd.read_csv(basic_path + test_path)



train_label = train_data['label']

del train_data['label']

split_id = len(train_data)



data = train_data.append(test_data)

data.head()
tag_data = pd.read_csv("../input/tag-process/user_tag_K.csv")

tag_data.head()
user_id = data['user_id']

del data['user_id']

del data['listing_id']
from sklearn import manifold

from sklearn.cluster import KMeans
# tsne = manifold.TSNE(n_components=1, init='pca', random_state=2019,n_iter=250)

# X_tsne = tsne.fit_transform(x)

x = np.array(data)

estimator = KMeans(n_clusters=50)#构造聚类器

estimator.fit(x)#聚类
data = pd.DataFrame()

data['user_id'] = user_id

data['tag_label'] = np.array(estimator.labels_)

data.head()
data = pd.merge(data,tag_data,on='user_id',how='left')

data.head()
user_id = data['user_id']

del data['user_id']
import matplotlib.pyplot as plt
tag_data = data.groupby(data['tag_label']).mean()

tag_data['tag_label'] = tag_data.index

tag_data.head()
assist = data['0'].isnull()

for index in range(len(data)):

    if assist[index]:

        tag_ = int(data.iloc[index]['tag_label'])

        data.iloc[index] = tag_data.iloc[tag_]
data['user_id'] = user_id 
data.to_csv('user_tag.csv',index=False)