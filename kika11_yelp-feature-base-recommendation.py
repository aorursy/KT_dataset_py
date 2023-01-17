# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!head -10 /kaggle/input/yelp-dataset/yelp_academic_dataset_business.json
import json
from tqdm import tqdm

from collections import defaultdict
filename = '/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json'
reviews = []
target_user_ids = dict()
target_item_ids = dict()

with open(filename, 'rt') as f:
    with tqdm(total=8021122) as pbar:
        for line in f:
            data = json.loads(line)

            if not 'date' in data:
                continue
            
            dt = data.get('date').split()[0]
        
            if dt.startswith('2019-12'):
                reviews.append({
                    key: data[key]
                    for key in ['review_id', 'user_id', 'business_id', 'stars']
                })
                target_user_ids[data['user_id']] = True
                target_item_ids[data['business_id']] = True
            pbar.update(1)

filename = '/kaggle/input/yelp-dataset/yelp_academic_dataset_business.json'
items = list()

with open(filename, 'rt') as f:
    with tqdm(total=209393) as pbar:
        for line in f:
            data = json.loads(line)
            
            if data['business_id'] in target_item_ids:
                items.append(data)
            pbar.update(1)
filename = '/kaggle/input/yelp-dataset/yelp_academic_dataset_user.json'

users = list()

with open(filename, 'rt') as f:
    with tqdm(total=1968703) as pbar:
        for line in f:
            data = json.loads(line)
            
            if data['user_id'] in target_user_ids:
                users.append(data)
            pbar.update(1)
import gc

del target_user_ids
del target_item_ids
gc.collect()
users[0]
user_df = pd.DataFrame(users)
item_df = pd.DataFrame(items)
review_df = pd.DataFrame(reviews)
user_df
item_df.categories
print(item_df.categories)
category_arrays = item_df.categories.apply(lambda x : x.split(', ') if x is not None else [])

counter = defaultdict(int)
for category_array in category_arrays:
    for category in category_array:
        counter[category] += 1
        
counter

sorted_arrays = sorted(counter.items(), key=lambda x:x[1], reverse=True)
target_categories = list(map(lambda x:[x[0]], sorted_arrays[0:5]))

from sklearn.preprocessing import MultiLabelBinarizer
# binarizer = MultiLabelBinarizer().fit(target_categories)
binarizer = MultiLabelBinarizer().fit(category_arrays)
binarizer.classes_
from sklearn.decomposition import PCA
pca = PCA(n_components=25)
#item_category_df = pd.DataFrame(binarizer.transform(category_arrays.values), columns=binarizer.classes_)

item_category_df = pd.DataFrame(pca.fit_transform(binarizer.transform(category_arrays.values)), columns=["pca"+str(i) for i in range(25)])

item_category_df
review_df
review_df['y'] = review_df.stars #.apply(lambda star: 1 if star >= 4 else 0)
review_df
user_df.columns
user_df
f_user_df = user_df.drop(['yelping_since', 'elite', 'friends', 'name'], axis=1)
f_user_df.shape
f_item_df = item_df.drop(['name', 'address', 'city', 'state', 'postal_code', 'attributes', 'categories', 'hours'], axis=1)
f_item_df = pd.concat([f_item_df, item_category_df], axis=1)
f_item_df.shape

f_review_df = review_df.drop(['stars'], axis=1)
f_review_df.shape
merge_df = pd.merge(f_review_df, f_user_df, on='user_id')
merge_df = pd.merge(merge_df, f_item_df, on='business_id')
merge_df = merge_df.drop(['review_id', 'user_id', 'business_id'], axis=1)
!pip install mord==0.6
X = merge_df.drop(['y'], axis=1).values
Y = merge_df.y.values.astype(np.int64)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mord import LogisticAT
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print(X_train.shape)
print(X_test.shape)

model = RandomForestClassifier(n_jobs=-1)
l_model = LogisticRegression(max_iter=1000, n_jobs=-1, verbose=1)
at_model = LogisticAT()
l_model.fit(X_train, Y_train)
model.fit(X_train, Y_train)
at_model.fit(X_train, Y_train)
print(len(Y_test))
print(sum(Y_test))
print(sum(Y_test)/len(Y_test))
print(len(Y_train))
print(sum(Y_train))
print(sum(Y_train)/len(Y_train))
print("random:      ", model.predict(X_test[:15]))
print("logistic:    ", l_model.predict(X_test[:15]))
print("logistic_at: ", at_model.predict(X_test[:15]))
print("Y:           ", Y_test[:15])
print("random:  ", model.predict_proba(X_test[:15]))
print("logistic:", l_model.predict_proba(X_test[:15]))
#
model.feature_importances_
merge_columns = merge_df.drop('y',axis=1).columns

for column, importance in zip(merge_columns, model.feature_importances_):
    print(column, importance)
l_model.coef_
merge_columns = merge_df.drop('y',axis=1).columns

for column, coef in zip(merge_columns, l_model.coef_[0]):
    print(column, coef)
model.predict(X_test[:10])
Y_test[:10]
model.predict_proba(X_test[:10])
at_model.predict(X_test[:10])
Y_test[:10]
at_model.predict_proba(X_test[:10])
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, model.predict(X_test))
