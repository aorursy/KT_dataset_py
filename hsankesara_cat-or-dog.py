# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dogs_train_files = os.listdir('../input/dog vs cat/dataset/training_set/dogs')
dogs_train_files_size = len(dogs_train_files)
print(dogs_train_files_size)
limit = 1000
dogs_training_data = [None] * limit
j = 0
for i in dogs_train_files:
    if j < limit:
        dogs_training_data[j] = cv2.imread('../input/dog vs cat/dataset/training_set/dogs/' + i, cv2.IMREAD_GRAYSCALE)
        j += 1
    else:
        break
dogs_training_data = np.array(dogs_training_data)
dogs_training_data[0].shape
import matplotlib.pyplot as plt
plt.imshow(dogs_training_data[0])
plt.plot()
cats_train_files = os.listdir('../input/dog vs cat/dataset/training_set/cats')
cats_train_files_size = len(cats_train_files)
print(cats_train_files_size)
cats_training_data = [None] * limit
j = 0
for i in cats_train_files:
    if j < limit:
        cats_training_data[j] = cv2.imread('../input/dog vs cat/dataset/training_set/cats/' + i, cv2.IMREAD_GRAYSCALE)
        j += 1
    else:
        break
cats_training_data = np.array(cats_training_data)
plt.imshow(cats_training_data[123])
plt.show()
s00 = 0
s01 = 0
s10 = float('inf')
s11 = float('inf')
for i in range(limit):
    s00 = max(s00, dogs_training_data[i].shape[0])
    s01 = max(s01, dogs_training_data[i].shape[1])
    s10 = min(s10, dogs_training_data[i].shape[0])
    s11 = min(s11, dogs_training_data[i].shape[1])
print(s00, s01)
print(s10, s11)
s00 = 0
s01 = 0
s10 = float('inf')
s11 = float('inf')
for i in range(limit):
    s00 = max(s00, cats_training_data[i].shape[0])
    s01 = max(s01, cats_training_data[i].shape[1])
    s10 = min(s10, cats_training_data[i].shape[0])
    s11 = min(s11, cats_training_data[i].shape[1])
print(s00, s01)
print(s10, s11)
img_size = (256, 256)
i = dogs_training_data[53]
plt.imshow(i)
plt.show()
l = img_size[0] - i.shape[0]
w = img_size[1] - i.shape[1]
up = l // 2
lo = l - up
wdl = w // 2
wdr = w - wdl
image = cv2.resize(i, img_size)
print(image.shape)
plt.imshow(image)
plt.plot()
m1, m2 = img_size
for j in range(limit):
    i = dogs_training_data[j]
    dogs_training_data[j] = cv2.resize( i, img_size)
    
s = (0,0)
s2 = (702, 1050)
for i in range(limit):
    s = max(s, dogs_training_data[i].shape)
    s2 = min(s2, dogs_training_data[i].shape)
print(s)
print(s2)
m1, m2 = img_size
for j in range(limit):
    i = cats_training_data[j]
    cats_training_data[j] = cv2.resize( i, img_size)
    
s = (0,0)
s2 = (702, 1050)
for i in range(limit):
    s = max(s, cats_training_data[i].shape)
    s2 = min(s2, cats_training_data[i].shape)
print(s)
print(s2)
j = 0
sum = 0
for i in cats_training_data:
    if i.shape != img_size:
        print(j)
        sum += 1
    j += 1
j = 0
sum = 0
for i in dogs_training_data:
    if i.shape != img_size:
        print(j)
        sum += 1
    j += 1

flatten_size = img_size[0] * img_size[1]
m = len(dogs_training_data)
for i in range(m):
    dogs_training_data[i] = np.ndarray.flatten(dogs_training_data[i]).reshape(flatten_size, 1)

dogs_training_data = np.dstack(dogs_training_data)
dogs_training_data.shape
dogs_training_data = np.rollaxis(dogs_training_data, axis=2, start=0)
dogs_training_data.shape
m = len(cats_training_data)
for i in range(m):
    cats_training_data[i] = np.ndarray.flatten(cats_training_data[i]).reshape(flatten_size, 1)
cats_training_data = np.dstack(cats_training_data)
cats_training_data.shape
cats_training_data = np.rollaxis(cats_training_data, axis=2, start=0)
cats_training_data.shape
dogs_training_data.shape
dogs_training_data = dogs_training_data.reshape(m, flatten_size)
cats_training_data = cats_training_data.reshape(m, flatten_size)
dogs_training_data = pd.DataFrame(dogs_training_data)
cats_training_data = pd.DataFrame(cats_training_data)
dogs_training_data['is_cat'] = pd.Series(np.zeros(m), dtype=int)
dogs_training_data.head()
cats_training_data['is_cat'] = pd.Series(np.ones(m), dtype=int)
cats_training_data.head()
df = pd.concat([dogs_training_data, cats_training_data])
df.head()

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
df = shuffle(df).reset_index()
df.head()
df = df.drop(['index'], axis = 1)
df.head()

df.info()
df_train, df_test = train_test_split(df, test_size = 0.15)
df_train, df_validation = train_test_split(df_train, test_size = 0.176)
df_train.info()
df_validation.info()
from sklearn.linear_model import LogisticRegression as Lr
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.ensemble import GradientBoostingClassifier as Gbc
from xgboost import XGBClassifier as Xgb
df_train_y = df_train['is_cat']
df_train_x = df_train.drop(['is_cat'], axis = 1)
df_train_x.head()
lr = Lr()
lr.fit(df_train_x, df_train_y)
lr.score(df_train_x, df_train_y)
df_validation_y = df_validation['is_cat']
df_validation_x = df_validation.drop(['is_cat'], axis = 1) 
lr.score(df_validation_x, df_validation_y)
rfc = Rfc()
rfc.fit(df_train_x, df_train_y)
rfc.score(df_train_x, df_train_y)
plt.plot(rfc.feature_importances_)
plt.show
rfc.score(df_validation_x, df_validation_y)
xgb = Xgb()
xgb.fit(df_train_x, df_train_y)
xgb.score(df_train_x, df_train_y)
plt.plot(xgb.feature_importances_)
plt.show
xgb.score(df_validation_x, df_validation_y)
df_train_x = df_train_x.reset_index()
df_train_y = df_train_y.reset_index()
df_train_x = df_train_x.drop(['index'], axis = 1)
df_train_y = df_train_y.drop(['index'], axis = 1)
df_train_x.head()
def max_vote(x):
    s1 = lr.predict(x)
    s2 = rfc.predict(x)
    s3 = xgb.predict(x)
    result = pd.DataFrame(s1 + s2 + s3, columns=['is_cat'])
    result['is_cat'] = pd.Series(np.where(result['is_cat'] > 1, 1, 0))
    return result
result_maxvote = max_vote(df_train_x)
result_maxvote.head()
def check_accuracy(result, output):
    total = result.shape[0]
    true = (result.is_cat == output.is_cat).sum()
    print(true)
    print(true / total)
    return (true / total)
check_accuracy(result_maxvote, df_train_y)
result_validation =  max_vote(df_validation_x)
check_accuracy(result_validation, df_validation_y)
from sklearn.decomposition import PCA
pca = PCA(n_components=512)
pca.fit(df_train_x)
plt.plot(pca.explained_variance_ratio_)
plt.show()
plt.plot(pca.singular_values_)
plt.show()
print(np.sum(pca.explained_variance_ratio_))
df_train_pca_x = pd.DataFrame(pca.transform(df_train_x))
df_validation_pca_x = pd.DataFrame(pca.transform(df_validation_x))
df_train_pca_x.head()
lr_pca = Lr()
lr_pca.fit(df_train_pca_x, df_train_y, )
lr_pca.score(df_train_pca_x, df_train_y)
lr_pca.score(df_validation_pca_x, df_validation_y)
from sklearn.svm import LinearSVC as SVC
svc = SVC()
svc.fit(df_train_pca_x, df_train_y)
svc.score(df_train_pca_x, df_train_y)
svc.score(df_validation_pca_x, df_validation_y)
