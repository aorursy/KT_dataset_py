# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
tqdm.pandas()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
MAIN_PATH = "../input/plant-pathology-2020-fgvc7"
SUB_PATH = MAIN_PATH + '/sample_submission.csv'
print(os.listdir(MAIN_PATH))

IMAGE_PATH = MAIN_PATH + "/images/"
train_df = pd.read_csv(MAIN_PATH + "/train.csv")
test_df = pd.read_csv(MAIN_PATH + "/test.csv")
# 이미지 갯수, csv shape 확인
print('train data shape: ', train_df.shape)
print('Total images in train set: ', train_df['image_id'].count())
print('[train_csv example]\n', train_df.head(3))
print('---------------------------------------------------')
print('test data shape: ', test_df.shape)
print('Total images in test set: ', train_df['image_id'].count())
print('[test_csv example]\n', test_df.head(3))
# 분류되지 않은 데이터 확인
print('train set')
print(train_df.info())
print('------------------------------------------')
print('test set')
print(test_df.info())
# class 추출
temp=[]
classes = {}
for col in train_df.columns:
    temp.append(col) 
temp.remove('image_id')
for i in range(len(temp)):
    classes[i] = temp[i]
# 클래스 별 샘플 수 체크
for c in range(0,len(classes)):
    print(f"#{classes[c]} samples: {train_df[classes[c]].sum()}")
# 중복된 데이터 확인
train_id = set(train_df.image_id.values)
print(f"#Unique train images: {len(train_id)}")
test_id = set(test_df.image_id.values)
print(f"#Unique train images: {len(test_id)}")
both_images = train_id.intersection(test_id)
print(f"#Images in both train set and test set: {len(both_images)}")
def load_image(image_id):
    image = cv2.imread(IMAGE_PATH + image_id + '.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (40, 40))
    return image.flatten().astype(np.float32)

train_X_flatten = train_df['image_id'].progress_apply(load_image)
test_X_flatten = test_df['image_id'].progress_apply(load_image)
train_X_flatten =np.stack(train_X_flatten.to_numpy())
test_X_flatten = np.stack(test_X_flatten.to_numpy())
train_X_flatten = train_X_flatten / 255.
test_X_flatten = test_X_flatten / 255.
train_Y = train_df[['healthy', 'multiple_diseases', 'rust', 'scab']].to_numpy()
train_Y = train_Y[:, 0] + train_Y[:, 1]*2 + train_Y[:, 2]*3 + train_Y[:, 3]*4 - 1
print(train_Y)
print(f"Train set(flatten) shape: {train_X_flatten.shape}")
print(f"Test set(flatten) shape: {test_X_flatten.shape}")

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train_X_std, train_Y)
score = neigh.score(train_X_std, train_Y)
print(f"Model accuracy: {score}")
Y_hat = neigh.predict_proba(test_X_std)
sub = pd.read_csv(SUB_PATH)
sub.loc[:, 'healthy':] = Y_hat
sub.to_csv('submission.csv', index=False)
sub.head()
