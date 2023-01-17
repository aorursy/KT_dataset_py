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
import pandas as pd

import numpy as np

import cv2

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sys import getsizeof
train = pd.read_csv('../input/ava/train.csv', index_col=0)

test = pd.read_csv('../input/ava/test.csv', index_col=0)

train
X_train = train['image'].values

y_train = train['label'].values

X_test = test['image'].values
shapes = []

for i in range(len(X_train)):

    path = '../input/ava/dataset/dataset/'+str(X_train[i])+'.jpg'

    img = cv2.imread(path)

    shapes.append(img.shape)

shapes = np.array(shapes[:])

print(np.mean(shapes[:,0]), np.mean(shapes[:,1]), np.mean(shapes[:,2]))
img_number = np.random.randint(len(X_train))

path = '../input/ava/dataset/dataset/'+str(X_train[img_number])+'.jpg'

img = cv2.imread(path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img, (500, 500), interpolation = cv2.INTER_AREA)

fig, m_axs = plt.subplots(1, 2, figsize=(12, 5))

ax1, ax2 = m_axs



ax1.set_title('Original Image')

ax1.imshow(img, cmap='gray')

ax2.set_title('resized '+ str(y_train[img_number]))

ax2.imshow(img2);
def get_feature(img):

    img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)

    return np.array(img).flatten()
def create_features_data(image_names):

    FEATURES = []

    for i in range(len(image_names)):

        path = '../input/ava/dataset/dataset/'+str(image_names[i])+'.jpg'

        img = cv2.imread(path)

        FEATURES.append(get_feature(img))

    return np.array(FEATURES)

features_train = create_features_data(X_train)

features_test = create_features_data(X_test)
%%time

lr = LogisticRegression()

lr.fit(features_train,y_train)
# accuracy_score(lr.predict(features_test), y_test)
prediction = pd.DataFrame()

prediction['labels'] = lr.predict(features_test)
prediction
prediction.to_csv("submittion.csv", index_label='id')