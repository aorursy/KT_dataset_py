# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import numpy as np

import cv2 

import os

import pandas as pd



from sklearn.svm import SVC

from sklearn.decomposition import PCA as PCA

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline



# Any results you write to the current directory are saved as output.
dataset_train = "../input/2019-fall-pr-project/train/train/"

train_data = [dataset_train + i for i in os.listdir(dataset_train)]
images = []  #images

labels = []  #labels



for i in train_data:

  image = cv2.imread(i)

  image = cv2.resize(image, (32,32))

  images.append(image)



for i in train_data:

  if 'cat' in i:

    labels.append(0)

  elif 'dog' in i:

    labels.append(1)
X = np.array(images)

y = np.array(labels)

X = X.reshape(20001, -1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
pca = PCA(n_components=150, whiten=True, random_state=42)

svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)



param_grid = {'svc__C': [1, 5, 10, 50],

              'svc__gamma': [0.0005, 0.001, 0.005, 0.01]}

grid = GridSearchCV(model, param_grid)



%time grid.fit(X_train, y_train)

print(grid.best_params_)
dataset_test = "../input/2019-fall-pr-project/test1/test1/"

test_data = [dataset_test + i for i in os.listdir(dataset_test)]
images = []

labels = []



for i in test_data:

  image = cv2.imread(i)

  image = cv2.resize(image, (32,32))

  images.append(image)



for i in test_data:

  if 'cat' in i:

    labels.append(0)

  elif 'dog' in i:

    labels.append(1)
X_test = np.array(images)

y_test = np.array(labels)

X_test = X_test.reshape(5000, -1)
model = grid.best_estimator_

result = model.predict(X_test)
# numpy 를 Pandas 이용하여 결과 파일로 저장



import pandas as pd



print(result.shape)

df = pd.DataFrame(data=result, index=range(1,5001), columns=['label'])

df = df.replace('dog',1)

df = df.replace('cat',0)



df.to_csv('result.csv',index=True, header=True, index_label='id')