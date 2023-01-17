# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report



import cv2

import glob
%matplotlib inline

import matplotlib.pyplot as plt
path="../input/"



train_path=path+"train/train/"

test_path=path+"test/test/"
sorted(os.listdir(train_path));
train_set = pd.read_csv(path + '/train.csv').sort_values('id')

train_set.sort_values('id');
train_labels = train_set['has_cactus']

train_labels;
files = sorted(glob.glob(train_path + '*.jpg'))

files;
train = [cv2.imread(image) for image in files]
train = np.array(train, dtype='int32')
train_images_set = np.reshape(train,[train.shape[0],train.shape[1]*train.shape[2]*train.shape[3]])
X_train, X_test, y_train, y_test = train_test_split(train_images_set,

                                                    train_labels,

                                                    test_size=0.30,

                                                    random_state=42)
pca = PCA(n_components=3000, whiten=True,

          svd_solver='randomized', random_state=42)

pca_fitted = pca.fit(X_train)

pca_fitted;

plt.plot(np.cumsum(pca_fitted.explained_variance_ratio_))
pca = PCA(n_components=2100, whiten=True,

          svd_solver='randomized', random_state=42)

svc = SVC(kernel='rbf', class_weight='balanced', probability=True, C=5, gamma=0.0005)

model = make_pipeline(pca, svc)



model.fit(X_train, y_train)



y_fit = model.predict(X_test)
print(classification_report(y_test, y_fit))
err_train = np.mean(y_train != model.predict(X_train))

err_test  = np.mean(y_test  != model.predict(X_test))

print(err_train, err_test)

print(model.score(X_train, y_train))
# prepare test set

test_files = sorted(glob.glob(test_path + '*.jpg'))

test = [cv2.imread(image) for image in test_files]

test = np.array(test, dtype='int32')

test_images_set = np.reshape(test,[test.shape[0],test.shape[1]*test.shape[2]*test.shape[3]])
y_fit_test_set = model.predict_proba(test_images_set)

y_fit_test_set[:10]
test_data = pd.read_csv(path + 'sample_submission.csv')

test_data[:10]
test_data['has_cactus'] = y_fit_test_set

test_data[:10]
test_data.to_csv('sample_submission.csv',index=False)
pd.read_csv('sample_submission.csv').shape