# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for visualization
import seaborn as sns # personal choise for viz

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print('List of files in the input directory:')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Set seaborn
sns.set()
# Load train and test dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Inspect the shape of the datasets
print('Shape of train dataset - Rows: %d; Columns: %d' % train.shape)
print('Shape of test dataset - Rows: %d; Columns: %d' % test.shape)
print('Training dataset columns:')
print(train.columns)
print('\nTesting dataset columns:')
print(test.columns)
print('8 arbitrary columns of first 5 records in the training dataset:')
print(train[['label', 'pixel0', 'pixel234', 'pixel75', 'pixel387', 'pixel412', 'pixel111', 'pixel782', 'pixel623']].head())
print('8 arbitrary columns of first 5 records in the testing dataset:')
print(test[['pixel0', 'pixel234', 'pixel75', 'pixel387', 'pixel412', 'pixel111', 'pixel782', 'pixel623']].head())
print('Training dataset - describe:')
print(train[['label', 'pixel0', 'pixel234', 'pixel75', 'pixel387', 'pixel412', 'pixel111', 'pixel782', 'pixel623']].describe())
print('\nTesting dataset - describe:')
print(test[['pixel0', 'pixel234', 'pixel75', 'pixel387', 'pixel412', 'pixel111', 'pixel782', 'pixel623']].describe())
# Are there any NaNs?
print('No. of null values in the training dataset: %d' % np.sum(np.sum(train.isnull())))
print('No. of null values in the testing dataset: %d' % np.sum(np.sum(test.isnull())))
sns.countplot(train['label'])
plt.show()
# pick 8 random digits in training dataset
np.random.seed(12345)
idx = np.random.randint(0, len(train), 8)

for n, i in enumerate(idx):
    ax = plt.subplot(2, 4, n+1)
    tmp = train.iloc[i]
    lbl = tmp['label']
    tmp = tmp.drop(['label'])
    tmp = tmp.values.reshape(28, 28)
    ax.imshow(tmp, cmap='gray', interpolation='gaussian')
    ax.set_title('Label: %d' % lbl)
    ax.axis('off')

plt.tight_layout()
plt.show()
# pick 4 random digits in testing dataset
np.random.seed(12345)
idx = np.random.randint(0, len(test), 4)

for n, i in enumerate(idx):
    ax = plt.subplot(2, 2, n+1)
    tmp = test.iloc[i]
    tmp = tmp.values.reshape(28, 28)
    ax.imshow(tmp, cmap='gray', interpolation='gaussian')
    ax.set_title('Label: ???')
    ax.axis('off')

plt.tight_layout()
plt.show()
# import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# split the training dataset to check for model performance
X = train.drop(['label'], axis=1)
y = train['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

knn = KNeighborsClassifier(n_neighbors=5)
_ = knn.fit(X_train, y_train)
print('The accurancy of the model: %f' % knn.score(X_test, y_test))
# make prediction on the set provided
submitLabel = knn.predict(test)
submitLabelDF = pd.DataFrame({'ImageId': range(1, len(submitLabel) + 1), 'Label': submitLabel})

# Let us print the first few lines of the submission file
print(submitLabelDF.head())
# write the submission file
submitLabelDF.to_csv('submission.csv', index=False)