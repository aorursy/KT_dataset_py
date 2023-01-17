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
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
train, test = mnist.load_data()
distrain = pd.DataFrame(data=train[1], columns=['Label'])
_ = distrain['Label'].value_counts().plot(kind='bar')
plt.show()
import random
from matplotlib import pyplot as plt
rand = random.randint(0,60000)
plt.imshow(train[0][rand])
print(train[0].shape, train[1].shape)
print(test[0].shape, test[1].shape)
train_x = train[0].reshape(60000,784)
test_x = test[0].reshape(10000,784)
train_y = train[1]
test_y = test[1]
print(train_x.shape, test_x.shape)
print(train_x[20])
from sklearn import preprocessing
train_x = preprocessing.scale(train_x)
print(train_x[20])

model = LogisticRegression()
model.fit(train_x, train_y)
pre = model.predict(test_x)

print(classification_report(test_y, pre))
print(confusion_matrix(test_y, pre))
print(accuracy_score(test_y, pre))
from sklearn.decomposition import PCA
pca = PCA(0.95)
pca.fit(train_x)
print(pca.n_components_)
train = pca.transform(train_x)
test = pca.transform(test_x)
model = LogisticRegression()
model.fit(train,train_y)
pre = model.predict(test)
print(classification_report(test_y, pre))
print(confusion_matrix(test_y, pre))
print(accuracy_score(test_y, pre))