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
data_train = pd.read_csv('../input/train.csv')
x_test = pd.read_csv('../input/test.csv')
X_train = np.array(data_train[['pixel' + str(i) for i in range(0,784)]])
y_train = np.array(data_train['label'])
X_test = np.array(x_test[['pixel' + str(i) for i in range(0,784)]])
pca = PCA(n_components=30, whiten=True)
pca.fit(X_train)
x_train = pca.transform(X_train)
X_test = pca.transform(X_test)
clf = SVC()
clf.fit(x_train, y_train)
y = clf.predict(X_test)
submission = pd.DataFrame()
submission['ImageId'] = [i for i in range(1, len(x_test)+1)]
submission['Label'] = y
submission.to_csv('submission2.csv', index=False)