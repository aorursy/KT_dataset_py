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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)
del train
sns.countplot(Y_train)
X_train.info()
Y_train.describe()
X_train = X_train / 255.0
test = test / 255.0
knn = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree')
knn.fit(X_train, Y_train)
y_predict = knn.predict(test)
y_predict.shape
submission = pd.DataFrame({
    'ImageId': np.arange(28000),
    'Label': y_predict
})
submission.to_csv('digit.csv', index=False)