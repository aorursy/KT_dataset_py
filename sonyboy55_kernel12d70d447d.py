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
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train.info()
test.info()
train.head()
train_X = np.array(train.drop(['label'], 1))
train_Y = np.array(train['label'])
test = np.array(test)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

digit = train_X[0]
digit_image = digit.reshape(28, 28)
plt.imshow(digit_image, cmap="binary")
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier()
clf.fit(train_X, train_Y)
pred = clf.predict(test)
print(pred)
sub = pd.DataFrame(pred)
sub.index.name='ImageId'
sub.index+=1
sub.columns=['Label']
sub.to_csv('submission_knn.csv',header=True)