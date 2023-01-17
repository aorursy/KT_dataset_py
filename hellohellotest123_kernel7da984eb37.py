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
import numpy as np

import pandas as pd
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape)

print(test.shape)
x_train = train.drop('label',axis = 1)
#x_train = x_train/255
print(x_train.shape)
print(x_train)
#x_train_df = pd.DataFrame(x_train)
#x_train_df
x_train1 = x_train[0:1]
x_train2 = np.array(x_train1)
x_train3 = np.reshape(x_train2, (28, 28))
print(x_train3.shape)
from PIL import Image

import matplotlib.pyplot as plt
plt.imshow(x_train3, cmap = "gray")
y_train = train['label']
print(y_train)
print(type(y_train))
#y_train_df = pd.DataFrame(y_train)
#y_train_df1 = y_train_df['label']
#y_train_df1
from sklearn.model_selection import train_test_split
#x_train_train, x_train_test = train_test_split(x_train, train_size=30000)

#y_train_train, y_train_test = train_test_split(y_train, train_size=30000)

x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train, stratify = y_train, random_state=0)
x_train_train
y_train_train
print(x_train_train.shape)

print(x_train_test.shape)

print(y_train_train.shape)

print(y_train_test.shape)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(x_train_train, y_train_train) 

print('accuracy: %.3f' % clf.score(x_train_test, y_train_test))
y_pred = clf.predict(test)
y_pred

print(y_pred.shape)
sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sub.head
sub
sub['Label'] = list(map(int,y_pred))

sub.to_csv('submission.csv',index=False)
sub