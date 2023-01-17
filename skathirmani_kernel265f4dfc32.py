# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df_train, df_validate = train_test_split(train, test_size=0.3,random_state=100)

model_dt = DecisionTreeClassifier(max_depth=5)
train_x = df_train.drop('label', axis=1)
train_y = df_train['label']
validate_x = df_validate.drop('label', axis=1)
validate_y = df_validate['label']
model_dt.fit(train_x, train_y)
validate_pred = model_dt.predict(validate_x)
accuracy_score(validate_y, validate_pred)
test_pred = model_dt.predict(test)
df_test_pred = pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId'] = test.index + 1
df_test_pred[['ImageId', 'Label']].to_csv('submission_1.csv', index=False)
sample_submission.head()
test.head()
pixels = np.array(train.iloc[3].values[1:])
pixels = pixels.reshape((28, 28))
import matplotlib.pyplot as plt
plt.imshow(pixels, cmap='gray')
