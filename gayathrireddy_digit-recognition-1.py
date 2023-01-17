# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test_x = pd.read_csv("../input/test.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

from sklearn.ensemble import RandomForestClassifier

train.head()
 

#from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.
# import matplotlib.pyplot as plt
# first_digit = train.iloc[3].drop('label').values.reshape(28,28)
# plt.imshow(first_digit)
train_x = train.drop('label',axis=1)
train_y = train['label']
model = RandomForestClassifier(random_state=100)
model.fit(train_x,train_y)
test_pred = model.predict(test_x)
# test_pred.index = test_x.index+1
index1 = test_x.index+1
df_pred = pd.DataFrame({'ImageId': index1,'Label': test_pred})
df_pred.head()

df_pred[['ImageId', 'Label']].to_csv('submission_1.csv', index=False)