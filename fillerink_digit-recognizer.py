# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
print(train.shape)

print(test.shape)

x_train = train.drop('label',axis=1)

y_train = train.label



print(x_train)

print(y_train)
model = RandomForestClassifier()

model.fit(x_train,y_train)
model.score(x_train,y_train)
output = model.predict(test)



output
output_csv = pd.DataFrame({'Label':output})

output_csv['ImageId'] = range(1, len(output_csv) + 1)



output_csv = output_csv[['ImageId','Label']]
output_csv.to_csv('submission.csv',index=False)