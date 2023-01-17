# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/digit-recognizer"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

print(train_data.head())

test_data = pd.read_csv('../input/digit-recognizer/test.csv')

print(test_data.head())

train_X = train_data[train_data.columns[1:]]

train_y = train_data.label

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(random_state=1)



model.fit(train_X, train_y)



predictions = model.predict(test_data)



print(predictions)

result = pd.DataFrame({'ImageId': range(1, len(test_data) + 1), 'Label': predictions})

print(result)