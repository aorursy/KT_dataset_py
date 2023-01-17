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
# Importing libraries
import numpy as np
import pandas as pd

# Making the splits
training = pd.read_csv('../input/train.csv')
testing = pd.read_csv('../input/test.csv')

X_train = training.iloc[:, 0:-1].values
y_train = training.iloc[:, -1].values

X_test = testing.iloc[:, 1:].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)
df_rfout = pd.DataFrame(testing['Id'].values, columns=['Id'])
df_rfout['Solution'] = y_pred
# save decision tree prediction:
df_rfout.to_csv('solution1.csv', index=False)