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
import torch

import numpy as np

import pandas as pd

import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor

sc = StandardScaler()

torch.manual_seed(1)



# Load data

xy_data = pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv')

x_test = pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv')

submit = pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv')

xy_data = np.array(xy_data)

x_test = np.array(x_test)
x_train = torch.FloatTensor(xy_data[:,1:-1])

y_train = torch.FloatTensor(xy_data[:,-1])

x_test = torch.FloatTensor(x_test[:,1:-1])
print(x_train.shape)

print(x_test.shape)
sc.fit(x_train)

x_train_std = sc.transform(x_train)

x_test_std = sc.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsRegressor(n_neighbors=25, p = 2)

knn.fit(x_train_std,y_train) # knn 학습
y_test_pred = knn.predict(x_test_std)

threshold = 0.7

y_test_pred = y_test_pred>=0.5

y_test_pred = y_test_pred.astype(np.int32)

y_test_pred
for i in range(len(submit)):

  submit['Label'][i] = (int)(y_test_pred[i]);
submit.to_csv('submit.csv', mode='w', header=True, index= False)
y_test_pred
submit['Label'] = submit['Label'].astype("int32")
submit.to_csv('submit.csv', mode='w', header=True, index= False)
submit