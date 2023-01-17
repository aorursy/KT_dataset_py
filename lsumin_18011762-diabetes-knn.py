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

import torch

import torch.optim as optim

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler  #



torch.manual_seed(1)

xy_train = np.loadtxt('/kaggle/input/logistic-classification-diabetes-knn/train.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(1,10))



x_train = torch.from_numpy(xy_train[:,0:-1])



y_data = xy_train[:,[-1]].squeeze()

y_train = torch.LongTensor(y_data)



xy_test = np.loadtxt('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(1,9))

test_data = torch.from_numpy(xy_test)



print(x_train)

print(x_train.shape)

print(y_train)

print(test_data)
knn = KNeighborsClassifier(n_neighbors=25,p=2)

knn.fit(x_train, y_train)
predict = knn.predict(test_data)
submit = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')

for i in range(len(predict)):

  submit['Label'][i] = predict[i].item()



submit['Label'] = submit['Label'].astype(int)



submit
submit.to_csv('18011762.csv', index=False, header=True)