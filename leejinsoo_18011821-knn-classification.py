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

import pandas as pd

import torch.optim as optim

import numpy as np

torch.manual_seed(1)

device = torch.device("cpu")

#데이터

xy_data = pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv')

x_test = pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv')

submit = pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv')
xy_np = xy_data.to_numpy()
xy_data
x_data = xy_np[:,1:9]

y_data = xy_np[:,-1]

print(x_data.shape)

print(y_data.shape)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5,p=2)



knn.fit(x_data,y_data)
y_train_pred=knn.predict(x_data)



print((y_data != y_train_pred).sum())
x_test_np = x_test.to_numpy()
x_test_data = x_test_np[:,1:-1]
x_data.shape
x_test_data.shape
predict = knn.predict(x_test_data)
predict.shape
for i in range(len(predict)):

    submit['Label'][i]=predict[i]

submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header= True, index= False)