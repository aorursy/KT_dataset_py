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

train = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')
test = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')
submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')
print(train.shape)
print(test.shape)
print(train.head())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = train.iloc[:,1:-1]
x_train = np.array(x_train)
x_train = scaler.fit_transform(x_train)
x_train = torch.FloatTensor(x_train)

y_train = train.iloc[:,-1]
y_train = np.array(y_train)
y_train = torch.FloatTensor(y_train)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.3,random_state=1)


x_test = test.iloc[:,1:]
x_test = np.array(x_test)
x_test = scaler.fit_transform(x_test)
x_test = torch.FloatTensor(x_test)

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=90, weights='distance')
knn.fit(x_train, y_train)
y_val_pred = knn.predict(x_val)
val=0
for i in range(len(y_val_pred)):
  val += (y_val[i]-y_val_pred[i])**2
accuracy = (val/len(y_val_pred))**(1/2) / 100

print(accuracy)
y_test_pred = knn.predict(x_test)

y_test_pred
for i in range(len(y_test_pred)):
  submit['Expected'][i] = y_test_pred[i]

submit = submit.astype(np.int32)
submit
submit.to_csv('submit.csv', mode='w', header=True, index=False)
!kaggle competitions submit -c mlregression-cabbage-price -f submit.csv -m "Message"