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
import numpy as np
import torch.optim as optim

torch.manual_seed(1)

train = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv')
test = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv')
submit = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#scaler = StandardScaler()
x_train = train.iloc[:,1:-1] #x_train
x_train = np.array(x_train)
#x_train = scaler.fit_transform(x_train)
x_train = torch.FloatTensor(x_train)

y_train = train.iloc[:,-1]
y_train = torch.FloatTensor(y_train)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.08, random_state=1, stratify=y_train)

x_test = test.iloc[:,1:-1]
x_test = np.array(x_test)
#x_test = scaler.fit_transform(x_test)
x_test = torch.FloatTensor(x_test)

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)
print(x_test.shape)
print(x_val, y_val)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=22) #15 7894 18 80 20 807
knn.fit(x_train, y_train)
y_train_pred = knn.predict(x_train)
y_val_pred = knn.predict(x_val)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, y_train_pred))
print(accuracy_score(y_val, y_val_pred))

print(y_train_pred)
print(y_train)
print(y_val)
print(y_val_pred)
y_test_pred = knn.predict(x_test)

for i in range(len(y_test_pred)):
  submit['Label'][i] = y_test_pred[i]

submit = submit.astype(np.int32)
submit
submit.to_csv('submit.csv', mode='w', header=True, index = False)
  !kaggle competitions submit -c logistic-classification-diabetes-knn -f submit.csv -m "Message"