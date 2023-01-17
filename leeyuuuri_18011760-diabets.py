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
!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
import torch
import pandas as pd
import torch.optim as optim
import numpy as np

xy_data = pd.read_csv('../input/logistic-classification-diabetes-knn/train.csv')
x_test = pd.read_csv('../input/logistic-classification-diabetes-knn/test_data.csv')
submit = pd.read_csv('../input/logistic-classification-diabetes-knn/submission_form.csv')
X_train= xy_data.drop('8', axis =1)
y_train = xy_data['8']
x_test= x_test.drop('8', axis =1)
from sklearn.preprocessing import StandardScaler
sc  = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =5, p=2)
knn.fit(X_train_std, y_train)

y_train_pred = knn.predict(X_train_std)
y_test_pred = knn.predict(X_test_std)
for i in range(len(y_test_pred)):
  submit['Label'][i] = y_test_pred[i].item()

df = pd.DataFrame(submit, columns=["ID", "Label"])
df = df.astype({'Label':'int'})
df.to_csv("submit.csv", index = False)
