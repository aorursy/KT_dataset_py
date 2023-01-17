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
from torch.nn import BCELoss
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
torch.manual_seed(1)
device = torch.device("cuda")
xy = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv')
z = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv')

xy = np.array(xy)
z = np.array(z)
x = torch.FloatTensor(xy[:,1:-1])
y = torch.FloatTensor(xy[:,-1])
z = torch.FloatTensor(z[:,1:-1])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1,stratify=y)


sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)
knn=KNeighborsClassifier(n_neighbors=3,p=2)
knn.fit(x_train_std,y_train)
y_train_pred=knn.predict(x_train_std)
y_test_pred=knn.predict(x_test_std)
print(accuracy_score(y_test,y_test_pred))
z_std=sc.transform(z)
guess=knn.predict(z_std)
result = pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')
print(result)
for i in range(len(guess)):
  result['Label'][i]=guess[i]
result=result.astype(np.int32)
result.to_csv('result.csv', mode='w', header= True, index= False)
print(result)

