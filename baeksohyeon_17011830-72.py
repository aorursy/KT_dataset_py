import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
!pip uninstall -y kaggle
!pip install --upgrade kaggle
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c white-wine-quality-evalutation
!unzip white-wine-quality-evalutation.zip
train_data=pd.read_csv('train.csv',header=None,skiprows=1, usecols=range(1,13))
test_data=pd.read_csv('test.csv',header=None,skiprows=1, usecols=range(1,12))
from sklearn.model_selection import train_test_split

x_train_data=train_data.loc[:,0:11]
y_train_data=train_data.loc[:,12]
print(x_train_data)

del x_train_data[7]
print(x_train_data)

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)
print(x_train_data.shape)


x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.LongTensor(y_train_data)

#학습 방법 변경

from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators=1000)
model.fit(x_train_data, y_train_data)
x_test_data=test_data.loc[:,:]
del x_test_data[7]
x_test_data=np.array(x_test_data)
print(x_test_data)
x_test_data=torch.from_numpy(x_test_data).float()
print(x_test_data.shape)


prediction = model.predict(x_test_data)

print(prediction)





submit=pd.read_csv('sample_submission.csv')
submit
for i in range(len(prediction)):
  submit['quality'][i]=prediction[i].item()

submit
submit.to_csv('submission.csv',index=False,header=True)

!kaggle competitions submit -c white-wine-quality-evalutation -f submission.csv -m "RF"