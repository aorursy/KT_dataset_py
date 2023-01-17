!pip install kaggle

from google.colab import files

files.upload()
!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/



!chmod 600 ~/.kaggle/kaggle.json



!kaggle competitions list
!kaggle competitions download -c logistic-classification-diabetes-knn

!ls

!unzip
import torch

import pandas as pd

import torch.optim as optim

import numpy as np

torch.manual_seed(1)



xy=pd.read_csv('train.csv',header=None)



x_data=xy.loc[1:,1:8]

y_data=xy.loc[1:,9]

x_data=np.array(x_data)

y_data=np.array(y_data)



x_train=torch.FloatTensor(x_data)

y_train=torch.LongTensor(y_data)



print(x_train)

print(len(y_train))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5,p=2)



knn.fit(x_train,y_train)
test=pd.read_csv('test_data.csv',header=None)

x_data=test.loc[1:,1:8]



x_data

x_data=np.array(x_data)

x_test=torch.FloatTensor(x_data)
y_train_pred=knn.predict(x_train)



y_test_pred=knn.predict(x_test)
for i in range(len(y_test_pred)):

  submit['Label'][i]=y_test_pred[i]
submit=submit.astype(np.int32)

submit.to_csv('submit.csv', mode='w', header= True, index= False)
!kaggle competitions submit -c logistic-classification-diabetes-knn -f submit.csv -m "Message"