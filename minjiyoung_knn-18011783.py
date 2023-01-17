!pip install kaggle
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions list
! kaggle competitions download -c logistic-classification-diabetes-knn
! ls
! unzip
import torch
import pandas as pd
import torch.optim as optim
import numpy as np
torch.manual_seed(1)

#데이터
xy_data = pd.read_csv('train.csv')
x_test = pd.read_csv('test_data.csv')
submit = pd.read_csv('submission_form.csv')

xy_data = np.array(xy_data)
x_train = torch.FloatTensor(xy_data[:,1:-1])
y_train = torch.LongTensor(xy_data[:,9])
x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test[:,1:-1])
from sklearn.neighbors import KNeighborsClassifier  #KNN 불러오기
knn=KNeighborsClassifier(n_neighbors=5,p=2) #5개의 인접한이웃, 거리측정기준:유클리드 
knn.fit(x_train, y_train)
y_train_pred=knn.predict(x_train) #train data의 y값 예측치
y_test_pred=knn.predict(x_test)  #모델을 적용한 test data의 y값 예측치
y_test_pred

for i in range(len(y_test_pred)):
  submit['Label'][i]=y_test_pred[i]
submit=submit.astype(np.int32)
submit.to_csv('submit.csv', mode='w', header= True, index= False)
 !kaggle competitions submit -c logistic-classification-diabetes-knn -f submit.csv -m "Message"