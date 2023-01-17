! pip uninstall --y kaggle
! pip install --upgrade pip
! pip install kaggle==1.5.6
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle -v
! kaggle competitions download -c star-classifier
! unzip star-classifier.zip
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)
train = pd.read_csv('star_train.csv',header=None, skiprows=1, usecols=range(1,8))
train
test = pd.read_csv('star_test.csv', header=None, skiprows=1, usecols=range(1,7))
test
x_train = train.loc[:,1:6] #크게 다른 건 없지만, 데이터 파싱하는 부분이 달랐고
y_train = train.loc[:,7:7]

x_train = np.array(x_train)
y_train = np.array(y_train).squeeze(1) 
# 어떤 식으로 하셔서 된 건지 모르겠지만, 이 부분에서 squeeze로 matrix 스케일 맞춰줌
# cross_entrophy loss 실행 부분에서 multi~ 오류 해결 위해 넣음

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = test
x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test)
nb_class = 6 # 문제에서 6개라고 주어줌
nb_data=len(y_train)

w = torch.zeros((6, 6), requires_grad=True)
b = torch.zeros(6, requires_grad=True)
optimizer = optim.Adam([w, b], lr=1e-1) 
# 차이점 SGD보다 Adam이 optimizer중에 성능이 더 좋다고 해서...
epochs = 10000

for epoch in range(epochs + 1):
  hypothesis = x_train.matmul(w)+b
  cost = F.cross_entropy(hypothesis,y_train)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 500 == 0:
      print('{:.6f}'.format(cost.item()))
hypothesis2 = F.softmax(x_test.matmul(w)+b,dim=1)

predict=torch.argmax(hypothesis2, dim=1)
submit = pd.read_csv('star_sample.csv')
submit
for i in range(len(predict)):
  submit['Label'][i] = int(predict[i])
submit.dtypes
submit = submit.astype(int)
submit
! kaggle competitions submit -c star-classifier -f submission.csv -m "Message"