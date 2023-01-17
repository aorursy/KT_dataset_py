!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls -lha kaggle.json

!chmod 600 ~/.kaggle/kaggle.json
!ls -lha kaggle.json
! kaggle competitions download -c rainyseason
!unzip rainyseason.zip
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F

torch.manual_seed(1)
xy_train = np.loadtxt('rainyseason-train.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(4,13))

x_train = torch.from_numpy(xy_train[:,1:])

y_data = xy_train[:,[0]].squeeze()
y_train = torch.LongTensor(y_data)

xy_test = np.loadtxt('rainyseason-test.csv', delimiter=',', dtype=np.float32, skiprows=1, usecols=range(5,13))
test_x_data = torch.from_numpy(xy_test)

print(x_train)
print(x_train.shape)
print(y_train)
print(test_x_data)
print(test_x_data.shape)
w = torch.zeros((8,2), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w, b], lr=1e-1, momentum=0.8)
epochs = 10000

for epoch in range(epochs+1):
  hypothesis = F.softmax(x_train.matmul(w) + b, dim=1)

  y_one_hot = torch.zeros(len(y_train), 2)
  y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
  cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch%1000==0 :
    print("Epoch: {}    Cost: {:.6f}".format(epoch, cost.item()))
hypothesis = F.softmax(test_x_data.matmul(w) + b, dim=1)
predict = torch.argmax(hypothesis, dim=1)

print(predict)
submit = pd.read_csv('sample.csv')
submit
for i in range(len(predict)):
  submit['RainySeason'][i] = predict[i].item()

submit['RainySeason'] = submit['RainySeason'].astype(int)

submit
submit.to_csv('baseline.csv', index=False, header=True)
! kaggle competitions submit -c rainyseason -f baseline.csv -m "18011762이수민"
hypothesis = F.softmax(x_train.matmul(w)+b, dim=1)
predict = torch.argmax(hypothesis, dim=1)

correct_prediction = predict.float() == y_train
print(correct_prediction)

accuracy = correct_prediction.sum().item() / len(correct_prediction)

print("Accuracy: {:2.2f}%".format(accuracy*100))