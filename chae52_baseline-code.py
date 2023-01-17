!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6

from google.colab import files
#kaggle이라는 폴더를 만든다
!mkdir -p ~/.kaggle
#폴더에 kaggle.json을 copy paste 한다
!cp kaggle.json ~/.kaggle/
#권한을 넣어준다
! chmod 600 ~/.kaggle/kaggle.json
#자세한 내용 출력
!ls -lha kaggle.json

#kaggle 버전 확인
!kaggle -v
!kaggle competitions download -c parkinglot
#위 코드를 통해 얻게되는 zip파일을 푼다
!unzip parkinglot.zip
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

torch.manual_seed(1)
#13개의 열에 대해서 분석해야되므로 처음0에서 14까지를 불러온 다음 0~12까지는 x 13은 y 로 저장한다
xy=np.loadtxt('train.csv',delimiter=',',dtype=np.float32, skiprows=1,usecols=range(0,14)) 
print(xy[:5])
print(xy.shape)
x_train=torch.from_numpy(xy[:,0:-1])
y_train=torch.from_numpy(xy[:,[-1]])
#13개의 feature이므로 13,1로 설정해준다
W = torch.zeros((13, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#nan이 뜨지 않게 잘 작동하는 learning rate를 찾아서 적어준다.
optimizer = optim.SGD([W, b], lr=1e-10)
nb_epochs = 10000

for epoch in range(nb_epochs + 1):
   hypothesis = torch.sigmoid(x_train.matmul(W) + b)
   cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()

   optimizer.zero_grad()
   cost.backward()
   optimizer.step()

   if epoch % 1000 == 0:
       #item():스칼라 값으로 보기 위해서
       print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
#test 데이터를 불러온다 test는 x값 열 0~12까지 존재한다
xy_test=np.loadtxt('test.csv',delimiter=',',dtype=np.float32, skiprows=1,usecols=range(0,13))  
test_x=torch.from_numpy(xy_test)
print(test_x[0:5])
#학습된파라미터 값으로 결과를 추측한다
hypothesis = torch.sigmoid(test_x.matmul(W) + b)
print(hypothesis[:5])

predict = hypothesis >= torch.FloatTensor([0.5])
print(predict[:5])
#결과를 제출하기 위해 제출 양식 파일을 불러온다
import pandas as pd
submit=pd.read_csv('submission.csv')

#파일에 추측값을 넣는다
for i in range(len(predict)):
  submit['Expected'][i]=int(predict[i])

#int형으로 바꾸어준다
submit['Expected']=submit['Expected'].astype(int)
submit[:5]
#제출한다
submit.to_csv('submit.csv',mode='w',index=False) 
!kaggle competitions submit -c parkinglot -f submit.csv -m "submit"