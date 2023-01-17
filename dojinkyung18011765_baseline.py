!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
! mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle -v
! kaggle competitions download -c 18011765watermelon-price
!unzip  18011765watermelon-price.zip
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
xy=np.loadtxt('train_water_melon_price.csv', delimiter=',',dtype = np.float32, skiprows = 1, usecols = range(1,9))

x_data = torch.from_numpy(xy[:,0:-1])
y_data = torch.from_numpy( xy[:,[-1]])

xy_test=np.loadtxt('test_watermelon_price.csv', delimiter=',',dtype = np.float32, skiprows = 1, usecols = range(1,8))
test_x_data=torch.from_numpy(xy_test)
print(x_data)
print(y_data)
print(test_x_data)



# For reproducibility
torch.manual_seed(1)
# 모델 초기화
w = torch.zeros((7,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([w,b], lr=0.0005)

epochs = 1750000
for epoch in range(epochs + 1):
    
    # H(x) 계산
    hypothesis = x_data.matmul(w) + b

    # cost 
    cost = F.mse_loss(hypothesis,y_data)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 10000번마다 로그 출력
    if epoch % 10000== 0:
        print('Epoch: {} Cost: {}'.format(
            epoch,cost.item()
        ))
prediction=test_x_data.matmul(w)+b
print(test_x_data)
print(prediction)

import pandas as pd
submit=pd.read_csv('submit_sample.csv')
submit
for i in range(len(prediction)):
  submit['Expected'][i]=prediction[i].item()
submit
submit.to_csv('baseline.csv',mode='w',index=False)
!kaggle competitions submit -c 18011765watermelon-price -f baseline.csv -m "Message"