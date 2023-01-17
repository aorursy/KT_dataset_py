!pip install kaggle
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions list
! kaggle competitions download -c mlregression-cabbage-price
! ls
! unzip
import torch
import pandas as pd
import torch.optim as optim
import numpy as np
torch.manual_seed(1)

#데이터
xy_data = pd.read_csv('train_cabbage_price.csv')
x_test = pd.read_csv('test_cabbage_price.csv')
submit = pd.read_csv('sample_submit.csv')

xy_data = np.array(xy_data)
x_train = torch.FloatTensor(xy_data[:,1:-1])
y_train = torch.LongTensor(xy_data[:,-1])
x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test[:,1:])
x_test
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=20,weights = "distance")
regressor.fit(x_train, y_train)
guesses = regressor.predict(x_test)
for i in range(len(guesses)):
  submit['Expected'][i]=guesses[i]
submit=submit.astype(np.int32)
submit.to_csv('submit.csv', mode='w', header= True, index= False)

 !kaggle competitions submit -c mlregression-cabbage-price -f submit.csv -m "Message"