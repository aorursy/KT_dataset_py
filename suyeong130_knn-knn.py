!pip install kaggle

from google.colab import files

files.upload()
!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/



!chmod 600 ~/.kaggle/kaggle.json



!kaggle competitions list
!kaggle competitions download -c mlregression-cabbage-price

!ls

!unzip
import numpy as np

import torch



xy_train=np.loadtxt('train_cabbage_price.csv',delimiter=',',dtype=np.float32,skiprows=1,usecols=range(1,6))

x_data=torch.from_numpy(xy_train[:,0:-1])

y_data=torch.from_numpy(xy_train[:,[-1]])



xy_test=np.loadtxt('test_cabbage_price.csv',delimiter=',',dtype=np.float32,skiprows=1,usecols=range(1,5))

test_x_data=torch.from_numpy(xy_test)
from sklearn.neighbors import KNeighborsRegressor



#regressor=KNeighborsRegressor(n_neighbors=5, weights="distance")

regressor=KNeighborsRegressor(n_neighbors=6)

regressor.fit(x_data,y_data)
#y_data_pred=regressor.predict(x_data)



y_test_pred=regressor.predict(test_x_data)
y_test_pred
import pandas as pd 



submit=pd.read_csv('sample_submit.csv')



submit
for i in range(len(y_test_pred)):

  submit['Expected'][i]=y_test_pred[i].item()



submit
submit.to_csv('submit.csv',mode='w',index=False)
!kaggle competitions submit -c mlregression-cabbage-price -f submit.csv -m "submit6"