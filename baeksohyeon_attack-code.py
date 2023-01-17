!pip uninstall -y kaggle
!pip install --upgrade kaggle
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp "/content/drive/My Drive/kaggle.json" ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c childpark
!unzip childpark.zip
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import re

torch.manual_seed(1)
device = 'cuda'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
#데이터 로드
xy = pd.read_csv('train_classi.csv', header=None)
print(xy)

xy=pd.DataFrame.dropna(xy, axis=0, how='any', thresh=None, subset=None, inplace=False)


x_data = xy.loc[1:,1:8]
y_data = xy.loc[1:, 9]



date = xy.loc[1:,0]
A = date.str.extract(r'(\d+)[:]', expand=True)
print(A)
x_data["date1"] = A
x_data["date2"] = A


x_data = x_data.apply(pd.to_numeric)
y_data = y_data.apply(pd.to_numeric)

x_data = np.array(x_data)
y_data = np.array(y_data)



minMaxScaler = MinMaxScaler()
print(minMaxScaler.fit(x_data))
x_data = minMaxScaler.transform(x_data)

print(x_data[5:15])
print(y_data[5:15])
print(x_data.shape)
print(y_data.shape)


x_train=torch.FloatTensor(x_data)
y_train=torch.LongTensor(y_data)


from sklearn.ensemble import RandomForestClassifier

#랜덤포레스트 모델 사용

model = RandomForestClassifier(max_depth= 12, min_samples_leaf= 2, min_samples_split= 2, n_estimators = 3000, random_state=0)
model.fit(x_train, y_train)
#테스트데이터 로드
test = pd.read_csv('test.csv', header=None)
test=pd.DataFrame.dropna(test, axis=0, how='any', thresh=None, subset=None, inplace=False)
print(test)
test_data = test.loc[1:,1:8]


date_t = test.loc[1:,0]

B = date_t.str.extract(r'(\d+)[:]', expand=True)
print(B)
test_data["date1"] = B
test_data["date2"] = B

test_data = test_data.apply(pd.to_numeric)

test_data=np.array(test_data)

minmaxScaler = MinMaxScaler()
print(minmaxScaler.fit(test_data))
test_data = minmaxScaler.transform(test_data)

print(test_data[:5])
test_data=torch.FloatTensor(test_data)
print(test_data[0])



predict=model.predict(test_data)

print(predict.shape)
#제출
submit = pd.read_csv('submit.csv')




submit["result"]=predict
submit.to_csv("submit.csv",index=False,header=True)

!kaggle competitions submit -c childpark -f submit.csv -m ""