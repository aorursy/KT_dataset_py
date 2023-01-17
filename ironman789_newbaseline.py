import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

from sklearn import preprocessing
Scaler = preprocessing.StandardScaler()
random.seed(777)
torch.manual_seed(777)
!unzip white-wine-quality-evalutation.zip
train_data=pd.read_csv('train.csv',header=None,skiprows=1, usecols=range(1,13))
test_data=pd.read_csv('test.csv',header=None,skiprows=1, usecols=range(1,12))
train_data=pd.read_csv('train.csv',header=None,skiprows=1, usecols=range(1,13))
test_data=pd.read_csv('test.csv',header=None,skiprows=1, usecols=range(1,12))
from sklearn.model_selection import train_test_split

x_train_data=train_data.loc[:,0:11]
y_train_data=train_data.loc[:,12]
print(x_train_data)

del x_train_data[7]
del x_train_data[8]
print(x_train_data)

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)
x_train_data = Scaler.fit_transform(x_train_data)

print(x_train_data.shape)


x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.LongTensor(y_train_data)
from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators=2000)
model.fit(x_train_data, y_train_data)
x_test_data=test_data.loc[:,:]
print(x_test_data)
del x_test_data[7]
del x_test_data[8]

x_test_data=np.array(x_test_data)
x_test_data = Scaler.transform(x_test_data)

print(x_test_data)
x_test_data=torch.from_numpy(x_test_data).float()
print(x_test_data.shape)


prediction = model.predict(x_test_data)
prediction = torch.torch.from_numpy(prediction).float()


print(prediction)

ans = pd.read_csv('solution.csv')
ans = ans.loc[:,'quality']
ans = np.array(ans)
ans = torch.torch.from_numpy(ans).float()
print(ans)
correct_prediction = prediction.float() == ans
accuracy = correct_prediction.sum().item() / len(correct_prediction)
print('The model has an accuracy of {:2.4f}% for the training set.'.format(accuracy * 100))
submit=pd.read_csv('sample_submission.csv')
submit
for i in range(len(prediction)):
  submit['quality'][i]=prediction[i].item()

submit
submit.to_csv('NewLine.csv',index=False,header=True)