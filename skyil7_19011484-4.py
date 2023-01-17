!pip uninstall kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!ls -lha kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c 2020-ai-exam-fashionmnist-4
! unzip mnist_train_label.csv.zip
! unzip mnist_test.csv.zip
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import numpy as np

from sklearn import preprocessing
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)
train = pd.read_csv('mnist_train_label.csv', header=None)
test = pd.read_csv('mnist_test.csv', header=None, index_col=0)
train.head()
test.head()
y_train = train.iloc[:,0]
x_train = train.iloc[:,1:784]
print(x_train.shape)
print(test.shape)
x_train = np.array(x_train)
y_train = np.array(y_train)
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = np.array(test)
x_test = scaler.transform(x_test)
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=10000,
                                          shuffle=True,
                                          drop_last=True)
x_test = x_test.to(device)
l1 = torch.nn.Linear(783, 128, bias=True)
l2 = torch.nn.Linear(128, 128, bias=True)
l3 = torch.nn.Linear(128, 64, bias=True)
l4 = torch.nn.Linear(64, 64, bias=True)
l5 = torch.nn.Linear(64, 10, bias=True)
re = torch.nn.ReLU()
#드랍아웃
do = torch.nn.Dropout()
#모델 초기화
torch.nn.init.xavier_uniform_(l1.weight)
torch.nn.init.xavier_uniform_(l2.weight)
torch.nn.init.xavier_uniform_(l3.weight)
torch.nn.init.xavier_uniform_(l4.weight)
torch.nn.init.xavier_uniform_(l5.weight)

model = torch.nn.Sequential(l1, re, do, l2, re, do, l3, re, do, l4, re, do, l5).to(device)
!pip install adamp
from adamp import AdamP
costf = torch.nn.CrossEntropyLoss().to(device)
optimizer = AdamP(model.parameters(), lr=0.01, weight_decay=1e-2)
total_batch = len(data_loader)
epochs=15
for epoch in range(epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = costf(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    if epoch% 3 ==0 or True:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
with torch.no_grad():
    print(x_test.shape)
    pred = model(x_test)

submit = pd.read_csv('submission.csv')
print(pred.shape)
print(submit.shape)
pred = torch.argmax(pred, axis=1)
print(pred[:5])
print(pred.shape)
for i in range(len(pred)):
  submit['Category'][i]=pred[i].item()
submit.head()
submit.to_csv('4.csv', index=0)
!kaggle competitions submit -c 2020-ai-exam-fashionmnist-4 -f 4.csv -m "submit"
