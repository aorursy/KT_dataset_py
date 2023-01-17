!pip uninstall -y kaggle

!pip install --upgrade pip

!pip install kaggle==1.5.6



!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle

!chmod 600 ~/.kaggle/kaggle.json



!kaggle competitions download -c aidefensegame18011862

!unzip aidefensegame18011862.zip
import torch

import torch.optim as optim

import numpy as np

import torch.nn as nn

import pandas as pd



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
if torch.cuda.is_available() is True:

  device = torch.device('cuda')

else:

  device = torch.device('cpu')
data = pd.read_csv('18011862Aitrain.csv')

pd_test = pd.read_csv('alltestdata.csv')



data.info()

pd_test.info()
##nan처리

data.replace('불명',np.NAN, inplace=True)

data.dropna(inplace=True)



##경찰서 처리

data['관할경찰서'][data.관할경찰서.str.contains('서울')] = '0'

data['관할경찰서'][data.관할경찰서.str.contains('인천')] = '2'

data['관할경찰서'][data.관할경찰서.str.contains('수원')|data.관할경찰서.str.contains('일산')|data.관할경찰서.str.contains('성남')|data.관할경찰서.str.contains('용인')|data.관할경찰서.str.contains('안양')|data.관할경찰서.str.contains('안산')|data.관할경찰서.str.contains('과천')|data.관할경찰서.str.contains('광명')|data.관할경찰서.str.contains('군포')|data.관할경찰서.str.contains('부천')|

                   data.관할경찰서.str.contains('시흥')|data.관할경찰서.str.contains('김포')|data.관할경찰서.str.contains('안성')|data.관할경찰서.str.contains('오산')|data.관할경찰서.str.contains('의왕')|data.관할경찰서.str.contains('이천')|data.관할경찰서.str.contains('평택')|data.관할경찰서.str.contains('하남')|data.관할경찰서.str.contains('화성')|data.관할경찰서.str.contains('여주')|

                   data.관할경찰서.str.contains('양평')|data.관할경찰서.str.contains('고양')|data.관할경찰서.str.contains('구리')|data.관할경찰서.str.contains('남양주')|data.관할경찰서.str.contains('동두천')|data.관할경찰서.str.contains('양주')|data.관할경찰서.str.contains('의정부')|data.관할경찰서.str.contains('파주')|data.관할경찰서.str.contains('포천')|data.관할경찰서.str.contains('연천')|

                   data.관할경찰서.str.contains('가평')|data.관할경찰서.str.contains('분당')] = '1'



label1 = data['관할경찰서'] == '0'

label2 = data['관할경찰서'] == '1'

label3 = data['관할경찰서'] == '2'



pd_train = data[label1| label2| label3]

pd_train['관할경찰서'] = pd_train['관할경찰서'].astype(int)



##성별 처리

replace_values = {'남자' : 1,'여자' : 2}

pd_train = pd_train.replace({"성별": replace_values})

pd_test = pd_test.replace({"성별" : replace_values})



pd_train = pd_train.iloc[:,1:]
##시간 처리_train



pd_train['측정일시'] = pd.to_datetime(pd_train['측정일시'], format = '%Y-%m-%d %H:%M',errors = 'raise')



pd_train['측정요일'] = pd_train['측정일시'].dt.weekday

pd_train['측정시각'] = pd_train['측정일시'].dt.hour



pd_train = pd_train[['성별','적발횟수','측정요일','측정시각','관할경찰서','나이']]





##시간 처리_test

pd_test['측정일시'] = pd.to_datetime(pd_test['측정일시'], format = '%Y-%m-%d %H:%M',errors = 'raise')



pd_test['측정요일'] = pd_test['측정일시'].dt.weekday

pd_test['측정시각'] = pd_test['측정일시'].dt.hour



pd_test = pd_test[['성별','적발횟수','측정요일','측정시각','관할경찰서']]
pd_y = pd_train['나이'].astype(int)

pd_x = pd_train.iloc[:,:-1]





pd_y[pd_y<40] = 0

pd_y[pd_y>=40] = 1



pd_y.sum()
x_train = torch.FloatTensor(np.array(pd_x)).to(device)

y_train = torch.FloatTensor(np.array(pd_y).reshape(-1,1)).to(device)



x_test = torch.FloatTensor(np.array(pd_test)).to(device)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)
torch.manual_seed(1)

torch.cuda.manual_seed_all(1)





X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.1, random_state = 100)

print(X_train.shape)

print(X_valid.shape)

print(Y_train.shape)

print(Y_valid.shape)
d = torch.utils.data.TensorDataset(X_train, Y_train)

data_loader = torch.utils.data.DataLoader(dataset = d,

                                          batch_size=batch_size,

                                          shuffle = True,

                                          drop_last=True)
lr = 1e-3

batch_size = 200

Epochs = 100

torch.manual_seed(1)

torch.cuda.manual_seed_all(1)



linear1 = nn.Linear(5,512,bias=True)

linear2 = nn.Linear(512, 128, bias=True)

linear3 = nn.Linear(128, 1, bias=True)



relu = nn.ReLU()



model = nn.Sequential(linear1, relu, linear2, relu, linear3).to(device)



for layer in model.children():

  if isinstance(layer, nn.Linear):

    nn.init.xavier_uniform_(layer.weight)



loss = nn.BCELoss().to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
total_batch = len(data_loader);

best_acc = 0

accuracy = 0



for epoch in range(Epochs+1):

  model.train()

  avg_cost = 0;



  for X, Y in data_loader:

    X = X.view(-1,5)

    optimizer.zero_grad()

    H = torch.sigmoid(model(X))

    cost=(loss(H,Y))

    cost.backward()

    optimizer.step()



    avg_cost += cost / total_batch

    # scheduler.step(cost)

  with torch.no_grad():

    model.eval()

    # import pdb;pdb.set_trace()



    valid = torch.sigmoid(model(X_valid))

    valid[valid<0.5] = 0

    valid[valid>=0.5] = 1

    

    accuracy = accuracy_score(Y_valid.to('cpu'), valid.to('cpu').detach().numpy())*100



    if best_acc < accuracy :

      best_acc = accuracy

      print("save bestmodel, epoch: {:4d}".format(epoch))

      torch.save(model, './best_model.ptr')



  print("Epoch{:4d}/{}, cost : {:.06f}, accuracy : {:.06f}".format(epoch,

                                              Epochs,

                                              avg_cost,

                                              accuracy))

  

print('finish')

model = torch.load("./best_model.ptr")
with torch.no_grad():

  

  model.eval()

  predict = torch.sigmoid(model(x_test))

predict[predict>=0.5] = 1

predict[predict<0.5] = 0

predict.sum()
label = predict.to('cpu').detach().numpy()

Id = np.array([int(i) for i in range(len(predict))]).reshape(-1,1)



result = np.hstack((Id, label))

result
df = pd.DataFrame(result, columns=(['ID','Label']))

df['ID'] = df['ID'].astype(int)

df.to_csv("submission.csv",index=False, header=True)
!kaggle competitions submit -c aidefensegame18011862 -f submission.csv -m "Message"