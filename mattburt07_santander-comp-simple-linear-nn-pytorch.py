import torch

import torch.nn as nn                         

from torch.autograd import Variable           

import torch.utils.data as Data               

import torchvision           

%matplotlib inline

import pandas as pd

import numpy as np
train = pd.read_csv("../input/train.csv")

train = train.sample(frac=1)
trainx = train.drop(['ID_code', 'target'], axis=1); trainy = train['target'] 

train_x = torch.from_numpy(trainx[:160000].values); train_y = torch.from_numpy(trainy[:160000].values)

test_x = torch.from_numpy(trainx[160000:].values); test_y = torch.from_numpy(trainy[160000:].values)
class MyDataset(Data.Dataset):

    def __init__(self, X, y): self.data = X; self.target = y.long()

    def __getitem__(self, index): x = self.data[index]; y = self.target[index]; return x, y

    def __len__(self): return len(self.data)
trainDataset = MyDataset(train_x, train_y)

trainLoader = Data.DataLoader(dataset=trainDataset, batch_size=2048, shuffle=True)
class Model(nn.Module):

    def __init__(self): 

        super().__init__()

        self.linear = nn.Sequential(nn.Linear(200, 100),nn.ReLU(),nn.BatchNorm1d(100),)

        self.linear2 = nn.Sequential(nn.Dropout(0.1),nn.Linear(100, 50),nn.ReLU(),nn.BatchNorm1d(50),)

        self.output = nn.Linear(50, 2)

  

    def forward(self, x):

        x = self.linear(x.float())

        x = self.linear2(x.float())

        output = self.output(x)

        return output, x
linear_nn = Model()

lossF = nn.CrossEntropyLoss(); epoch = 2

print(linear_nn)
def TrainNN(dataLoader, model, num_epochs, loss_function, lr):

    for i in range(num_epochs):

        for step, (items,labels) in enumerate(dataLoader): 

            images_x = Variable(items);labels_y = Variable(labels)

            output = model(images_x)[0];

            loss = loss_function(output, labels_y)

            optimiser = torch.optim.Adam(model.parameters(), lr)

            optimiser.zero_grad();

            loss.backward()

            optimiser.step()



            if step % 1000 == 0:

                test_output, last_layer = model(test_x)

                pred_y = torch.max(test_output, 1)[1].data.squeeze()

                accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))

                print('Epoch #: '+ str(i) + '| Test Accuracy: %.4f' % accuracy)

    print('\n')
lrs = [1e-2, 1e-3, 1e-4]

for i in lrs:

    print('Learning rate: ' + str(i) + '\n')

    TrainNN(trainLoader, linear_nn, epoch,lossF, i)
test = pd.read_csv('../input/test.csv')

test2 = (test.drop(['ID_code'], axis=1)).values

test2 = torch.from_numpy(test2.astype(float))

test_output, last_layer = linear_nn(test2)

pred_y = (test_output[:, 1]).detach().numpy()

pred_y = (pred_y-min(pred_y))/(max(pred_y)-min(pred_y))
predictions = []; predictions.append(pred_y); predictions[0][:10]
test['target'] = predictions[0]

#test[['ID_code', 'target']].to_csv('submission.csv', index=False)