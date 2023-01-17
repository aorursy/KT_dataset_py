import numpy as np

import pandas as pd



import random



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from torchvision.utils import make_grid

from torch.utils.data import TensorDataset, DataLoader



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')





print(train.shape)

train.head()
print(test.shape)

test.head()
x_train_df = train.iloc[:,1:]

y_train_df = train.iloc[:,0]



print(x_train_df.shape, y_train_df.shape)
x_train = x_train_df.values/255.

y_train = y_train_df.values



x_test = test.values/255
x_train = np.reshape(x_train, (-1, 1, 28,28))

x_test = np.reshape(x_test, (-1, 1, 28,28))





x_train.shape, x_test.shape
# This is to ensure reproducibility

random_seed = 234

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)





x_train.shape, x_val.shape, y_train.shape, y_val.shape
def display(rows, columns, images, values=[], predictions=[]):

    fig = plt.figure(figsize=(9, 11))



    ax = []



    for i in range( columns*rows ):

        img = images[i]

        ax.append(fig.add_subplot(rows, columns, i+1))

        

        title = ""

        

        if(len(values) == 0):

            title = "Pred:" + str(predictions[i])

        elif(len(predictions) == 0):

            title = "Value:" + str(values[i])

        elif(len(values) != 0 and len(predictions) != 0):

            title = "Value:" + str(values[i]) + "\nPred:" + str(predictions[i])

        

        ax[-1].set_title(title)  # set title

        plt.imshow(img)



    plt.show()

    

idx = np.random.randint(1, 1000, size=9)



images = x_train[idx,:]

images = images[:,0]



values = y_train[idx]



display(rows=3, columns=3, images=images, values=values, predictions=[])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)

        self.conv3 = nn.Conv2d(32,64, kernel_size=5)

        self.fc1 = nn.Linear(3*3*64, 256)

        self.fc2 = nn.Linear(256, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(F.max_pool2d(self.conv3(x),2))

        x = F.dropout(x, p=0.5, training=self.training)

        x = x.view(-1,3*3*64 )

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    

net = Net()



net.to(device)



net
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
torch_x_train = torch.from_numpy(x_train).type(torch.FloatTensor)

torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)



train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)



train_loader = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = False)
%%time



#Seed

torch.manual_seed(1234)



for epoch in range(10):

    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):

        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        

        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        

        # print statistics

        running_loss += loss.item()

        if i % 500 == 499:    # print every 500 mini-batches

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, loss.item()))

#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i, running_loss / 500))

#             running_loss = 0.0



print('Finished Training')
#Validate trained model

torch_x_val = torch.from_numpy(x_val).type(torch.FloatTensor)

torch_y_val = torch.from_numpy(y_val).type(torch.LongTensor)



torch_x_val, torch_y_val = torch_x_val.to(device), torch_y_val.to(device)



val = net(torch_x_val)



_, predicted = torch.max(val.data, 1)



#Get accuration

print('Accuracy of the network %d %%' % (100 * torch.sum(torch_y_val==predicted) / len(y_val)))
# Get random data from the valication dataset and the predicted values

idx = np.random.randint(1, 1000, size=9)



images = x_val[idx,:]

images = images[:,0]



values = y_val[idx]



predicted = predicted.cpu()



predictions = predicted.data.numpy()

predictions = predictions[idx]



display(rows=3, columns=3, images=images, values=values, predictions=predictions)
torch_x_test = torch.from_numpy(x_test).type(torch.FloatTensor)



torch_x_test = torch_x_test.to(device)



y_test = net(torch_x_test)



_, predicted = torch.max(y_test.data, 1)
idx = np.random.randint(1, 1000, size=9)



images = x_test[idx,:]

images = images[:,0]



predicted = predicted.cpu()



predictions = predicted.data.numpy()

predictions = predictions[idx]



display(rows=3, columns=3, images=images, values=[], predictions=predictions)
ImageId = np.arange(1, len(x_test)+1)

Label = predicted.data.numpy()



my_submission = pd.DataFrame({'ImageId': ImageId, 'Label': Label})

my_submission.to_csv('submission.csv', index=False)



my_submission.head()