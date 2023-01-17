import torch

import torch.utils.data

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



%matplotlib inline





import os

print(os.listdir("../input/digit-recognizer/"))
train_data = pd.read_csv('../input/digit-recognizer/train.csv')
X = train_data.iloc[:, 1:].values / 255

y = train_data.iloc[:, 0].values



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.15)
Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape
## train

torch_Xtrain = torch.from_numpy(Xtrain).type(torch.FloatTensor)

torch_Xtrain = torch_Xtrain.view(-1, 1, 28, 28) # creating 28 by 28 images

torch_ytrain = torch.from_numpy(ytrain).type(torch.LongTensor)

# test

torch_Xtest = torch.from_numpy(Xtest).type(torch.FloatTensor)

torch_Xtest = torch_Xtest.view(-1, 1, 28, 28)

torch_ytest = torch.from_numpy(ytest).type(torch.LongTensor)



# datasets

train_set = torch.utils.data.TensorDataset(torch_Xtrain, torch_ytrain)

test_set = torch.utils.data.TensorDataset(torch_Xtest, torch_ytest)



# dataloaders

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
images, labels = next(iter(train_loader))
fig = plt.figure(figsize=(12, 10))

x, y = 8 ,3

for i in range(24):

    plt.subplot(y, x, i+1)

    plt.imshow(images[i].reshape(28, 28))

plt.show()
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 32, 3),

            nn.ReLU(),

            nn.MaxPool2d(2),

        )

        

        self.conv2 = nn.Sequential(

            nn.Conv2d(32, 64, 3),

            nn.ReLU(),

            nn.MaxPool2d(2),

        )

        

        self.conv3 = nn.Sequential(

            nn.Conv2d(64, 128, 3),

            nn.ReLU(),

            nn.MaxPool2d(2),

        )

        

        self.fc1 = nn.Linear(128*1*1, 256)

        self.fc2 = nn.Linear(256, 256)

        self.fc3 = nn.Linear(256, 10)

        

        self.dropout = nn.Dropout(p=.5)

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.dropout(x)

        x = self.conv2(x)

        x = self.dropout(x)

        x = self.conv3(x)

        x = self.dropout(x)

        # flattening

        #print(x.shape) #check shape before flattening

        x = x.view(-1,128*1*1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.fc3(x)

        

        return x
model = Net()

model
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()
epochs = 30

model.train()

for epoch in range(epochs):

    running_loss = 0

    accuracy = 0

    for step, (x_b, y_b) in enumerate(train_loader):

        X_batch = Variable(x_b)

        Y_batch = Variable(y_b)

        

        optimizer.zero_grad()

        output = model(X_batch)

        loss = criterion(output, Y_batch)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

        if step % 50 == 0:

            model.eval()

            pred = torch.max(output, 1)[1]

            accuracy += (pred == Y_batch).sum()

            print(f"Epoch: {epoch}, Train Loss: {running_loss}, Test Accuracy: {float(accuracy*100)/ float(len(train_loader))}")

            

        running_loss = 0
model.eval()

for images, labels in test_loader:

    #print(test_imgs.shape)

    images = Variable(images).float()

    output = model(images)

    predicted = torch.max(output,1)[1]

    accuracy += (predicted == labels).sum()

print("Test accuracy:{:.3f}% ".format(float(accuracy*100) / float(len(test_loader))))
test_data = pd.read_csv('../input/digit-recognizer/test.csv')

test_data.head()
test_data = test_data.iloc[:, :].values / 255

test_data_torch = torch.from_numpy(test_data).type(torch.FloatTensor)

test_data_torch = test_data_torch.view(-1, 1, 28, 28)
test_data_torch.shape
model.eval()

test_Images = Variable(test_data_torch).float()

output = model(test_Images)

y_pred = torch.max(output, 1)[1]
# checking all values are being predicted

y_pred.unique()
y_pred
# visualize the predictions

fig = plt.figure(figsize=(12, 10))

x, y = 8 ,3

for i in range(24):

    ax = plt.subplot(y, x, i+1,xticks=[], yticks=[])

    ax.imshow(test_data_torch[i].reshape(28, 28))

    ax.set_title(f"{y_pred[i].item()}")
# saving the model

torch.save(model.state_dict(), 'pytorch_model.pt')
# creating the submission file

submission = pd.DataFrame({"ImageId":[i+1 for i in range(len(test_data_torch))],

                           "Label": y_pred})

submission.head()
submission.to_csv("submission_with_pytorch.csv", index=False, header=True)