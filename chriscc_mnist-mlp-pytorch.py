

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import torch

from torchvision import datasets

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
class Net(nn.Module):

## The following two lines are the reinforced format for contructing a Pytorch network class

    def __init__(self):

        super(Net, self).__init__()

# Input layer

        self.input = nn.Linear(28 * 28, 512)

# Hidden layer

        self.hidden = nn.Linear(512, 256)

# Output layer

        self.output = nn.Linear(256, 10)

    

    def forward(self, x):

        x = x.view(-1, 28 * 28)

        x = F.sigmoid(self.input(x))

        x = F.sigmoid(self.hidden(x))

        x = self.output(x)

        return x

    

model = Net()

print(model)
batch_size = 128

transform = transforms.ToTensor()





x_train, x_val, y_train, y_val = train_test_split(

    train.values[:,1:], train.values[:,0], test_size=0.2)





train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train.astype(np.float32)/255),

                                               torch.from_numpy(y_train))



val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_val.astype(np.float32)/255),

                                               torch.from_numpy(y_val))



test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test.values[:,:].astype(np.float32)/255))



# data loader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
n_epochs = 30

for epoch in range(n_epochs):

    train_loss = 0.0

    for data, target in train_loader:

        optimizer.zero_grad()

        # Forward propagation

        output = model(data)

        # Calculate the loss

        loss = criterion(output, target)

        # Back propagation

        loss.backward()

        # Update weights using the optimizer

        optimizer.step()

        # Calculate the cumulated loss

        train_loss += loss.item()*data.size(0)

    

    train_loss = train_loss/len(train_loader.dataset)

    

    print(f"Epoch: {epoch}, train loss: {train_loss}")
val_loss = 0.0

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



model.eval() # prep model for evaluation



for data, target in val_loader:

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the loss

    loss = criterion(output, target)

    # update val loss 

    val_loss += loss.item()*data.size(0)

    # convert output probabilities to predicted class

    _, pred = torch.max(output, 1)

    # compare predictions to true label

    correct = np.squeeze(pred.eq(target.data.view_as(pred)))

    # calculate val accuracy for each object class

    for i in range(len(target)):

        label = target.data[i]

        class_correct[label] += correct[i].item()

        class_total[label] += 1



# calculate and print avg val loss

val_loss = val_loss/len(val_loader.sampler)

print('val Loss: {:.6f}\n'.format(val_loss))



for i in range(10):

    if class_total[i] > 0:

        print('val Accuracy of %5s: %2d%% (%2d/%2d)' % (

            str(i), 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('val Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nval Accuracy (Overall): %2d%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
model.eval() # prep model for evaluation



preds = []



for data in test_loader:

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data[0])

    # calculate the loss

    _, pred = torch.max(output, 1)

    preds.extend(pred.tolist())

    # compare predictions to true label

submission['Label'] = preds

submission.to_csv('submission.csv', index=False)