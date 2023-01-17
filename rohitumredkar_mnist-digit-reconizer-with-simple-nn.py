# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os



import torch

from torch import cuda

from torch.utils.data import DataLoader, TensorDataset, Dataset

from torch import nn, optim

import torch.nn.functional as F
train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train.shape,test.shape
x = train.drop(['label'], axis=1)

y = np.array(train['label'])
torch_x_train = torch.from_numpy(x.values).type(torch.FloatTensor)/255

torch_y_train = torch.from_numpy(y).type(torch.LongTensor)



mydataset = torch.utils.data.TensorDataset(torch_x_train, torch_y_train, )

valid_no = int(0.2* len(mydataset))



# So divide the data into trainset and testset

train_set, test_set = torch.utils.data.random_split(mydataset, (len(mydataset)-valid_no, valid_no))



print(f'Length of Trainset: {len(train_set)} \nLength of Testset: {len(test_set)}')

batch_size = 45



train_loader = DataLoader(train_set, batch_size= batch_size, shuffle=True)

test_loader = DataLoader(test_set, batch_size = batch_size, shuffle=True)
print(type(torch_x_train), torch_x_train.shape, torch_x_train)

print(type(torch_y_train), torch_y_train.shape, torch_y_train)
class Network(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 128)        

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 10)

        

        

        # Dropout module with 0.2 drop Probabilities

        self.dropout = nn.Dropout(p = 0.05)

        



    def forward(self, x):

        # Make sure input tensor id flattened

        x = x.view(x.shape[0], -1)

        

        # Now with dropout

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        

        # Output so no dropout here

        x = F.log_softmax(self.fc3(x), dim=1)

        

        return(x)

model = Network()

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.NLLLoss()


epochs = 10

train_losses, test_losses = [],[]



for e in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        

    else:

        test_loss = 0

        accuracy  = 0

        

        with torch.no_grad():

            model.eval()

            for images, labels in test_loader:

                log_ps = model(images)

                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class==labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        model.train()

        train_losses.append(running_loss/len(train_loader))

        test_losses.append(test_loss/len(test_loader))

        

        print('Epoch: {}/{}..'.format(e+1, epochs),

             'Training Loss: {:.3f}..'.format(running_loss/len(train_loader)),

             'Test Loss: {:.3f}..'.format(test_loss/len(test_loader)),

             'Test Accuracy: {:.3f}..'.format(accuracy/len(test_loader)))
print('Our Model: \n\n', model, '\n')

print("The state dict keys: \n\n", model.state_dict().keys())
torch.save(model.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth')

print(state_dict.keys())
model.load_state_dict(state_dict)
checkpoint = {'input_size': 784,

              'output_size': 10,

              'hidden_layers': [588,392,196,66],

              'state_dict': model.state_dict()}



torch.save(checkpoint, 'checkpoint.pth')
test_images = test.loc[:,test.columns != 'label'].values

test_dataset = torch.from_numpy(test_images).type(torch.FloatTensor)/255

print(test_dataset.shape)

new_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= 100, shuffle=False)
results = []

with torch.no_grad():

    for images in new_test_loader:

        output = model(images)

        ps = torch.exp(output)

        top_p, top_class = ps.topk(1, dim=1)

        results += top_class.numpy().tolist()
predictions = np.array(results).flatten()

print(predictions[:5])

print(predictions.shape)
submission = pd.DataFrame({'ImageId': list(range(1, len(predictions)+1)),

                           'Label': predictions})

submission.to_csv("my_submissions.csv", index=False, header=True)