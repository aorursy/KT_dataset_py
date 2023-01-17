# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../../kaggle/input/digit-recognizer/train.csv')
data.head()
labels=data.label

images=np.array(data.iloc[:,1:])
import torch 

import torch.nn as nn

import torch.nn.functional as F
labels=np.array(labels)
images=torch.Tensor(images)

labels=torch.Tensor(labels)

labels=labels.type(torch.LongTensor)

images=images.view(images.shape[0],1,28,28)
train_images=images[:30000]

train_labels=labels[:30000]



test_images=images[30000:]

test_labels=labels[30000:]
from torchvision import transforms

trainset=torch.utils.data.TensorDataset(train_images,train_labels)

testset=torch.utils.data.TensorDataset(test_images,test_labels)
from torch.utils.data import Dataset, TensorDataset



import torchvision

import torchvision.transforms as transforms



class CustomTensorDataset(Dataset):

    """TensorDataset with support of transforms.

    """

    def __init__(self, tensors, transform=None):

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)

        self.tensors = tensors

        self.transform = transform



    def __getitem__(self, index):

        x = self.tensors[0][index]



        if self.transform:

            x = self.transform(x)



        y = self.tensors[1][index]



        return x, y



    def __len__(self):

        return self.tensors[0].size(0)
transform=transforms.Normalize((0.5,), (0.5,))

trainset=CustomTensorDataset(tensors=(train_images,train_labels),transform=transform)

testset=CustomTensorDataset(tensors=(test_images,test_labels),transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=64)

testloader=torch.utils.data.DataLoader(testset,batch_size=64)
class Classifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc4 = nn.Linear(64, 10)



        # Dropout module with 0.2 drop probability

        self.dropout = nn.Dropout(p=0.2)



    def forward(self, x):

        # make sure input tensor is flattened

        x = x.view(x.shape[0], -1)



        # Now with dropout

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))



        # output so no dropout here

        x = F.log_softmax(self.fc4(x), dim=1)



        return x
from torch import optim



model = Classifier()

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.003)



epochs = 100

steps = 0



train_losses, test_losses = [], []

for e in range(epochs):

    running_loss = 0

    for images, labels in trainloader:

        

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        test_loss = 0

        accuracy = 0

        

        # Turn off gradients for validation, saves memory and computations

        with torch.no_grad():

            model.eval()

            for images, labels in testloader:

                log_ps = model(images)

                test_loss += criterion(log_ps, labels)

                

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        model.train()

        

        train_losses.append(running_loss/len(trainloader))

        test_losses.append(test_loss/len(testloader))



        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(train_losses[-1]),

              "Test Loss: {:.3f}.. ".format(test_losses[-1]),

              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

        torch.save(model.state_dict(), 'checkpoint.pth')
test=pd.read_csv('../../kaggle/input/digit-recognizer/test.csv')

sub=pd.read_csv('../../kaggle/input/digit-recognizer/sample_submission.csv')
test_images=np.array(test)
test_images=torch.Tensor(test_images)
test_images=test_images.view(test_images.shape[0],1,28,28)
test_images.shape
dummy_labels=torch.rand(28000)
dummy_labels
test_dataset=CustomTensorDataset(tensors=(test_images,dummy_labels),transform=transform)
testloader=torch.utils.data.DataLoader(test_dataset,batch_size=test_images.shape[0])
model.eval()



dataiter = iter(testloader)

images, labels = dataiter.next()                
images.shape
ps=model(images)
ps.shape
top_p,top_class=ps.topk(1,dim=1)
top_class
classes=top_class.numpy()
classes.reshape(classes.shape[0],-1)
classes=top_class.view(-1,28000)
classes=classes.numpy()
classes=classes[0]
classes
mysub=pd.DataFrame()
sub['Label']=classes
mysub['Label']=classes
sub.head()
sub.to_csv('my_submission.csv',index=False)
newsub=pd.read_csv('my_submission.csv')
from IPython.display import FileLink

FileLink(r'my_submission.csv')
