%reload_ext autoreload

%autoreload 2

%matplotlib inline
import torchvision

import torch

import torchvision.transforms as transforms

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from torch.utils.data import Dataset



class MNIST(Dataset):

    def __init__(self, csv_path, is_test_set = False, transform=None):

        data = pd.read_csv(csv_path).to_numpy()

        n_samples = data.shape[0]

        if is_test_set:

            images = data.reshape(n_samples, 28, 28, 1).astype(np.float32)

            # The test dataset from kaggle contains no labels, hence we use -1

            labels = np.ones(n_samples)*(-1) 

        else:

            images = data[:,1:].reshape(n_samples, 28, 28, 1).astype(np.float32)

            labels = data[:,0]

    

        self.samples = []

        for i in range(n_samples):

                self.samples.append((images[i,:], labels[i]))

                

        self.transform = transform

        

    def __len__(self):

        return len(self.samples)

    

    def __getitem__(self, idx):

        img, label = self.samples[idx]



        if self.transform:

           img = self.transform(img)

            

        return img, label
from torch.utils.data import DataLoader



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

trainset = MNIST(csv_path='../input/digit-recognizer/train.csv', transform=transform)

testset = MNIST(csv_path='../input/digit-recognizer/test.csv', is_test_set=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
for i in range(16):

    img, label = trainset[i]

    plt.subplot(4,4,i+1)

    plt.axis('off')

    plt.imshow(img.numpy().squeeze()*-1, cmap='Greys',  interpolation='nearest')

    plt.title(label)



plt.tight_layout()

plt.rcParams["figure.figsize"] = (5,4)

plt.show()
import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1,32,3, 1)

        self.conv2 = nn.Conv2d(32,64,3, 1)

        self.dropout1 = nn.Dropout2d(0.25)

        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(9216, 128)

        self.fc2 = nn.Linear(128, 10)

        

    def forward(self,x):

        x = self.conv1(x)

        x = F.relu(x)

        x = self.conv2(x)

        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.dropout2(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output



net = Net()



# Pass network to GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");

net.to(device)
dataiter = iter(trainloader)

data = dataiter.next()

imgs, labels = data[0].to(device), data[1].to(device)

output = net(imgs)

test_output = torch.exp(output[0])

test_output.cpu().detach().numpy()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adadelta(net.parameters(), lr=0.001)



for epoch in range(25):

    

    running_loss = 0.0

    for i,data in enumerate(trainloader,0):

        data, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        

        outputs = net(data)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        if i % 500 == 499:

            print('Epoch %d: loss: %.3f' % (epoch + 1, running_loss / 2000))

            running_loss = 0.0



print('Finished Training!')
torch.save(net.state_dict(), "mnist_cnn.pt")
net = Net()

net.load_state_dict(torch.load('./mnist_cnn.pt'))

net.to(device)
n_tests = len(testset)

test_loss = 0

correct = 0

net.eval()



predictions = np.zeros((n_tests,2))

with torch.no_grad():

    for i,(data,_) in enumerate(testset):

        data = torch.unsqueeze(data, 0).to(device)

        output = net(data)

        pred = output.argmax(dim=1, keepdim=True)

        predictions[i,0] = i+1

        predictions[i,1] = pred
predictions = predictions.astype(int)

df = pd.DataFrame({'ImageId': predictions[:,0],

                   'Label': predictions[:,1]})
from IPython.display import HTML

HTML(df.iloc[0:10,:].to_html(index=False))
df.to_csv(path_or_buf='./predictions.csv',index=False)