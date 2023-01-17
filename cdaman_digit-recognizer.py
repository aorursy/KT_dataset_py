%matplotlib inline
import numpy as np 
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df.head()
df.describe()
label = [i for i in range(10)]
height = [df['label'].value_counts()[i] for i in range(10)]
plt.bar(label,height,color = 'r')
plt.xticks(label);
idx = np.random.randint(1,42000,20)
fig = plt.figure(figsize=(25, 4))
img = df.loc[idx,df.columns != 'label'].values.reshape(-1,28,28)
label = df.loc[idx,'label'].values
for i in np.arange(20):
    ax = fig.add_subplot(2, 20/2, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(img[i]), cmap='gray')
    ax.set_title(str(label[i]))
img = df.loc[50,df.columns != 'label'].values.reshape(28,28)
fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap = 'Greys_r')
plt.title(df.loc[50,'label'])
plt.xticks([])
plt.yticks([]);
thresh = img.max()/2.5
height = width = 28
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
X = torch.from_numpy(df.loc[:,df.columns != 'label'].values/255)#scaling
X = X.reshape(42000,1,28,28)

y = torch.from_numpy(df.label.values).type(torch.LongTensor)


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = .05)

trainset = torch.utils.data.TensorDataset(X_train,y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valset = torch.utils.data.TensorDataset(X_val,y_val)
valloader = torch.utils.data.DataLoader(valset,batch_size = 64,shuffle = True)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
model = Net()
model
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
n_epoch = 50
loss_min = np.Inf
model = model.double()
for i in range(1,n_epoch+1):
    train_loss = 0
    valid_loss = 0
    

    model.train()
    for data,target in trainloader: 
        optimizer.zero_grad()
        #data = data.astype(np.double
        out = model(data.double())
        loss = criterion(out,target)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()*data.size(0)

    model.eval()
    for data,target in valloader:
        out = model(data)
        loss = criterion(out,target)
        valid_loss += loss.item()*data.size(0)

    train_loss = train_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(valloader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        i, train_loss, valid_loss))

    if valid_loss <= loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        loss_min = valid_loss
model.load_state_dict(torch.load('/kaggle/working/model_cifar.pt'))
img,label = next(iter(valloader))
out = model(img)
_,out = torch.max(out,1)

fig = plt.figure(figsize=(25, 4))
for i in np.arange(20):
    ax = fig.add_subplot(2, 20/2, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(img[i]), cmap='gray')
    ax.set_title(f'{label[i]}({out[i]})',color = 'g' if label[i] == out[i] else 'r')