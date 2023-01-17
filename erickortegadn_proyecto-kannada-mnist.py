# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.utils.data

from torch.autograd import Variable

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

print(train.shape)
y = train['label'].values

x = train.drop(['label'],1).values 



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

def plot_random_digit():

    random_index = np.random.randint(0,x_train.shape[0])

    plt.imshow(x_train[random_index].reshape((28,28)), cmap='nipy_spectral')
plot_random_digit()
batch_size = 32



torch_x_train = torch.from_numpy(x_train).type(torch.LongTensor)

torch_x_train = torch_x_train.view(-1,1,28,28).float()

torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)



torch_x_test = torch.from_numpy(x_test).type(torch.LongTensor)

torch_x_test = torch_x_test.view(-1,1,28,28).float()

torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)



train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)

test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)



train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)



print(torch_x_train.shape)

print(torch_x_test.shape)
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)

        self.fc1 = nn.Linear(3*3*64, 256)

        self.fc2 = nn.Linear(256, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(F.max_pool2d(self.conv3(x),2))

        x = F.dropout(x, p=0.5, training=self.training)

        x = x.view(-1,3*3*64)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

 

cnn = CNN()

print(cnn)



it = iter(train_loader)

X_batch, y_batch = next(it)

print(cnn.forward(X_batch).shape)
def fit(model, train_loader):

    optimizer = torch.optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    epochs = 5

    model.train()

    for epoch in range(epochs):

        correct = 0

        print('Epoch: {}'.format(epoch))

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):

            var_x_batch = Variable(x_batch).float()

            var_y_batch = Variable(y_batch)

            optimizer.zero_grad()

            output = model(var_x_batch)

            loss = criterion(output, var_y_batch)

            loss.backward()

            optimizer.step()

            predicted = torch.max(output.data, 1)[1] 

            correct += (predicted == var_y_batch).sum()

            if batch_idx % 50 == 0:

                print('({:.0f}%)\tLoss: {:.6f}\t Accuracy: {:.3f}%'

                      .format(100.*batch_idx / len(train_loader),

                              loss.data, 

                              float(correct*100) / float(batch_size*(batch_idx+1)))

                     )

        print('------------------------------------------------------------------------------------')
fit(cnn, train_loader)
def evaluate(model):

    correct = 0

    for test_imgs, test_labels in test_loader:

        test_imgs = Variable(test_imgs).float()

        output = model(test_imgs)

        predicted = torch.max(output,1)[1]

        correct += (predicted == test_labels).sum()

    print("Test accuracy: {:.3f}% ".format(100*(float(correct) / (len(test_loader)*batch_size))))
evaluate(cnn)
def view_classify(img, ps, version="MNIST"):

    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    if version == "MNIST":

        ax2.set_yticklabels(np.arange(10))

   

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()

    plt.show()
images, labels = next(iter(train_loader))

random_index = np.random.randint(0,images.shape[0])

img = images[random_index].view(-1, 1,28,28).float()



with torch.no_grad():

    output = cnn(img).cpu()



ps = torch.exp(output)

view_classify(img.view(1, 28, 28), ps)
raw_test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

print(raw_test.shape)

raw_test = raw_test.drop("id",axis="columns") 

print(raw_test.shape)

raw_test = raw_test / 255 

tests = raw_test.values.reshape(-1,28,28,1)

print(tests.shape)
torch_x_test = torch.from_numpy(tests).type(torch.LongTensor)

torch_x_test = torch_x_test.view(-1,1,28,28).float()



with torch.no_grad():

    output = cnn(torch_x_test).cpu()



softmax = torch.exp(output)

prob = list(softmax.numpy())

predictions = np.argmax(prob, axis=1)
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.to_csv("submission.csv", index=False)

submission.head()