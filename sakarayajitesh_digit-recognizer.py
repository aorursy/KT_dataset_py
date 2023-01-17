# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import torch





dataset = pd.read_csv("../input/train.csv", dtype = np.float32)

targets_numpy = dataset.label.values

features_numpy = dataset.loc[:, dataset.columns != 'label'].values
features_numpy = features_numpy/255
print(len(features_numpy),len(targets_numpy))
train_features_numpy, train_labels_numpy = features_numpy[:20000], targets_numpy[:20000]

print(len(train_features_numpy), len(train_labels_numpy))
valid_features_numpy, valid_labels_numpy = features_numpy[20000:], targets_numpy[20000:]

print(len(valid_features_numpy), len(valid_labels_numpy))
train_features, train_labels = torch.from_numpy(train_features_numpy), torch.from_numpy(train_labels_numpy).type(torch.LongTensor)
valid_features, valid_labels = torch.from_numpy(valid_features_numpy), torch.from_numpy(valid_labels_numpy).type(torch.LongTensor)
import torchvision.transforms as transforms

from torchvision import datasets



train = torch.utils.data.TensorDataset(train_features,train_labels)

test = torch.utils.data.TensorDataset(valid_features,valid_labels)

train_loaders = torch.utils.data.DataLoader(dataset = train, batch_size = 64, shuffle = True)

test_loaders = torch.utils.data.DataLoader(dataset = test, batch_size = 64, shuffle = True)
import matplotlib.pyplot as plt

plt.imshow(features_numpy[2].reshape(28,28))

plt.axis("off")

plt.title(str(targets_numpy[2]))

plt.savefig('graph.png')

plt.show()
import torch.nn as nn

import torch.nn.functional as F



class Network(nn.Module):

    def __init__(self):

        super(Network,self).__init__()

        

        self.fc1 = nn.Linear(28*28,400)

        self.fc2 = nn.Linear(400,10)

    

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
model = Network()
if torch.cuda.is_available():

    model.cuda()
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr = 0.001)
from torch.autograd import Variable



epochs = 150

count = 0

loss_list = []

iteration_list = []

model.train()

model.cuda()



for epochs in range(epochs):

    for i,(images, labels) in enumerate(train_loaders):

        model.train()

        train = Variable(images.view(-1, 28*28))

        labels = Variable(labels)

        

        if torch.cuda.is_available():

            train, labels = train.cuda(), labels.cuda()

        

        optimizer.zero_grad()

        

        outputs = model(train)

        

        loss = criterion(outputs, labels)

        

        loss.backward()

        optimizer.step()

        

        count += 1

        

        if count % 50 == 0 and count!=0:

            correct = 0

            total = 0

            

            for image, label in test_loaders:

                if len(image)==len(label):

                    train = Variable(image.view(-1, 28*28))

                

                    if torch.cuda.is_available():

                        train = train.cuda()

                        label = label.cuda()

                    model.eval()

                    

                    output = model(train)

                

                    predicted = torch.max(output.data, 1)[1]

                    #print(len(label),len(train))

                

                    total += len(label)

                

                    correct += (predicted == label).sum()

                

                    accuracy = 100 * correct / float(total)

            

                loss_list.append(loss.data)

                iteration_list.append(count)

        if count % 500 == 0:

            # Print Loss

            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))

        

        

        

        

        
model.eval()
testset = pd.read_csv("../input/test.csv", dtype = np.float32)

test_numpy = testset.values

test_numpy = test_numpy/255

print(len(test_numpy))
test_features = torch.from_numpy(test_numpy)
plt.imshow(testset.values[2].reshape(28,28))

plt.axis("off")

plt.title("something")

plt.savefig('graph.png')

plt.show()
model.cpu()

feat = Variable(test_features[2].view(-1, 28*28))

output = model(feat)

predicted = torch.max(output.data, 1)[1]

print(predicted.numpy()[0])

key = []

value = []

for i in range(len(test_features)):

    feat = Variable(test_features[i].view(-1, 28*28))

    output = model(feat)

    predicted = torch.max(output.data, 1)[1]

    key.append(i+1)

    value.append(predicted.numpy()[0])
print(len(key),len(value))
op = 2

print(key[op],value[op])
submission = pd.DataFrame({'ImageId':key,'Label':value})
submission.head()
submission.to_csv("digit_recognizer3.csv",index=False)