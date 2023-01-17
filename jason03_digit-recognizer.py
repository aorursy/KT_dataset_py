import torch

import torch.nn as nn

import torch.nn.functional as F

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as sns

from keras.utils.np_utils import to_categorical
test_x = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")

train_y = train['label']

train_x = train.drop('label', axis=1)
print("train_x describe: \n", train_x.isnull().any().describe())

print("\ntest_x describe: \n", test_x.isnull().any().describe())

print("\ntrain_y describe: \n", train_y.value_counts())

sns.countplot(train_y)
train_x /= 255

test_x /= 255
train_x = train_x.values.reshape(-1, 1, 28, 28)

test_x = test_x.values.reshape(-1, 1, 28, 28)
train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.1, stratify=train_y)
train_x = torch.FloatTensor(np.array(train_x))

validation_x = torch.FloatTensor(np.array(validation_x))

test_x = torch.FloatTensor(np.array(test_x))

train_y = torch.LongTensor(np.array(train_y))

validation_y = torch.LongTensor(np.array(validation_y))
num_epochs = 10

num_classes = 10

learning_rate = 0.01

batch_size = 100
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

validation_dataset = torch.utils.data.TensorDataset(validation_x, validation_y)

test_dataset = torch.utils.data.TensorDataset(test_x)



train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
class ConvNet(nn.Module):

    def __init__(self, num_classes):

        super(ConvNet, self).__init__()

        # 28*28*1

        self.layer1 = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=3),  # 28*28*8

            nn.BatchNorm2d(8),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 14*14*8

            nn.Dropout(0.5))



        self.layer2 = nn.Sequential(

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3),  # 14*14*16

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 7*7*16

            nn.Dropout(0.5))



        self.fc = nn.Linear(7*7*16, num_classes)



    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)  # 即将batch*7*7*32的数据集改为batch*(7*7*32)的大小，进入全连接层

        out = self.fc(out)

        return out
model = ConvNet(num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []

total_train_step = len(train_loader)

total_validation_step = len(validation_loader)
for epoch in range(num_epochs):

    correct = 0

    for step, (train_x, train_y) in enumerate(train_loader):

        model.train()

        outputs = model(train_x)

        train_loss = criterion(outputs, train_y)

        correct += (torch.max(outputs, 1)[1] == train_y).sum().item()



        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()

        

        if (step + 1) % 100 == 0:

            print("step[{}/{}], loss:{:.4f}".format(step + 1, total_train_step, train_loss))



    train_losses.append(train_loss.item())

    train_acc = correct / (total_train_step * batch_size)

            

    model.eval()

    correct = 0

    for step, (validation_x, validation_y) in enumerate(validation_loader):

        outputs = model(validation_x)

        validation_loss = criterion(outputs, validation_y)

        correct += (torch.max(outputs, 1)[1] == validation_y).sum().item()

    validation_acc = correct / (total_validation_step * batch_size)

    print('epoch[{}/{}], train loss:{:.4f}, train acc:{:.4f}, validation loss:{:.4f}, validation acc:{:.4f}'.format(epoch + 1, num_epochs, train_loss.item(), train_acc, validation_loss.item(), validation_acc))
model.eval()

with torch.no_grad():

    for step, test_x in enumerate(test_loader):

        test_x = test_x[0]

        output = model(test_x)

        if not step:

            test_y = outputs

        else:

            test_y = torch.cat((test_y, output), 0)

test_y = pd.DataFrame(torch.argmax(test_y, 1).numpy())

test_id = pd.DataFrame([i for i in range(1, test_y.shape[0] + 1)])

test_y = pd.concat([test_id, test_y], axis=1)

test_y.columns = ['ImageId', 'Label']

test_y.to_csv('submission.csv', index=False, encoding='utf8')



# plt.figure(figsize=(6, 4), dpi=144)

plt.plot([i + 1 for i in range(num_epochs)], train_losses, 'r-', lw=1)

plt.yticks([x * 0.1 for x in range(15)])

plt.show()