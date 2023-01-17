import time

import numpy as np

from skimage.transform import warp, AffineTransform

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.transforms as transforms



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



device = "cpu"

if torch.cuda.is_available():

    device = torch.device('cuda:0')

print(device)
class DatasetMNIST(Dataset):

    

    def __init__(self, file_path, has_label=True):

        self.data = pd.read_csv(file_path)

        self._has_label = has_label

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        # load image as ndarray type (Height * Width * Channels)

        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]

        # in this example, i don't use ToTensor() method of torchvision.transforms

        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)

        if self._has_label:

            image = self.data.iloc[index, 1:].values.astype(np.float32).reshape((28, 28))

            label = self.data.iloc[index, 0]

        else:

            image = self.data.iloc[index, 1:].values.astype(np.float32).reshape((28, 28))

            label = []

        

        apply_ = np.random.random()

        if apply_ >= 0.7:

            # angle is in radian

            tform = None

            trans_ = np.random.random()

            if trans_ <= 0.5:

            # if 0.0 <= trans_ < 0.4:

                angle = np.random.randint(35) * np.pi / 180

                tform = AffineTransform(rotation=angle)

                image = warp(image, tform.inverse, output_shape=image.shape)

            # elif 0.4 <= trans_ < 0.8:

            else:

                angle = np.random.randint(25) * np.pi / 180

                tform = AffineTransform(shear=angle)

                image = warp(image, tform.inverse, output_shape=image.shape)

            # elif 0.6 <= trans_ < 0.8:

            # else:

            #     image = random_noise(image, var=0.05)

            # else:

            #     image = image[:, ::-1]

        image /= 255.0

        image = torch.tensor(image.reshape(-1, 28, 28).astype(np.float32))

        return image, label
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

train.head(10)
batch_size = 128



train = DatasetMNIST("/kaggle/input/Kannada-MNIST/train.csv")

test = DatasetMNIST("/kaggle/input/Kannada-MNIST/test.csv", has_label=False)



train_size = int(0.99 * len(train))

test_size = len(train) - train_size

train, validation = random_split(train, [train_size, test_size])



train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
plt.figure(figsize=(13, 5))

for i, data in enumerate(train_loader):

    inputs, labels = data

    inputs = inputs * 255.0

    imgs = inputs[:32]

    for i, img in enumerate(imgs):

        plt.subplot(4, 8, i+1)

        plt.title(labels.numpy()[i])

        plt.axis('off')

        plt.imshow(img.numpy().reshape(28, 28), cmap="gray")

    break

plt.tight_layout()

plt.show()
class NeuralNetwork(nn.Module):



    def __init__(self):

        super(NeuralNetwork, self).__init__()



        self.conv1 = nn.Conv2d(1, 64, 3)

        self.conv2 = nn.Conv2d(64, 64, 3)

        self.conv1_bn = nn.BatchNorm2d(64)

        self.mpool0 = nn.MaxPool2d(2, 1)

        self.dp0 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(64, 64, 3)

        self.conv4 = nn.Conv2d(64, 64, 3)

        self.conv2_bn = nn.BatchNorm2d(64)

        self.mpool1 = nn.MaxPool2d(2)

        self.dp1 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(5184, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc2_bn = nn.BatchNorm1d(256)

        self.dp2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 10)

        self.softmax = nn.LogSoftmax(dim=1)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.mpool0(self.conv1_bn(F.relu(self.conv2(x))))

        x = self.dp0(x)

        x = F.relu(self.conv3(x))

        x = self.mpool1(self.conv2_bn(F.relu(self.conv4(x))))

        x = self.dp1(x)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))

        x = self.fc2_bn(F.relu(self.fc2(x)))

        x = self.dp2(x)

        x = self.fc3(x)

        return self.softmax(x)



    def num_flat_features(self, x):

        size = x.size()[1:]

        num_features = 1

        for s in size:

            num_features *= s

        return num_features



model = NeuralNetwork()

model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-4)

print(model)
loss_ = []

pred_ = []

start = time.time()

for epoch in range(12):

    for i, data in enumerate(train_loader):

        inputs, labels = data

        # Forward

        outputs = model(inputs.to(device))

        loss = criterion(outputs.to(device), labels.to(device))

        # Backward

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    loss_.append(loss.item())

    

    pred = []

    for i, data in enumerate(validation_loader):

        inputs, labels = data

        outputs = model(inputs.to(device))

        _, predicted = torch.max(outputs.data, 1)

        y_pred = predicted.cpu().numpy().ravel()

        pred += list(y_pred == labels.cpu().numpy().ravel())

    pred_.append(np.round(np.mean(pred), 5))

    

    # print statistics

    if epoch % 1 == 0:

        tm = time.time() - start

        print('[%d] loss: %.10f, accuracy: %.3f, time: %.2f' % (epoch + 1, np.mean(loss_) / 2000, np.mean(pred), tm))

        start = time.time()



plt.figure(figsize=(7, 5))

plt.subplot(1, 2, 1)

plt.plot(loss_)

plt.xlabel("Epocs")

plt.ylabel("Loss")

plt.subplot(1, 2, 2)

plt.plot(pred_)

plt.xlabel("Epocs")

plt.ylabel("Accuracy")

plt.tight_layout()

plt.show()
pred = []

for i, data in enumerate(validation_loader):

    inputs, labels = data

    outputs = model(inputs.to(device))

    _, predicted = torch.max(outputs.data, 1)

    y_pred = predicted.cpu().numpy().ravel()

    pred += list(y_pred == labels.cpu().numpy().ravel())

print(f"{np.round(np.mean(pred) * 100, 2)}%")
y_pred = []

for i, data in enumerate(test_loader):

    inputs, labels = data

    outputs = model(inputs.to(device))

    _, predicted = torch.max(outputs.data, 1)

    y_pred += list(predicted.cpu().numpy().ravel())

y_pred = np.array(y_pred)

print(y_pred[:20])
sub = pd.DataFrame()

sub["id"] = list(range(0, y_pred.shape[0]))

sub["label"] = y_pred

sub.to_csv('submission.csv', index=False)