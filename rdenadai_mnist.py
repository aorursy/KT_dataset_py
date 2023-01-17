import numpy as np

from scipy.ndimage import rotate, zoom

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision.transforms as transforms



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



if torch.cuda.is_available():

    device = torch.device('cuda:0')

print(device)
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train.head(10)
def add_noise(img):

    noise = torch.randn(img.size()) * np.random.uniform(0, 0.1)

    noisy_img = img + noise

    return noisy_img





def min_max_normalization(tensor, min_value, max_value):

    min_tensor = tensor.min()

    tensor = (tensor - min_tensor)

    max_tensor = tensor.max()

    tensor = tensor / max_tensor

    tensor = tensor * (max_value - min_value) + min_value

    return tensor
y_train = train["label"].to_numpy().astype(np.int)

x_train = train.drop(labels = ["label"],axis = 1) 

x_train = x_train.values.reshape(-1, 1, 28, 28).astype(np.float32)



# Rotation

# x_rotate = torch.tensor(rotate(x_train, 30, reshape=False))



x_train = torch.tensor(x_train)

x_test = torch.tensor(test.values.reshape(-1, 1, 28, 28).astype(np.float32))



# Noise

x_noise = add_noise(x_train)



# Normalization

x_train /= 255.0

x_test /= 255.0

x_noise /= 255.0

# x_rotate /= 255
class NeuralNetwork(nn.Module):



    def __init__(self):

        super(NeuralNetwork, self).__init__()



        self.conv1 = nn.Conv2d(1, 32, 3)

        self.conv2 = nn.Conv2d(32, 64, 3)

        self.conv3 = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(7744, 2048)

        self.fc2 = nn.Linear(2048, 128)

        self.fc3 = nn.Linear(128, 10)

        self.dp1 = nn.Dropout(0.25)

        self.dp2 = nn.Dropout(0.5)



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = self.dp1(x)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.dp2(x)

        x = self.fc3(x)

        return x



    def num_flat_features(self, x):

        size = x.size()[1:]

        num_features = 1

        for s in size:

            num_features *= s

        return num_features



model = NeuralNetwork()

model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

print(model)
batch_size = 128

inputs = x_train.to(device)

inputs_noise = x_noise.to(device)

# inputs_rotate = x_rotate.to(device)

labels = torch.tensor(y_train)

for epoch in range(20):

    mb = np.ceil(inputs.size()[0] / batch_size).astype(np.int32)

    r = 0

    for _ in range(mb):

        ini, end = r * batch_size, (r + 1) * batch_size

        r += 1

        for k in range(2):

            if k == 0:

                batch_X, batch_y = inputs[ini:end, :], labels[ini:end]

            else:

                batch_X, batch_y = inputs_noise[ini:end, :], labels[ini:end]

            # Forward

            outputs = model(batch_X)

            loss = criterion(outputs.to(device), batch_y.to(device))

            # Backward

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    # print statistics

    if epoch % 1 == 0:

        print('[%d] loss: %.10f' % (epoch + 1, loss.item() / 2000))

print('Finished Training')
inputs_test = x_test.to(device)

outputs = model(inputs_test)

_, predicted = torch.max(outputs.data, 1)

y_pred = predicted.cpu().numpy()

print(np.unique(y_pred))
sub = pd.DataFrame()

sub["ImageId"] = list(range(1, y_pred.shape[0] + 1))

sub["Label"] = y_pred

sub.to_csv('submission.csv', index=False)