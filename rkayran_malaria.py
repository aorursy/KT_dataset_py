import torch

import numpy as np

import os

import cv2
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array
device = torch.device('cuda')

dtype = torch.float32
positive = os.listdir('../input/cell_images/cell_images/Parasitized/')

negative = os.listdir('../input/cell_images/cell_images/Uninfected/')
x = []

y = []

for pos in positive:

    if pos.endswith('png'):

        img = cv2.resize(cv2.imread('../input/cell_images/cell_images/Parasitized/' + pos), (64,64))

        x.append(img_to_array(img, data_format='channels_first'))

        y.append(1)

for neg in negative:

    if neg.endswith('png'):

        img = cv2.resize(cv2.imread('../input/cell_images/cell_images/Uninfected/' + neg), (64,64))

        x.append(img_to_array(img, data_format='channels_first'))

        y.append(0)



x = np.array(x)

y = np.array(y)
x_train,x_validation,y_train,y_validation = train_test_split(x, y, train_size=27000, random_state=42)
from torch.nn.modules import Conv2d,BatchNorm2d,ReLU,MaxPool2d,Sigmoid,Linear,Tanh



class Net(torch.nn.Module):

    def __init__(self):

        torch.nn.Module.__init__(self)

        self.layer_1 = Conv2d(3, 16, 3)

        self.layer_2 = BatchNorm2d(16)

        self.layer_3 = ReLU()

        self.layer_4 = MaxPool2d(2)

        self.layer_5 = Conv2d(16, 32, 3)

        self.layer_6 = BatchNorm2d(32)

        self.layer_7 = ReLU()

        self.layer_8 = MaxPool2d(2)

        self.layer_9 = Linear(6272, 2048)

        self.layer_10 = ReLU()

        self.layer_11 = Linear(2048, 512)

        self.layer_12 = ReLU()

        self.layer_13 = Linear(512, 1)

        self.layer_14 = Sigmoid()

        self.to(device)



    def forward(self, x):

        z1 = self.layer_1(x)

        z2 = self.layer_2(z1)

        z3 = self.layer_3(z2)

        z4 = self.layer_4(z3)

        z5 = self.layer_5(z4)

        z6 = self.layer_6(z5)

        z7 = self.layer_7(z6)

        z8 = self.layer_8(z7)

        z9 = self.layer_9(z8.view(z8.shape[0], -1))

        z10 = self.layer_10(z9)

        z11 = self.layer_11(z10)

        z12 = self.layer_12(z11)

        z13 = self.layer_13(z12)

        z14 = self.layer_14(z13)

        return z14



model = Net()
x_train = torch.tensor(x_train, device=device, dtype=dtype)

y_train = torch.tensor(y_train, device=device, dtype=dtype)
x_validation = torch.tensor(x_validation, device=device, dtype=dtype)

y_validation = torch.tensor(y_validation, device=device, dtype=dtype)
optimizer = torch.optim.Adam(model.parameters())

data_loader = torch.utils.data.dataloader.DataLoader(torch.utils.data.dataset.TensorDataset(x_train,y_train), shuffle=True, batch_size=32)
for epoch in range(20):

    for u,v in data_loader:

        v_hat = model(u)

        optimizer.zero_grad()

        loss = torch.nn.functional.binary_cross_entropy(v_hat, v)

        loss.backward()

        optimizer.step()

    pred = model(x_validation)

    pred = torch.ge(pred, 0.5).float().reshape(-1)

    error = torch.sub(pred, y_validation).abs().mean()

    print('acc: %f' % (1 - error.item()))