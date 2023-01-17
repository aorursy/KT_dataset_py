# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn

import torch.nn.functional as F



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load in the 28x28 images



train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
labels = train["label"]

train = train.drop(["label"], axis=1)
tensor = torch.Tensor(train.values)

tensor.shape
train_data = tensor.reshape(-1,1,28,28)
class CNNClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.c1 = nn.Conv2d(1,4, kernel_size=7, stride=2)

        self.c2 = nn.Conv2d(4,8, kernel_size=3)

        self.c3 = nn.Conv2d(8,16, kernel_size=3)

        self.lin = nn.Linear(16, 10)

        

        self.b1 = nn.BatchNorm2d(4)

        self.b2 = nn.BatchNorm2d(16)

        self.b3 = nn.BatchNorm2d(10)

        

    def forward(self, x):

        # pass through the conv net

        x = F.relu(self.b1(self.c1(x)))

        x = F.relu(self.b2(self.c3(self.c2(x))))

        

        # make a prediction

        x = (self.lin(x.mean(dim=[2,3])))

        

        # return torch.argmax(x, dim=1) // apparently CrossEntropyLoss takes care of the vector... totally forgot

        return x
# init the model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)

model = CNNClassifier().to(device)

data_loader = torch.utils.data.DataLoader([x for x in zip(train_data, labels)], batch_size=256)
# some basic hyper parameters

e = 30

learning_rate = 1e-3



# define the loss and optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss = nn.CrossEntropyLoss()
global_step = 0

for epoch in range(e):

    model.train()

    for img, label in data_loader:

        img, label = img.to(device), label.to(device)



        res = model(img)

        loss_val = loss(res, label)

        

        optimizer.zero_grad()

        loss_val.backward()

        optimizer.step()

        

        if global_step % 256 == 0:

            print(loss_val)

        

        global_step += 1

    

    
# evaluate the model

model.eval()

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test.head()
test_tensor = torch.Tensor(test.values)

test_ = test_tensor.reshape(-1,1,28,28)

test_data = torch.utils.data.DataLoader(test_, batch_size=1)
import matplotlib.pyplot as plt



predictions = []

image_id = []

i = 0



for image in test_data:



    res = model(image.to(device))

    predictions.append(torch.argmax(res).item())

    img = image.reshape(28,28)



    # imgplot = plt.imshow(img.numpy(), cmap='gray')

    # plt.show()

    # print(predictions)

    

    # need the image id for the submission format

    i += 1

    image_id.append(i)

    
df = pd.DataFrame({'ImageId': image_id,'Label': predictions})

df.to_csv('/kaggle/working/submission.csv', index=False)