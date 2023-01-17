# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy  as np

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

from torch import nn, optim

from sklearn.model_selection import train_test_split

# 图像增广

import torchvision.transforms as transforms

from PIL import Image
train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")

train
label=train["label"]

train=train.drop(labels = ["label"],axis = 1) 

label
train.isnull().any().sum()
test.isnull().any().sum()
train=train/255

test=test/255
# xmean=train.mean()

# xstd=train.std()

# train=(train-xmean)/xstd

# xmean=test.mean()

# xstd=test.std()

# test=(test-xmean)/xstd
train=train.values.reshape(-1,1,28,28) #1,28,28 单通道，28行 28列 # 由于使用图像增广，去掉一维

test=test.values.reshape(-1,1,28,28) #1,28,28 单通道，28行 28列
from keras.preprocessing.image import ImageDataGenerator
#  randomly rotating, scaling, and shifting

# CREATE MORE IMAGES VIA DATA AUGMENTATION

datagen = ImageDataGenerator(

        rotation_range=10,  

        #zoom_range = 0.10,  

        width_shift_range=0.001, 

        height_shift_range=0.001

        )
# 训练时不跑

# b=np.array([datagen.flow(train[i:i+1]).next()[0] for i in range(30)])

# import matplotlib.pyplot as plt

# # PREVIEW IMAGES

# plt.figure(figsize=(15,4.5))

# for i in range(30):  

#     plt.subplot(3, 10, i+1)

#     plt.imshow(b[i].reshape((28,28)),cmap=plt.cm.binary)

#     plt.axis('off')

# plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

# plt.show()
# k=0

# for i in range(1312):

#     image=datagen.flow(train[k:k+32]).next()

#     train=np.concatenate((train, image), axis = 0) 

#     label=pd.concat([label,label[k:k+32]],axis = 0)

#     k=k+32
# transform = transforms.Compose([

#      transforms.ToPILImage(),

#      transforms.RandomRotation((10,80)),

#  ])
# X_train, X_val, Y_train, Y_val = train_test_split(train, label, test_size = 0.1, random_state=2)

# X_train=torch.tensor(X_train, dtype=torch.float)

# Y_train=torch.tensor(Y_train.values, dtype=torch.float)

train=torch.tensor(train, dtype=torch.float)

label=torch.tensor(label.values, dtype=torch.float)
batch_size = 32

train_data = torch.utils.data.TensorDataset(train,label)

train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True )
class cusmodule(nn.Module):

    def __init__(self):

        super(cusmodule,self).__init__()

        self.conv=nn.Sequential(

            nn.Conv2d(1, 16, 3), # in_channels, out_channels, kernel_size

            nn.ReLU(),

            nn.Conv2d(16, 16, 3),

            nn.ReLU(),

            nn.MaxPool2d(2, 2), # kernel_size, stride

            #nn.Dropout(0.4),

            #nn.BatchNorm2d(6),

            nn.Conv2d(16, 32, 3),

            nn.ReLU(),

            nn.Conv2d(32, 32, 3),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            #nn.Dropout(0.4),

            #nn.BatchNorm2d(16),

        )

        self.fc=nn.Sequential(

            nn.Linear(32*4*4,256),

            nn.ReLU(),

            #nn.Dropout(0.4),

            nn.Linear(256,84),

            nn.ReLU(),

            #nn.Dropout(0.4),

            nn.Linear(84,10)

        )

    def forward(self,img):

        feature=self.conv(img)

        output=self.fc(feature.view(img.shape[0], -1))

        return output
lr=0.0015

decay=0

num_epochs=40

net=cusmodule()

loss=nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=lr)
def train_net(net, train_iter,loss, num_epochs, lr, optimizer):

    for epochs in range(num_epochs):

        for x,y in train_iter:

            # 注意最后一次的数据不一定是符合batch_size的 所以用x.shape[0]代替

            if (epochs % 2)==0:

                b=np.array([datagen.flow(x[i:i+1]).next()[0] for i in range(x.shape[0])])

                x=torch.tensor(b, dtype=torch.float)

            y_hat=net(x) #模型计算

            l=loss(y_hat,y.long()) #损失计算

            optimizer.zero_grad() #梯度清0

            l.backward() #反向传播

            optimizer.step()

        print("epochs:"+str(epochs)+" loss:"+str(l))
train_net(net,train_iter,loss, num_epochs, lr, optimizer)
test=torch.tensor(test, dtype=torch.float)
sub=net(test)
sub=sub.argmax(dim=1)

sub=sub.numpy()
sub.reshape(-1,1)
submission=pd.read_csv("../input/digit-recognizer/sample_submission.csv")

submission["Label"]=sub
submission.to_csv('submission.csv', index=False)