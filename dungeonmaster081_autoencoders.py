import numpy as np

import pandas as pd

import os

from statistics import mean

from itertools import islice



from torch.utils.data.dataset import Dataset

from torch.utils.data.dataloader import DataLoader



from torchvision import transforms

from torchvision.utils import save_image

import torchvision



import torch

from torch import nn, optim

import torch.nn.functional as F



import matplotlib.pyplot as plt

%matplotlib inline
print(os.listdir('../input/'))
# here we will be using Kuzushiji-MNIST dataset which is having 70K images

train_labels = np.load('../input/kmnist-train-labels.npz')['arr_0']

test_labels = np.load('../input/kmnist-test-labels.npz')['arr_0']



train_images = np.load('../input/kmnist-train-imgs.npz')['arr_0']

test_images = np.load('../input/kmnist-test-imgs.npz')['arr_0']
char_df = pd.read_csv('../input/kmnist_classmap.csv')

print(char_df)
classes, count = np.unique(train_labels,return_counts=True)

print(classes,'\n', count)

# so the dataset is properly balanced among the classes
# also lets chech the distribution in the test set.

np.unique(test_labels,return_counts=True)
len(train_images)
# find the size of the images

print(train_images.shape)

print(test_images.shape)

print(train_labels.shape)

print(test_labels.shape)
# lets also create a visualisation fucntion

def plot_images(images, labels,random=False):

    if random ==  False:

        plt.figure(figsize=(12,12))

        for i in range(10):

            imgs = images[np.where(labels==i)]

            lbls = labels[np.where(labels==i)]

            for j in range(10):

                plt.subplot(10,10,10*i+j+1)

                plt.xticks([])

                plt.yticks([])

                plt.grid(False)

                plt.imshow(imgs[j],cmap=plt.cm.binary)

                plt.xlabel(lbls[j])

    else:

        plt.figure(figsize=(12,12))

        for i in range(10):

            imgs = images[np.where(labels==i)]

            lbls = labels[np.where(labels==i)]

            for j in range(10):

                plt.subplot(10,10,10*i+j+1)

                plt.xticks([]);plt.yticks([])

                plt.grid(False)

                index = np.random.randint(1, 10)

                plt.imshow(imgs[index],cmap=plt.cm.binary)

                plt.xlabel(lbls[index])
plot_images(train_images,train_labels, random=True)
#plot_images(test_images,test_labels)
class KMNIST_Dataset(Dataset):

    def __init__(self, images, labels, transform=None):

        self.images = images

        self.labels = labels

        self.transform = transform

        

    def __len__(self):

        return (len(self.images))

    

    def __getitem__(self, index):

        image = self.images[index].reshape(28,28,1)

        label = self.labels[index]

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image,label
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = KMNIST_Dataset(train_images, train_labels, transform=transform)
train_gen = DataLoader(train_dataset, batch_size=16, shuffle=True)

train_iter = iter(train_gen)

images, labels = next(train_iter)
print(images.size(),labels.size())
grid = torchvision.utils.make_grid(images)



plt.imshow(grid.numpy().transpose((1, 2, 0)))

plt.axis('off')

plt.title(labels.numpy());
class autoencoder(nn.Module):

    def __init__(self):

        super(autoencoder,self).__init__()

        # encoder 

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),  # b x 28 x 28 x 32     

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),               # b x 14 x 14 x 32

            nn.Conv2d(32, 16, 3, padding=1), # b x 14 x 14 x 16

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2),               # b x 7 x 7 x 16

            nn.Conv2d(16, 8, 3, padding=1),  # b x 7 x 7 x 8

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)                # b x 3 x 3 x 8

        )

        # decoder

        self.decoder = nn.Sequential(

            nn.Conv2d(8, 8, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(8, 16, 3, padding=1),

            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(16, 32, 3,padding=1),

            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(32, 1, 3, padding=3),    # padding = 3, was used to make the input and output of same size

            nn.Sigmoid()

        )

        

        

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x
epochs = 100

lr = 0.001
def to_img(x):

    x = 0.5 * (x + 1)

    x = x.clamp(0, 1)

    x = x.view(x.size(0), 1, 28, 28)

    return x
model = autoencoder().cuda()

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr,

                             weight_decay=1e-5)

losses = []

for epoch in range(epochs):

    running_loss = 0

    for img, _ in train_gen:

        img = img.to('cuda')

        output = model(img)

        optimizer.zero_grad()  

        #print(output.shape,img.shape)

        loss = criterion(output, img)

        running_loss += loss.item()

        #losses.append(loss.item())

        loss.backward()

        optimizer.step()

    

    print('epoch [{}/{}], loss:{:.4f}'

          .format(epoch+1, epochs, running_loss/len(train_gen)))

    losses.append(running_loss/len(train_gen))

    if epoch % 10 == 0:

        pic = to_img(output.cpu().data)

        save_image(pic, 'image_{}.png'.format(epoch))



torch.save(model.state_dict(), './conv_autoencoder.pth')
plt.plot(losses)