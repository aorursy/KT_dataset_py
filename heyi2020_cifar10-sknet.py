import os 

print(os.listdir("../input"))
import torch

import torch.nn.functional as F

from torchvision import datasets,transforms

from torch import nn

import matplotlib.pyplot as plt

import numpy as np

import torchvision.utils as vutils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
# Dowload the dataset

# from torchvision.datasets.utils import download_url 

# dataset_url = "http://files.fast.ai/data/cifar10.tgz" 

# download_url(dataset_url, '.') 

# import tarfile 

# Extract from archive 

# with tarfile.open('./cifar10.tgz', 'r:gz') as tar: 

#     tar.extractall(path='./data')
transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.

                                      #transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis

                                      #transforms.RandomRotation(10),     #Rotates the image to a specified angel

                                      #transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.

                                      #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params

                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch

                                      #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images

                                      ])

transform = transforms.Compose([transforms.Resize((32,32)),

                               transforms.ToTensor(),

                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                               ])

training_dataset = datasets.CIFAR10(root='./data', train=True, download=True,transform=transform_train) # Data augmentation is only done on training images

validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True,transform=transform)

print(training_dataset,validation_dataset)

# !pip install d2lzh

# import d2lzh as d2l 

# import os 

# import pandas as pd 

# import shutil 

# import time
# demo = True 

# if demo:     

#     import zipfile     

#     for f in ['train_tiny.zip', 'test_tiny.zip', 'trainLabels.csv.zip']:         

#         with zipfile.ZipFile('../data/kaggle_cifar10/' + f, 'r') as z:             

#             z.extractall('../data/kaggle_cifar10/')
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True) # Batch size of 100 i.e to work with 100 images at a time 

validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 100, shuffle=False)
training_loader.__iter__().__next__()[0].shape
# We need to convert the images to numpy arrays as tensors are not compatible with matplotlib.

def im_convert(tensor):     

    image = tensor.cpu().clone().detach().numpy() # This process will happen in normal cpu.

    image = image.transpose(1, 2, 0)   

#     image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))   

#     image = image.clip(0, 1)   

    return image



classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# We iter the batch of images to display 

dataiter = iter(training_loader) # converting our train_dataloader to iterable so that we can iter through it.

images, labels = dataiter.next() #going from 1st batch of 100 images to the next batch 

print(images.shape)

fig = plt.figure(figsize=(25, 4))   # We plot 20 images from our train_dataset 

# plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True),(1,2,0)))

# plt.show()

for idx in np.arange(20):   

    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])    

#     plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True),(1,2,0)))

    plt.imshow(images[idx].permute(1,2,0)) #converting to numpy array as plt needs it.

    ax.set_title(classes[labels[idx].item()]) 


class SKConv(nn.Module):

    def __init__(self, features, WH, M, G, r, stride=1 ,L=32):

            """ Constructor

            Args:

                features: input channel dimensionality.

                WH: input spatial dimensionality, used for GAP kernel size.

                M: the number of branchs.

                G: num of convolution groups.

                r: the radio for compute d, the length of z.

                stride: stride, default 1.

                L: the minimum dim of the vector z in paper, default 32.

            """

            super(SKConv, self).__init__()

            d = max(int(features/r), L)

            self.M = M

            self.features = features

            self.convs = nn.ModuleList([])

            for i in range(M):

                self.convs.append(nn.Sequential(

                    nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),#使用组卷积以减少参数

                    nn.BatchNorm2d(features),

                    nn.ReLU(inplace=False)

                ))

            # self.gap = nn.AvgPool2d(int(WH/stride))

            self.fc = nn.Linear(features, d)

            self.fcs = nn.ModuleList([])

            for i in range(M):

                self.fcs.append(

                    nn.Linear(d, features)

                )

            self.softmax = nn.Softmax(dim=1)



    def forward(self, x):

        for i, conv in enumerate(self.convs):

            fea = conv(x).unsqueeze_(dim=1)

            if i == 0:

                feas = fea

            else:

                feas = torch.cat([feas, fea], dim=1)

        fea_U = torch.sum(feas, dim=1)

        # fea_s = self.gap(fea_U).squeeze_()

        fea_s = fea_U.mean(-1).mean(-1)

        fea_z = self.fc(fea_s)

        for i, fc in enumerate(self.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:

                attention_vectors = vector

            else:

                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)

        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)

        fea_v = (feas * attention_vectors).sum(dim=1)

        return fea_v





class SKUnit(nn.Module):

    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):

        """ Constructor

        Args:

            in_features: input channel dimensionality.

            out_features: output channel dimensionality.

            WH: input spatial dimensionality, used for GAP kernel size.global adaptive pool

            M: the number of branchs.

            G: num of convolution groups.

            r: the radio for compute d, the length of z.

            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.

            stride: stride.

            L: the minimum dim of the vector z in paper.

        """

        super(SKUnit, self).__init__()

        if mid_features is None:

            mid_features = int(out_features/2)

        self.feas = nn.Sequential(

            nn.Conv2d(in_features, mid_features, 1, stride=1), #1*1卷积

            nn.BatchNorm2d(mid_features),

            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),

            nn.BatchNorm2d(mid_features),

            nn.Conv2d(mid_features, out_features, 1, stride=1),

            nn.BatchNorm2d(out_features)

        )

        if in_features == out_features: # when dim not change, in could be added diectly to out

            self.shortcut = nn.Sequential()

        else: # when dim not change, in should also change dim to be added to out

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_features, out_features, 1, stride=stride),

                nn.BatchNorm2d(out_features)

            )

    

    def forward(self, x):

        fea = self.feas(x)

        return fea + self.shortcut(x)





class SKNet(nn.Module):

    def __init__(self, class_num):

        super(SKNet, self).__init__()

        self.basic_conv = nn.Sequential(

            nn.Conv2d(3, 64, 3, padding=1),

            nn.BatchNorm2d(64)

        ) # 32x32

        self.stage_1 = nn.Sequential(

            SKUnit(64, 256, 32, 2, 8, 2, stride=1),

            nn.ReLU(),

            SKUnit(256, 256, 32, 2, 8, 2),

            nn.ReLU(),

            SKUnit(256, 256, 32, 2, 8, 2),

            nn.ReLU()

        ) # 32x32

        self.stage_2 = nn.Sequential(

            SKUnit(256, 512, 32, 2, 8, 2, stride=2),

            nn.ReLU(),

            SKUnit(512, 512, 32, 2, 8, 2),

            nn.ReLU(),

            SKUnit(512, 512, 32, 2, 8, 2),

            nn.ReLU()

        ) # 16x16

        self.stage_3 = nn.Sequential(

            SKUnit(512, 1024, 32, 2, 8, 2, stride=2),

            nn.ReLU(),

            SKUnit(1024, 1024, 32, 2, 8, 2),

            nn.ReLU(),

            SKUnit(1024, 1024, 32, 2, 8, 2),

            nn.ReLU()

        ) # 8x8

        self.pool = nn.AvgPool2d(8)

        self.classifier = nn.Sequential(

            nn.Linear(1024, class_num),

            # nn.Softmax(dim=1)

        )



    def forward(self, x):

        fea = self.basic_conv(x)

        fea = self.stage_1(fea)

        fea = self.stage_2(fea)

        fea = self.stage_3(fea)

        fea = self.pool(fea)

        fea = torch.squeeze(fea)

        fea = self.classifier(fea)

        return fea



model = SKNet(10).to(device) # run our model on cuda GPU for faster results

model
criterion = nn.CrossEntropyLoss().cuda()# same as categorical_crossentropy loss used in Keras models which runs on Tensorflow

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # fine tuned the lr
epochs = 15

running_loss_history = []

running_corrects_history = []

val_running_loss_history = []

val_running_corrects_history = []



for e in range(epochs): # training our model, put input according to every batch.

    running_loss = 0.0

    running_corrects = 0.0

    val_running_loss = 0.0

    val_running_corrects = 0.0

  

    for inputs, labels in training_loader:

        inputs = inputs.to(device) # input to device as our model is running in mentioned device.

        labels = labels.to(device)

        outputs = model(inputs) # every batch of 100 images are put as an input.

        loss = criterion(outputs, labels) # Calc loss after each batch i/p by comparing it to actual labels. 

    

        optimizer.zero_grad() #setting the initial gradient to 0

        loss.backward() # backpropagating the loss

        optimizer.step() # updating the weights and bias values for every single step.

    

        _, preds = torch.max(outputs, 1) # taking the highest value of prediction.

        running_loss += loss.item()

        running_corrects += torch.sum(preds == labels.data) # calculating te accuracy by taking the sum of all the correct predictions in a batch.





    with torch.no_grad(): # we do not need gradient for validation.

        for val_inputs, val_labels in validation_loader:

            val_inputs = val_inputs.to(device)

            val_labels = val_labels.to(device)

            val_outputs = model(val_inputs)

            val_loss = criterion(val_outputs, val_labels)



            _, val_preds = torch.max(val_outputs, 1)

            val_running_loss += val_loss.item()

            val_running_corrects += torch.sum(val_preds == val_labels.data)

      

    epoch_loss = running_loss/len(training_loader) # loss per epoch

    epoch_acc = running_corrects.float()/ len(training_loader) # accuracy per epoch

    running_loss_history.append(epoch_loss) # appending for displaying 

    running_corrects_history.append(epoch_acc)

    

    val_epoch_loss = val_running_loss/len(validation_loader)

    val_epoch_acc = val_running_corrects.float()/ len(validation_loader)

    val_running_loss_history.append(val_epoch_loss)

    val_running_corrects_history.append(val_epoch_acc)

    print('epoch :', (e+1))

    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))

    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))



plt.style.use('ggplot')

plt.plot(running_loss_history, label='training loss')

plt.plot(val_running_loss_history, label='validation loss')

plt.legend()
plt.style.use('ggplot')

plt.plot(running_corrects_history, label='training accuracy')

plt.plot(val_running_corrects_history, label='validation accuracy')

plt.legend()
import PIL.ImageOps
import requests

from PIL import Image



url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT76mSMtKQWGstcqGi-0kPWJyVBqz8RCp8SuplMipkidRY0z9Mc&usqp=CAU'

response = requests.get(url, stream = True)

img = Image.open(response.raw)

plt.imshow(img)
img = transform(img)  # applying the transformations on new image as our model has been trained on these transformations

plt.imshow(im_convert(img)) # convert to numpy array for plt
image = img.to(device).unsqueeze(0) # put inputs in device as our model is running there

output = model(image)

_, pred = torch.max(output, 1)

print(classes[pred.item()])


dataiter = iter(validation_loader)

images, labels = dataiter.next()

images = images.to(device)

labels = labels.to(device)

output = model(images)

_, preds = torch.max(output, 1)



fig = plt.figure(figsize=(25, 4))



for idx in np.arange(20):

  ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])

  plt.imshow(im_convert(images[idx]))

  ax.set_title("{} ({})".format(str(classes[preds[idx].item()]), str(classes[labels[idx].item()])), color=("green" if preds[idx]==labels[idx] else "red"))