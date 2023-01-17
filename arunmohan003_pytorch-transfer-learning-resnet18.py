#Import necessary libraries

from torch import nn

import numpy as np

import torch

import torch.nn.functional as F

import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from  torch import optim 

from torchvision import transforms, utils, models



from collections import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = "/kaggle/input/dogs-cats-images/dataset/"



train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                transforms.RandomHorizontalFlip(),

                                transforms.Resize((280,280)),

                                transforms.ToTensor(),

                                transforms.Normalize([0.5, 0.5, 0.5],

                                [0.5, 0.5, 0.5])])

test_transforms = transforms.Compose([

                                transforms.Resize((280,280)),

                                transforms.ToTensor(),

                                transforms.Normalize([0.5, 0.5, 0.5],

                                [0.5, 0.5, 0.5])])



train_data = datasets.ImageFolder(data_dir + 'training_set', transform=train_transforms)

test_data = datasets.ImageFolder(data_dir + 'test_set', transform=test_transforms)
for image,label in train_data:

    print(image.shape)

    break

    
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)

testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64,shuffle=True)
iter_ = iter(trainloader)

image,label = next(iter_)

print(image.shape)

print(label)

def image_convert(img):

    img = img.clone().cpu().numpy()

    img = img.transpose(1,2,0)

    std = [0.5,0.5,0.5]

    mean = [0.5,0.5,0.5]

    img = img*std + mean

    return img





def plot_10():

        iter_ = iter(trainloader)

        images,labels = next(iter_)

        an_ = {'0':'cat','1':'dog'}

        

        plt.figure(figsize=(20,10))

        for idx in range(10):

            plt.subplot(2,5,idx+1)

            img = image_convert(images[idx])

            label = labels[idx]

            plt.imshow(img)

            plt.title(an_[str(label.numpy())])

        plt.show()

            

    
plot_10()
model =  models.resnet18(pretrained=True).to(device)

print(model)
from torch import nn

class Dc_model(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear1=nn.Linear(512,120)

        self.linear2=nn.Linear(120,2)

    

    def forward(self,x):

        x = F.relu(self.linear1(x))

        x = self.linear2(x)

        return x
model_ = Dc_model().to(device)

model_
model.fc = model_

print(model)
for param in model.parameters():

        param.requires_grad = False

for param in model.fc.parameters():

        param.requires_grad = True
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)
train_loss = []

val_loss = []



epochs = 5



for epoch in range(epochs):

      print("epoch {}/{}".format(epoch+1,epochs))

      running_loss = 0.0

      running_score = 0.0

#       model.train()

      for image,label in trainloader:

          image = image.to(device)

          label = label.to(device)

          optimizer.zero_grad()

          y_pred = model.forward(image)

          loss = criterion(y_pred,label)         

          loss.backward() #calculate derivatives 

          optimizer.step() # update parameters

          val, index_ = torch.max(y_pred,axis=1)

          running_score += torch.sum(index_ == label.data).item()

          running_loss += loss.item()

      

      epoch_score = running_score/len(trainloader.dataset)

      epoch_loss = running_loss/len(trainloader.dataset)

      train_loss.append(epoch_loss)

      print("Training loss: {}, accuracy: {}".format(epoch_loss,epoch_score))

      

      with torch.no_grad():

          model.eval()

          running_loss = 0.0

          running_score = 0.0

          for image,label in testloader:

                image = image.to(device)

                label = label.to(device)

                optimizer.zero_grad()

                y_pred = model.forward(image)

                loss = criterion(y_pred,label)

                running_loss += loss.item()



                val, index_ = torch.max(y_pred,axis=1)

                running_score += torch.sum(index_ == label.data).item()

          

          epoch_score = running_score/len(testloader.dataset)

          epoch_loss = running_loss/len(testloader.dataset)

          val_loss.append(epoch_loss)

          print("Validation loss: {}, accuracy: {}".format(epoch_loss,epoch_score))

plt.plot(train_loss,label='train loss')

plt.plot(val_loss,label='test loss')

plt.legend()

plt.show()
def plot_val_images():



    label_dict = ['cat','dog']

 

    iter_ = iter(testloader)

    images,labels = next(iter_)

    images = images.to(device)

    pred_labels = labels.to(device)



    

    img_out = model.forward(images)

    value, index_val = torch.max(img_out, 1)



    # label = label_dict[str(label)]

    fig = plt.figure(figsize=(35,9))

    for idx in np.arange(10):

        ax = fig.add_subplot(2,5,idx+1)

        plt.imshow(image_convert(images[idx]))

        label = labels[idx]  

        pred_label = pred_labels[idx]

        ax.set_title('Act {},pred {}'.format(label_dict[label],label_dict[pred_label]))
plot_val_images()