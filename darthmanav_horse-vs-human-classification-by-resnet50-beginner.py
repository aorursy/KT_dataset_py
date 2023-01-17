%%time
!pip install '/kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4'
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import numpy as np
from torch import nn

import time
import os
import copy
import pretrainedmodels
import pretrainedmodels.utils as utils
from shutil import copyfile
os.environ['TORCH_HOME'] = '/kaggle/working/pretrained-model-weights-pytorch'
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device
data_dir = "/kaggle/input/horses-or-humans-dataset/horse-or-human/"

train_transforms= transforms.Compose([transforms.RandomRotation(30),
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

train_data=datasets.ImageFolder(data_dir+ 'train',transform=train_transforms)
test_data=datasets.ImageFolder(data_dir+ 'validation',transform=test_transforms)
for image,label in train_data:
    print(image.shape)
    break
trainloader=torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True)
testloader=torch.utils.data.DataLoader(dataset=test_data,batch_size=64,shuffle=True)
def copy_weights(model_name):
    found = False
    for dirname, _, filenames in os.walk('/kaggle/input/pretrained-model-weights-pytorch'):
        for filename in filenames:
            full_path = os.path.join(dirname, filename)
            if filename.startswith(model_name):
                found = True
                break
        if found:
            break
            
    base_dir = "/kaggle/working/pretrained-model-weights-pytorch/checkpoints"
    os.makedirs(base_dir, exist_ok=True)
    filename = os.path.basename(full_path)
    copyfile(full_path, os.path.join(base_dir, filename))
print(pretrainedmodels.model_names)
copy_weights('resnet50')
model_name = 'resnet50' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()
tf_img = utils.TransformImage(model)
model
model =  model.to(device)
model
class attaching_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(2048,512)
        self.linear2=nn.Linear(512,256)
        self.linear3=nn.Linear(256,64)
        self.linear4=nn.Linear(64,16)
        self.linear5=nn.Linear(16,2)
    
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x
model_ = attaching_model().to(device)
model_
model.last_linear=model_
print(model)
model.to(device)
for param in model.parameters():
        param.requires_grad = False
for param in model.last_linear.parameters():
        param.requires_grad = True
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.RMSprop(model.parameters(),lr=0.001)
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
import matplotlib.pyplot as plt
plt.plot(train_loss,label='train loss')
plt.plot(val_loss,label='test loss')
plt.legend()
plt.show()
class_names = train_data.classes
class_names
def imshow(inp, model=model, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = model.std * inp + model.mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
visualize_model(model)