import torch

from torch import optim, nn

import torchvision

from torchvision import datasets, models, transforms

import numpy as np
!wget -c https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/intro-to-pytorch/helper.py
train_dir = '../input/hackathon-blossom-flower-classification/flower_data/flower_data/train'

valid_dir = '../input/hackathon-blossom-flower-classification/flower_data/flower_data/valid'
train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(train_dir , transform=train_transforms)

test_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
print(len(train_data)/64)

print(len(test_data)/64)
import json



with open('../input/hackathon-blossom-flower-classification/cat_to_name.json', 'r') as f:

    cat_to_name = json.load(f)
# This is the contents of helper.py 

import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

from torch.autograd import Variable





def test_network(net, trainloader):



    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)



    dataiter = iter(trainloader)

    images, labels = dataiter.next()



    # Create Variables for the inputs and targets

    inputs = Variable(images)

    targets = Variable(images)



    # Clear the gradients from all Variables

    optimizer.zero_grad()



    # Forward pass, then backward pass, then update weights

    output = net.forward(inputs)

    loss = criterion(output, targets)

    loss.backward()

    optimizer.step()



    return True

def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))

    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax

def view_recon(img, recon):

    ''' Function for displaying an image (as a PyTorch Tensor) and its

        reconstruction also a PyTorch Tensor

    '''



    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

    axes[0].imshow(img.numpy().squeeze())

    axes[1].imshow(recon.data.numpy().squeeze())

    for ax in axes:

        ax.axis('off')

        ax.set_adjustable('box-forced')  
class_names = train_data.classes
images, labels = next(iter(testloader))
import torchvision

grid = torchvision.utils.make_grid(images, nrow = 20, padding = 2)

plt.figure(figsize = (15, 15))  

plt.imshow(np.transpose(grid, (1, 2, 0)))   

print('labels:', labels)

imshow(images[63])

labels[63].item()
train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
!ls -la
from torch.optim import lr_scheduler



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = models.resnet152(pretrained=True)



for name,child in model.named_children():

  if name in ['layer3','layer4']:

    print(name + 'is unfrozen')

    for param in child.parameters():

      param.requires_grad = True

  else:

    print(name + 'is frozen')

    for param in child.parameters():

      param.requires_grad = False



model.fc = nn.Sequential(nn.Linear(2048, 512),nn.ReLU(),nn.Linear(512,102),nn.LogSoftmax(dim=1))    



criterion = nn.NLLLoss()





optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()) , lr = 0.001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)





model.to(device);
def train_and_test():

    epochs = 10

    train_losses , test_losses = [] , []

    valid_loss_min = np.Inf 

    model.train()

    for epoch in range(epochs):

      running_loss = 0

      batch = 0

      scheduler.step()

      for images , labels in trainloader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        batch += 1

        if batch % 10 == 0:

            print(f" epoch {epoch} batch {batch} completed")

      test_loss = 0

      accuracy = 0

      with torch.no_grad():

        model.eval() 

        for images , labels in testloader:

          images, labels = images.to(device), labels.to(device)

          logps = model(images) 

          test_loss += criterion(logps,labels) 

          ps = torch.exp(logps)

          top_p , top_class = ps.topk(1,dim=1)

          equals = top_class == labels.view(*top_class.shape)

          accuracy += torch.mean(equals.type(torch.FloatTensor))

      train_losses.append(running_loss/len(trainloader))

      test_losses.append(test_loss/len(testloader))

      print("Epoch: {}/{}.. ".format(epoch+1, epochs),"Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),"Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),

        "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

      model.train() 

      if test_loss/len(testloader) <= valid_loss_min:

        print('test loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,test_loss/len(testloader))) 

        torch.save(model.state_dict(), 'checkpoint.pth')

        valid_loss_min = test_loss/len(testloader)
def load_model():

    state_dict = torch.load('checkpoint.pth')

    print(state_dict.keys())

    model.load_state_dict(state_dict)

    
for name,child in model.named_children():

  if name in ['layer1','layer2']:

    print(name + 'is unfrozen')

    for param in child.parameters():

      param.requires_grad = True

  else:

    print(name + 'is frozen')

    for param in child.parameters():

      param.requires_grad = False
load_model()
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()) , lr = 0.00001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
model.class_idx_mapping = train_data.class_to_idx

class_idx_mapping = train_data.class_to_idx

idx_class_mapping = {v: k for k, v in class_idx_mapping.items()}
data_dir = "../input/hackathon-blossom-flower-classification/test set"

valid_data = datasets.ImageFolder(data_dir, transform=test_transforms)

validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
import pandas as pd

import os

def predict(validloader, model_checkpoint, topk=1, device="cuda", idx_class_mapping=idx_class_mapping):

    model.to(device)

    model.eval()

    

    labels=[]

    

    with torch.no_grad():

        for images, _ in validloader:

            images = images.to(device)

            output = model(images)

            ps = torch.exp(output)

            top_p, top_class = ps.topk(1, dim=1) 

            for i in top_class:

                

                labels.append(idx_class_mapping[i.item()] )

                

                      

               

    return labels

class_label=predict(validloader,"checkpoint.pth", idx_class_mapping=idx_class_mapping)
class_label
image_col=[]

for img_filename in os.listdir('../input/hackathon-blossom-flower-classification/test set/test set'):

    image_col.append(img_filename)
category_map = sorted(cat_to_name.items(), key=lambda x: int(x[0]))

plant_name=[]

for label in class_label:

    name=cat_to_name[label]

    

    plant_name.append(name)

    
submission = pd.DataFrame({'image_test': image_col, 'pred_class': class_label,'species': plant_name})

submission.sort_values('image_test')
print(submission.head())
submission.to_csv('my_predictions_test.csv')