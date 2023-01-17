# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch

from torch import optim, nn

import torchvision

from torchvision import datasets, models, transforms

import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
train_dir = '../input/car_data/car_data/train'

test_dir = '../input/car_data/car_data/test'
train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(train_dir , transform=train_transforms)

test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
print(len(train_data))

print(len(test_data))
valid_size = 0.2

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,sampler=train_sampler)

validloader = torch.utils.data.DataLoader(train_data, batch_size=64, sampler=valid_sampler)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
print(len(trainloader))

print(len(validloader))

print(len(testloader))
print(len(train_data)/64)

print(len(test_data)/64)
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
images, labels = next(iter(trainloader))

import torchvision

grid = torchvision.utils.make_grid(images, nrow = 20, padding = 2)

plt.figure(figsize = (15, 15))  

plt.imshow(np.transpose(grid, (1, 2, 0)))   

print('labels:', labels)
for i in range(2):

    imshow(images[i])
train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
from torch.optim import lr_scheduler



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = models.resnet152(pretrained=True)



for param in model.parameters():

    param.requires_grad = False

        

model.fc = nn.Sequential(nn.Linear(2048, 512),nn.ReLU(),nn.Linear(512,196),nn.LogSoftmax(dim=1))    



criterion = nn.NLLLoss()





optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.fc.parameters()) , lr = 0.001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)





model.to(device);
def train_and_test(e):

    epochs = e

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

            print(f" epoch {epoch + 1} batch {batch} completed")

      test_loss = 0

      accuracy = 0

      with torch.no_grad():

        model.eval() 

        for images , labels in validloader:

          images, labels = images.to(device), labels.to(device)

          logps = model(images) 

          test_loss += criterion(logps,labels) 

          ps = torch.exp(logps)

          top_p , top_class = ps.topk(1,dim=1)

          equals = top_class == labels.view(*top_class.shape)

          accuracy += torch.mean(equals.type(torch.FloatTensor))

      train_losses.append(running_loss/len(trainloader))

      test_losses.append(test_loss/len(validloader))

      print("Epoch: {}/{}.. ".format(epoch+1, epochs),"Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),"Valid Loss: {:.3f}.. ".format(test_loss/len(validloader)),

        "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

      model.train() 

      if test_loss/len(validloader) <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,test_loss/len(validloader))) 

        torch.save(model.state_dict(), path)

        valid_loss_min = test_loss/len(validloader)
model_save_name = 'checkpoint.pth'

path = F"/kaggle/working/{model_save_name}"
def load_model():

    torch.load(path)
load_model()
os.listdir('/kaggle/working')
accuracy = 0

with torch.no_grad():

    model.eval()

    p_labels = []

    img_ids = []

    i = 0

    for inputs, labels in testloader:

        i += 1

        inputs = inputs.to(device)

        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        temp_acc = torch.sum(preds == labels.data)

        accuracy += temp_acc

        p_labels.append(preds)

        if i % 50 == 0:

            print(f"batch {i} completed...")

    for dir in os.listdir(test_dir):

        for file in os.listdir(os.path.join(test_dir, dir)):

            img_id = os.path.splitext(file)[0]

            img_ids.append(img_id)

    print('Accuracy =====>>', accuracy.item()/len(test_data))
pred_labels_expanded = []

for l in p_labels:

    for l1 in l:

        pred_labels_expanded.append(l1.item())

submission = pd.DataFrame({'Id': img_ids,'Predicted': pred_labels_expanded })
print(submission.head())
submission.to_csv('submission.csv', index=False)