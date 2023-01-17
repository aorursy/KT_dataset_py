import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np 

import pandas as pd 

import torch 

import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

import helper

import matplotlib.pyplot as plt

import IPython.display as ipd

import seaborn as sns



import warnings

import os

warnings.filterwarnings('ignore')



from torch import nn, optim

from torchvision import transforms, models, datasets



from fastai.callbacks import *

from sklearn.metrics import roc_curve, auc

from fastai.vision import *

sns.set(style='whitegrid')

plt.style.use('seaborn-darkgrid')



print(os.listdir("../input"))



data_dir = '../input/chest_xray/chest_xray'

train_dir = data_dir + '/train'

valid_dir = data_dir + '/val'

test_dir = data_dir + '/test'
train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],

                                                           [0.229, 0.224, 0.225])])

                                    

valid_transforms = transforms.Compose([transforms.Resize(254),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406], 

                                                           [0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(256),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406], 

                                                           [0.229, 0.224, 0.225])])



trainset = datasets.ImageFolder(train_dir,transform=train_transforms)

validset = datasets.ImageFolder(valid_dir,transform=valid_transforms )

testset = datasets.ImageFolder(test_dir,transform=valid_transforms )





trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

validloader = torch.utils.data.DataLoader(validset, batch_size=64)

testloader = torch.utils.data.DataLoader(testset, batch_size=64)

def imshow(img):

    img = img / 2 + 0.5

    plt.imshow(np.transpose(img, (1,2,0)));

    

dataiter = iter(validloader)

images, labels = dataiter.next()

images = images.numpy()



fig = plt.figure(figsize=(20,5))



for idx in np.arange(16):

    ax = fig.add_subplot(2, 16/2, idx+1, xticks=[], yticks=[]);

    imshow(images[idx]);

print(labels)  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
model = models.resnet34(pretrained=True)

print(model)
for param in model.parameters():

    param.requires_grad = False



classifier = nn.Sequential(nn.Linear(512, 200),

                           nn.ReLU(),

                          nn.Dropout(p=0.5),

                          nn.Linear(200, 2),

                          nn.LogSoftmax(dim=1))

model.fc = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

model.to(device);
def validation(model, validloader, criterion):

    model.to(device)

    valid_loss = 0

    accuracy = 0

    for data in validloader:

        images, labels = data

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)

        valid_loss += criterion(output, labels).item()

        

        ps = torch.exp(output)

        equality = (labels.data == ps.max(dim=1)[1])

        accuracy += equality.type(torch.FloatTensor).mean()

    

    return valid_loss, accuracy
epochs = 10

print_every = 20

steps = 0



model.to(device)



for e in range(epochs):

    model.train()

    running_loss = 0

    for ii, (images, labels) in enumerate(trainloader):

        steps += 1



        images, labels = images.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        outputs = model.forward(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

        if steps % print_every == 0:

            

            model.eval()

            

            # Turn off gradients for validation, saves memory and computations

            with torch.no_grad():

                valid_loss, accuracy = validation(model, validloader, criterion)

                

            print("Epoch: {}/{}.. ".format(e+1, epochs),

                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),

                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),

                  "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

            

            running_loss = 0

            

            # Make sure training is back on

            model.train()
model.eval()

    

with torch.no_grad():

    _, accuracy = validation(model, testloader, criterion)

                

print("Test Accuracy: {:.2f}%".format(accuracy*100/len(testloader)))
path = Path('../input/chest_xray/chest_xray/')

np.random.seed(42)



data = ImageDataBunch.from_folder(path, train='train', valid_pct=0.2, ds_tfms=get_transforms(),size=224, bs=64).normalize(imagenet_stats)

data
data.classes
data.show_batch(4, figsize=(15,10))
learn = create_cnn(data, models.alexnet, ps=0.5, model_dir="/tmp/model/", metrics=error_rate)

learn
learn.fit_one_cycle(6, 1e-2)

learn.recorder.plot_losses()

plt.show()
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)

Learning_rate = learn.recorder.min_grad_lr

print(Learning_rate)

plt.show()
learn.fit_one_cycle(3, Learning_rate)



plt.show()
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(4, figsize=(10,8), heatmap=False)

plt.show()
learn.save('pneumonia-model')
interp.plot_confusion_matrix(figsize=(10, 8), dpi=60)

plt.show()

interp.most_confused()
learn.show_results()