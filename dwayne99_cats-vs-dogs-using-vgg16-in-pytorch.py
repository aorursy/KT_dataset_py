# %%capture is to hide the output

%%capture

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%capture

!unzip '../input/dogs-vs-cats/train'

!unzip '../input/dogs-vs-cats/test1'
!mkdir train_
import shutil

import os



os.mkdir('train_/cat/')

os.mkdir('train_/dog/')



for f in os.listdir('train'):

    if f.split('.')[0] == 'cat':

        shutil.move('train/'+f,'train_/cat/'+f)

    else:

        shutil.move('train/'+f,'train_/dog/'+f)
os.rmdir('train')
print(f"Number of testing imgaes : {len(os.listdir('test1'))}")
import torch

from torchvision import transforms, datasets,models

import numpy as np

import pandas as pd

from torch import nn,optim

from PIL import Image
train_dir = 'train_'





# We create the transforms for train (Data Augmentation)

train_transform = transforms.Compose([

  transforms.Resize(256),

  transforms.CenterCrop(224),

  transforms.RandomRotation(30),

  transforms.RandomHorizontalFlip(),

  transforms.ToTensor(),

  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])





# Creating Train DataSet

train_dataset = datasets.ImageFolder(train_dir,transform=train_transform)



# Creating a Data Generator (to obtain data in batches)

train_dataloader = torch.utils.data.DataLoader(

  train_dataset,  

  batch_size = 128,

  shuffle = True

)
import matplotlib.pyplot as plt



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







images, labels = next(iter(train_dataloader))



title = 'Dog' if labels[0].item() == 1 else 'Cat'

imshow(images[0])
model = models.vgg16(pretrained=True)

model
# Freeze our feature parameters as we don't wanna retrain them to the new data

for param in model.parameters():

  param.requires_grad = False
from collections import OrderedDict



classifier = nn.Sequential(OrderedDict([

  # Layer 1

  ('dropout1',nn.Dropout(0.3)),

  ('fc1', nn.Linear(25088,500)),

  ('relu', nn.ReLU()),

  # output layer

  ('fc2', nn.Linear(500,2)),

  ('output', nn.LogSoftmax(dim=1))

]))



# Attach the classifier to the model

model.classifier = classifier
# Loss

criterion = nn.NLLLoss()



# Optimizer 

optimizer = optim.Adam(model.classifier.parameters(),lr =0.001)
# Lets use GPU if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
# Moving model to GPU

model = model.to(device)
from tqdm import tqdm

epochs = 5





for e in range(epochs):

  running_loss, total, correct = 0, 0 , 0



  model.train()

    

  for images,labels in tqdm(train_dataloader):

    

    # Moving input to GPU

    images, labels = images.to(device), labels.to(device)



    # Forward prop

    outputs = model(images)

    loss = criterion(outputs,labels)



    # Backward prop

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    # Metrics 

    running_loss += loss.item()

    total += labels.size(0)



    _, predicted = torch.max(torch.exp(outputs).data,1)

    correct += (predicted == labels).sum().item()

  else:

    # Logs 

    print(f'Epoch {e} Training: Loss={running_loss:.5f} Acc={correct/total * 100:.2f}')

# The checkpoints dictionary will consist of necessary details for rebuilding the model with pretrained weights

checkpoints = {

     'pre-trained':'vgg16',

     'classifier':nn.Sequential(OrderedDict([

          # Layer 1

          ('dropout1',nn.Dropout(0.3)),

          ('fc1', nn.Linear(25088,500)),

          ('relu', nn.ReLU()),

          # output layer

          ('fc2', nn.Linear(500,2)),

          ('output', nn.LogSoftmax(dim=1))

    ])),

    'state_dict':model.state_dict()

}



torch.save(checkpoints,'vgg16_catsVdogs.pth')
# Loading and Rebuilding the saved model



def load_saved_model(path):

    

    # Loading the checkpoint dictionary

    checkpoint = torch.load(path)

    

    # Loading features of the pretrained vgg16

    model = models.vgg16(pretrained=True)

    for param in model.parameters():

        param.requires_grad = False

        

    # Reconstruct the classifier by loading the structure from checkpoint

    model.classifier = checkpoint['classifier']

    

    # Loading the weights

    model.load_state_dict(checkpoints['state_dict'])

    

    # Set model to Evaluation mode to avoid training

    model.eval()

    

    return model
loaded_model = load_saved_model('vgg16_catsVdogs.pth')

loaded_model.to(device)
loaded_model
# Test Transform

test_transform = transforms.Compose([

          transforms.Resize(256),

          transforms.CenterCrop(224),

          transforms.ToTensor(),

          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])



predictions = []



for i in tqdm(range(1,12501)):

    path = 'test1/'+str(i)+'.jpg'

    X = Image.open(path).convert('RGB')

    X = test_transform(X)[:3,:,:]

    X = X.unsqueeze(0)

    X = X.to(device)

    outputs = loaded_model(X)

    predictions.append(torch.argmax(outputs).item())

    
# Making the subimission

data = {'id':list(range(1,12501)),'label':predictions}

df = pd.DataFrame(data)

df.shape
df.to_csv('cats-dogs-submission.csv',index=False)