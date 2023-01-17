!pip install torch==1.4.0 torchvision==0.5.0
!unzip ../input/dogs-vs-cats/train.zip
"""

All necessary imports

"""



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import torch

import torch.nn.functional as F

from torch import nn, optim

from torchvision import transforms, datasets, models



from PIL import Image

from collections import OrderedDict

import os, shutil, random, time

from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
"""

check duplicates

"""



all_files = os.listdir('train')



print(len(all_files))

print(len(np.unique(all_files)))
"""

shuffle and organize into train/test dirs

"""



random.shuffle(all_files)



dog_files = []

cat_files = []

err_cntr = 0

for file in all_files:

    if 'dog' in file:

        dog_files.append(file)

    elif 'cat' in file:

        cat_files.append(file)

    else:

        err_cntr = 0

        

print("="*60)

print('cat images:', len(cat_files))

print('dog images:', len(dog_files))

print('other files:', err_cntr)

print("="*60)



plt.title('Data distribution')

plt.ylabel('No. of images')

plt.bar(

    ['dog images', 'cat_images'],

    [len(dog_files), len(cat_files)]

)

plt.show()







"""

split and move to train/test dirs

"""



ratio = 0.8

lim = int(0.8*len(dog_files)) # prefectly balanced i.e len(dog_files) = len(cat_files)



for _dir in ['traindata', 'testdata']:

    os.makedirs(f"{_dir}/dog", exist_ok=True)    

    os.makedirs(f"{_dir}/cat", exist_ok=True)

    

train_dog_files = dog_files[:lim]

test_dog_files  = dog_files[lim:]

train_cat_files = cat_files[:lim]

test_cat_files  = cat_files[lim:]



for file in test_dog_files:

    shutil.move(f'train/{file}', f'testdata/dog/{file}')

for file in test_cat_files:

    shutil.move(f'train/{file}', f'testdata/cat/{file}')

for file in train_dog_files:

    shutil.move(f'train/{file}', f'traindata/dog/{file}')

for file in train_cat_files:

    shutil.move(f'train/{file}', f'traindata/cat/{file}')
for _dir in ['traindata', 'testdata']:

    print(f'+ {_dir}')

    dog_cnt = len(os.listdir(f'{_dir}/dog/'))

    cat_cnt = len(os.listdir(f'{_dir}/cat/'))

    print('\t+ Dog images: ', dog_cnt)    

    print('\t+ Cat images: ', cat_cnt)
"""

Data loader w/ augmentation and Preprocession 

"""



train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(255),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],

                                                           [0.229, 0.224, 0.225])])





train_data = datasets.ImageFolder('traindata/', transform=train_transforms)

test_data = datasets.ImageFolder('testdata/', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
x, y = next(iter(trainloader))

print(x.shape, y.shape)
"""

visualize preprocessed images

"""



row, col = 1, 3

fig, axarr = plt.subplots(row, col)

axarr = axarr.flatten()



for i in range(row*col):

    axarr[i].set_title('Dog' if y[i] == 1 else 'Cat')

    axarr[i].imshow(x[i].numpy().transpose((1, 2, 0)).astype('uint8'))

    axarr[i].axis('off')



plt.show()



# Note: preprocessed as in official docs. 
"""

define architecture

"""

model = models.densenet121(pretrained=True).to(device)



"""

freeze extractor and append classifier

"""

for param in model.parameters():

    param.requires_grad = False



from collections import OrderedDict

classifier = nn.Sequential(

    OrderedDict([

        # ----------------------------

        ('fc1', nn.Linear(1024, 512)),

        ('relu1', nn.ReLU()),

        # ----------------------------

        ('fc2', nn.Linear(512,256)),

        ('relu2', nn.ReLU()),

        # ----------------------------

        ('fc3', nn.Linear(256, 2)),

        ('output', nn.LogSoftmax(dim=1))

        # ----------------------------

    ]))

    

model.classifier = classifier.to(device)
"""

Display architecture

"""



model
"""

test behaviour

"""



img = torch.rand(64, 3, 224, 224).to(device)

model(img).shape
"""

Configure training parameters

"""



criterion = nn.NLLLoss().to(device)

optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
"""

Start training

"""



# history for post-training visualisation

class hist:

    traininglosses = []

    testinglosses = []

    testaccuracy = []

    totalsteps = []



epochs = 1

steps = 0



running_loss = 0

print_every = 50



for epoch in range(epochs):

    for inputs, labels in trainloader:

        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

            

        optimizer.zero_grad()

        

        # forward prop

        logps = model.forward(inputs)

        loss = criterion(logps, labels)

        loss.backward()

        optimizer.step()



        # avg taken every `print_every` step

        # and logged

        running_loss += loss.item()

        

        # Evaluate on testloader

        # and log after every `print_every` step

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0

            model.eval()

            with torch.no_grad():

                for inputs, labels in testloader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    # predictions in log scale

                    # Take exponents and get preds

                    # using `topk` which returns 

                    # top-preds and respective top-classes

                    logps = model.forward(inputs)

                    # Loss and simple categorical accuracy 

                    # accumulated for avg taken to be taken 

                    # for whole epoch

                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)

                    _, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            

            # Change back to train after evaluation

            model.train()

            running_loss = 0

            

            # ------------------------------------------------------------------

            # log performance ans populate history for vsiualisation

            hist.traininglosses.append(running_loss/print_every)

            hist.testinglosses.append(test_loss/len(testloader))

            hist.testaccuracy.append(accuracy/len(testloader))

            hist.totalsteps.append(steps)

            print(f"Epoch {epoch+1}of{epochs} "

                  f"Step {steps} \t"

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(testloader):.3f}.. "

                  f"Test accuracy: {accuracy/len(testloader):.3f}.. ")
plt.plot(hist.totalsteps, hist.traininglosses, label='Train Loss')

plt.plot(hist.totalsteps, hist.testinglosses, label='Test Loss')

plt.plot(hist.totalsteps, hist.testaccuracy, label='Test Accuracy')



plt.title('First epoch')

plt.xlabel('Steps')



plt.legend()

plt.grid()

plt.show()