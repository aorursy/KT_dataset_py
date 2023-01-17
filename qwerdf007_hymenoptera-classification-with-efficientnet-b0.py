# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install efficientnet_pytorch
import time

import copy

from pathlib import Path



import torch

import torch.optim as optim

import torchvision

from torch.utils.data import DataLoader

from torchvision import transforms, datasets, models

from efficientnet_pytorch import EfficientNet

import matplotlib.pyplot as plt

from PIL import Image
data_transforms = {

    "train": transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(0.5),

        transforms.ColorJitter((0.9,1.1), (0.9,1.1), (0.9,1.1)),

        transforms.ToTensor()

    ]),

    "val": transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor()

    ])

}
data_dir = '/kaggle/input/hymenoptera-data/hymenoptera_data/hymenoptera_data/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

image_datasets
dataloader = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

class_names
def imshow(inp, title=None):

    inp = inp.numpy().transpose(1,2,0)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)





images, classes = next(iter(dataloader['train']))

# out = torchvision.utils.make_grid(images)

# imshow(out, title=[classes[x] for x in classes])

print(images.shape)

print(classes.numpy())

plt.figure(figsize=(24,16))

for i in range(images.shape[0]):

    plt.subplot(images.shape[0]/4, 4, i+1)

    imshow(images[i])
model = EfficientNet.from_pretrained("efficientnet-b0")

num_fc = model._fc.in_features

model._fc = torch.nn.Linear(in_features=num_fc, out_features=len(class_names))
def training_loop(model, device, dataloaders, criterion, optimizer, scheduler, epochs=25):

    since = time.time()

    model = model.to(device)

    

    best_model_state = copy.deepcopy(model.state_dict())

    best_acc = 0.

    

    for epoch in range(1, epochs+1):

        print("Epoch: {}/{}".format(epoch, epochs))

        for phase in ['train','val']:

            if phase == 'train':

                model.train()

            else:

                model.eval()

                

            running_loss = 0.

            running_correct = 0

    

            start = time.time()

                

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, axis=1)

                    loss = criterion(outputs, labels)

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                running_correct += torch.sum(preds == labels.data)

            if phase == 'train':

                scheduler.step()



            end = time.time()

            data_size = len(dataloaders[phase].dataset)

            epoch_loss = running_loss / data_size

            epoch_acc = running_correct.item() / data_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase.capitalize(), epoch_loss, epoch_acc))

            for param_group in optimizer.param_groups:

                print("learning rate:{}".format(param_group['lr']))

            time_elapsed = end - start

            print('{} time elapsed: {:.0f}m {:.0f}s {:.0f}ms, {:.4f}s/image'.format(

                phase.capitalize(), time_elapsed // 60, time_elapsed % 60, time_elapsed * 1000 % 1000, time_elapsed / data_size))

        print('-'*20)



        if phase == 'val' and epoch_acc > best_acc:

            best_acc = epoch_acc

            best_model_state = copy.deepcopy(model.state_dict())

        if phase == 'val' and epoch % 10 == 0:

            torch.save(model.state_dict(), 'model{}-{:.3f}.pth'.format(epoch, epoch_acc))

        print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print("Best val Acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_state)

    return model    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = optim.Adam(model.parameters(), lr=1e-3)

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

criterion = torch.nn.CrossEntropyLoss()
model = training_loop(model, device, dataloader, criterion, optimizer, exp_lr_scheduler, epochs=100)
def eval_loop(model, device, class_names, input_dir):

    count = 0

    for filename in os.listdir(input_dir):

        p = Path(input_dir)

        filepath = p / filename

        img = Image.open(filepath)

        img = transforms.functional.resize(img, 256)

        img = transforms.functional.center_crop(img, 224)

        img = np.array(img)

        if img.ndim < 3:

            continue

        img = img.transpose((2,0,1))

        img_tensor = torch.from_numpy(img).float() / 255.0

        img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(device)

        with torch.no_grad():

            outputs = model(img_tensor)

            _, preds = torch.max(outputs, axis=1)

            outputs = torch.nn.functional.softmax(outputs, 1)

            count += 1

            print(count, preds.item(), class_names[preds.item()], outputs.cpu().numpy()[0][preds.item()], filename.split('/')[-1])
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

model.eval()

eval_loop(model, device, class_names, input_dir='/kaggle/input/hymenoptera-data/hymenoptera_data/train/bees/')
!pwd

!ls -lh