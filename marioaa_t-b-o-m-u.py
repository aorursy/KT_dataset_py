# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

   # for filename in filenames:

    #    print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from fastai.vision import*
root = Path('../input/123aa44/vahadane')
os.listdir(root)
np.random.seed(42)

data = ImageDataBunch.from_folder(root,train='train',valid='valid',ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,8))
learn = cnn_learner(data, models.resnet34, metrics=accuracy,model_dir= "/tmp/model/")
learn.fit_one_cycle(4)

learn.save('train1')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(5, max_lr=slice(1e-05,1e-02))

learn.validate()
learn.save('train2')
tfms = get_transforms()

data_test = ImageDataBunch.from_folder(root, train='train', valid='test', bs=64, ds_tfms=tfms, size=224).normalize(imagenet_stats)
data_test = (ImageList.from_folder(root)

        .split_by_folder(train='train', valid='test')

        .label_from_folder()

        .transform(tfms)

        .databunch()

        .normalize()

       ) 
learn = cnn_learner(data_test, models.resnet34, metrics=accuracy,model_dir= "/tmp/model/")

learn.load('train2')
learn.fit_one_cycle(5)

learn.validate(data_test.valid_dl)

import matplotlib.pyplot as plt

import numpy as np



import torch

import torch.nn as nn

import torch.optim as optim

import torchvision

import torchvision.models as models

import torchvision.transforms as transforms



import time

import os

import PIL.Image as Image

from IPython.display import display



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

print(torch.cuda.get_device_name(device))
dataset_dir = "../input/123aa22/vahadane/"



train_tfms = transforms.Compose([transforms.Resize((224, 224)),

                                 transforms.RandomHorizontalFlip(),

                                 transforms.RandomRotation(15),

                                 transforms.ToTensor(),

                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_tfms = transforms.Compose([transforms.Resize((224, 224)),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



dataset = torchvision.datasets.ImageFolder(root=dataset_dir+"train", transform = train_tfms)

trainloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=True, num_workers = 2)



dataset2 = torchvision.datasets.ImageFolder(root=dataset_dir+"val", transform = test_tfms)

testloader = torch.utils.data.DataLoader(dataset2, batch_size = 32, shuffle=False, num_workers = 2)
def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated





# Get a batch of training data

inputs, classes = next(iter(dataloaders['train']))



# Make a grid from batch

out = torchvision.utils.make_grid(inputs)



imshow(out, title=[class_names[x] for x in classes])
def train_model(model, criterion, optimizer, scheduler, n_epochs = 5):

    

    losses = []

    accuracies = []

    test_accuracies = []

    # set the model to train mode initially

    model.train()

    for epoch in range(n_epochs):

        since = time.time()

        running_loss = 0.0

        running_correct = 0.0

        for i, data in enumerate(trainloader, 0):



            # get the inputs and assign them to cuda

            inputs, labels = data

            #inputs = inputs.to(device).half() # uncomment for half precision model

            inputs = inputs.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            

            # forward + backward + optimize

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            

            # calculate the loss/acc later

            running_loss += loss.item()

            running_correct += (labels==predicted).sum().item()



        epoch_duration = time.time()-since

        epoch_loss = running_loss/len(trainloader)

        epoch_acc = 100/32*running_correct/len(trainloader)

        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))

        

        losses.append(epoch_loss)

        accuracies.append(epoch_acc)

        

        # switch the model to eval mode to evaluate on test data

        model.eval()

        test_acc = eval_model(model)

        test_accuracies.append(test_acc)

        

        # re-set the model to train mode after validating

        model.train()

        scheduler.step(test_acc)

        since = time.time()

    print('Finished Training')

    return model, losses, accuracies, test_accuracies

def eval_model(model):

    correct = 0.0

    total = 0.0

    with torch.no_grad():

        for i, data in enumerate(testloader, 0):

            images, labels = data

            #images = images.to(device).half() # uncomment for half precision model

            images = images.to(device)

            labels = labels.to(device)

            

            outputs = model_ft(images)

            _, predicted = torch.max(outputs.data, 1)

            

            total += labels.size(0)

            correct += (predicted == labels).sum().item()



    test_acc = 100.0 * correct / total

    print('Accuracy of the network on the test images: %d %%' % (

        test_acc))

    return test_acc
model_ft = models.resnet34(pretrained=True)

num_ftrs = model_ft.fc.in_features



# replace the last fc layer with an untrained one (requires grad by default)

model_ft.fc = nn.Linear(num_ftrs, 196)

model_ft = model_ft.to(device)



# uncomment this block for half precision model

"""

model_ft = model_ft.half()





for layer in model_ft.modules():

    if isinstance(layer, nn.BatchNorm2d):

        layer.float()

"""

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)



"""

probably not the best metric to track, but we are tracking the training accuracy and measuring whether

it increases by atleast 0.9 per epoch and if it hasn't increased by 0.9 reduce the lr by 0.1x.

However in this model it did not benefit me.

"""

lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)
model_ft, training_losses, training_accs, test_accs = train_model(model_ft, criterion, optimizer, lrscheduler, n_epochs=10)
