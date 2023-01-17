from PIL import Image

import random

from matplotlib import pyplot as plt

from os import listdir, path



from torch import nn

import torch.nn.functional as F



from torch.utils.data import Dataset, DataLoader

from glob import glob

import torch

import numpy as np



from torchvision.transforms.transforms import Compose, Resize, ToTensor, CenterCrop

import time
# all images will be resized to this number

IMG_SIZE = 49
def show_sample_images(root_dir='data', sample_size=10, from_class=None):

    

    '''

    This function samples images from input data and shows them in a plot.

    Samples are with replacement out of the given class. If no class is given,

    samples are extracted from random classes.

    '''

    

    # number of images on the x and y axes should be relatively close

    for i in range(int(sample_size**0.5 + 1), 0, -1):

        if sample_size % i == 0:

            N_images_x = i

            N_images_y = int(sample_size/N_images_x)

            break

    

    # sample data directory and show image in subplots

    fig, ax = plt.subplots(N_images_x, N_images_y, squeeze=False)

    for sample_number in range(sample_size):

        if not from_class:

            labels_list = listdir(path.join(root_dir, 'nonsegmentedv2'))

            label = str(random.sample(labels_list, 1)[0])

        

        parent_directory = path.join(root_dir, 'nonsegmentedv2', label)

        file_list = listdir(parent_directory)

        file_name = random.sample(file_list, 1)[0]

        img = Image.open(path.join(parent_directory, file_name))

        ax[sample_number % N_images_x][sample_number // N_images_x].imshow(img)

        ax[sample_number % N_images_x][sample_number // N_images_x].title.set_text(label)

        

    fig.set_figheight(N_images_y * 4)

    fig.set_figwidth(N_images_y * 4)

    fig.show()
# show a random sample of 20 images

show_sample_images('../input', 20)
# define a very small model (~400k parameters) that can work, improvements may be made later

class TinyModel(nn.Module):

    

    def __init__(self):

        super(TinyModel, self).__init__()

        self.bn1 = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5, 5))

        self.bn2 = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(in_channels=10 , out_channels=10, kernel_size=(5, 5), stride=2)

        self.fc1 = nn.Linear(4410, 100)

        self.fc2 = nn.Linear(100, 12)

        

    def forward(self, x):

        x = F.relu(self.conv1(self.bn1(x)))

        x = F.relu(self.conv2(self.bn2(x)))

        x = x.view(-1, 4410)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        

        return F.log_softmax(x, dim=1)

    

    def loss(self, prediction, true_values):



        return F.nll_loss(prediction, true_values)
# define a class for the dataset

class SeedlingDataset(Dataset):

    

    def __init__(self, data, n_labels, transform=None):

        

        self.data = data

        self.transform = transform

        self.n_labels = n_labels

        self.transform = transform

        

    def __len__(self):

        

        return len(self.data)

    

    def __getitem__(self, idx):

        

        data_file, image_code = self.data[idx]

        img = Image.open(data_file)

        img = img.convert('RGB')

        if self.transform:

            img = self.transform(img)

        img = torch.from_numpy(np.array(img))

        label = torch.tensor(image_code)

        

        return img, label

        

    

def getDataLoaders(root_dir, transforms, batch_size, test_ratio=0.2):

    

    '''

    This function defines train and test data loaders for the Seedling class of data.

    test_ratio is the ratio of the number of test images to the total number of images.

    '''

    

    all_labels = listdir(path.join(root_dir, 'nonsegmentedv2'))

    all_data = []

    

    # Create a list of tuples out of data samples. Each tuple includes images file name and an int as label code

    for label_code, parent_dir in enumerate(all_labels):

        this_dir_images = glob(path.join(root_dir, 'nonsegmentedv2', parent_dir, '*.png'))

        all_data += zip(this_dir_images, [label_code]*len(this_dir_images))

    

    # Shuffle data and create train and test parts

    random.shuffle(all_data)

    n_train_examples = int(len(all_data) * (1 - test_ratio))

    train_data = SeedlingDataset(all_data[:n_train_examples], len(all_labels), transform=transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    test_data = SeedlingDataset(all_data[n_train_examples:], len(all_labels), transform=transforms)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    

    return train_loader, test_loader
def train(model, device, train_dataloader, optimizer, epoch, verbose=False):

    

    model.train()

    total_loss = 0

    for batch_idx, (image, label) in enumerate(train_dataloader):

        input_var = image.to(device)

        target_var = label.to(device)

        optimizer.zero_grad()

        output = model(input_var)

        loss = model.loss(output, target_var)

        loss.backward()

        optimizer.step()

        total_loss += loss

        if batch_idx % 10 == 0 and verbose:

            print('Train Epoch: {0}, Train batch: {1}, Batch loss: {2}'

                  .format(epoch, batch_idx, loss))

    

    epoch_loss = total_loss/(batch_idx + 1)

    

    return epoch_loss

    

    

def test(model, device, test_dataloader):

    

    model.eval()

    correct = 0

    with torch.no_grad():

        for image, label in test_dataloader:

            input_var = image.to(device)

            target_var = label.to(device)

            output = model(input_var)

            prediction = output.argmax(dim=1, keepdim=True)

            correct += prediction.eq(target_var.view_as(prediction)).sum().item()

    

    accuracy = correct / len(test_dataloader.dataset)

    

    return accuracy
def AdjustLR(optim, lr):

    '''

    This is a simple hand-made function to decrease 

    optimizer's learning rate by a factor of 10 at a specific time

    '''

    for param_group in optim.param_groups:

            param_group['lr'] /= 10
# main

losses = []

accuracies = []

n_epochs = 20

learning_rate = 0.1

lr_decay = 5 # every 10 epochs, the learning rate is divided by 10

batch_size = 64

use_cuda = False # use True to switch to GPU



device = torch.device("cuda" if use_cuda else "cpu")

model = TinyModel().to(device)

print('The model has {0} parameters'.format(sum([len(i.reshape(-1)) for i in model.parameters()]) ))



train_transform = Compose([Resize(IMG_SIZE), CenterCrop(IMG_SIZE), ToTensor()])

train_loader, test_loader = getDataLoaders('../input', train_transform, batch_size=batch_size, test_ratio=0.1)

print('number of train examples: {0}, number of test examles: {1}'

      .format(len(train_loader.dataset), len(test_loader.dataset)))



optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



for epoch in range(1, n_epochs + 1):

    tic = time.time()

    epoch_loss = train(model, device, train_loader, optimizer, epoch)

    print('Training loss for epoch {0} is {1:.5f}'.format(epoch, epoch_loss))

    losses.append(epoch_loss)

    accuracy = test(model, device, test_loader)

    print('Test accuracy: {0:.3f}'.format(accuracy))

    accuracies.append(accuracy)

    tac = time.time()

    print('Epoch time: {0:0.1f} seconds'.format(tac - tic))

    if epoch % lr_decay == 0:

        AdjustLR(optimizer, learning_rate)
fig, ax1 = plt.subplots(figsize=(8,4))



ax1.set_xlabel('Epochs')

ax1.set_ylabel('Training loss', color='r')

ax1.plot(range(1, n_epochs + 1), losses, color='r')

ax1.tick_params(axis='y', labelcolor='r')



ax2 = ax1.twinx()

ax2.set_ylabel('Test accuracy', color='b')

ax2.plot(range(1, n_epochs + 1), accuracies, color='b')

ax2.tick_params(axis='y', labelcolor='b')

fig.tight_layout()

plt.show()