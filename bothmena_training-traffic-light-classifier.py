import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

from skimage import io, transform



import matplotlib.pyplot as plt

import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils



import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import torchvision

from torchvision import datasets, models, transforms

import time

import copy
!unzip ../input/traffic-light-boxes-dataset-extraction/output.zip > /var/null
!ls dataset/images/ -l | wc -l
class TrafficLightDataset(Dataset):

    def __init__(self, root_dir: str, transform=None):

        self.root_dir = root_dir

        self.transform = transform

        self.df = pd.read_csv(os.path.join(self.root_dir, 'annotations.csv'))



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        bbox_path = os.path.join(self.root_dir, 'images', self.df.loc[idx, 'bbox_filename'])

        bbox_img = cv2.imread(bbox_path)

        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)

        

        # classes: 0: go / green | 1: warning / orange | 2: stop / red

        annotation = self.df.loc[idx, 'annotation_tag']

        if annotation[:2] == 'go':

            y = np.array([1, 0, 0], dtype=np.float32)

        elif annotation[:7] == 'warning':

            y = np.array([0, 1, 0], dtype=np.float32)

        elif annotation[:4] == 'stop':

            y = np.array([0, 0, 1], dtype=np.float32)

            

        sample = {'image': bbox_img, 'target': y}  # .reshape((-1, 1))

            

        if self.transform:

            sample = self.transform(sample)

        

        return sample

dataset = TrafficLightDataset('dataset')
len(dataset)
def show_random_image():

    idx = np.random.randint(0, len(dataset))

    sample = dataset[idx]

    

    image = sample['image']

    target = sample['target']

    if isinstance(image, torch.Tensor):

        image = image.numpy()

        image = image.transpose((1, 2, 0))

        target = target.numpy()

    

    plt.imshow(image)

    img_cls = ['Green', 'Orange', 'Red'][target.argmax()]

    plt.title(img_cls)

    plt.show()

show_random_image()
stats = [0, 0, 0]

min_shape = 5000

for idx in range(len(dataset)):

    shape = dataset[idx]['image'].shape[0]

    if shape < min_shape:

        min_shape = shape

    stats[dataset[idx]['target'].argmax()] += 1



print('stats:', stats)

print('min shape:', min_shape)

class Rescale(object):

    """

    Rescale the image in a sample to a given size.

    """



    def __init__(self, output_size: int):

        self.output_size = output_size



    def __call__(self, sample):

        image, target = sample['image'], sample['target']

        image = transform.resize(image, (self.output_size, self.output_size))



        return {'image': image, 'target': target}

    

class ToTensor(object):

    """

    Convert ndarrays in sample to Tensors.

    """



    def __call__(self, sample):

        image, target = sample['image'], sample['target']



        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),

                'target': torch.from_numpy(target)}

trans = transforms.Compose([Rescale(28), ToTensor()])

dataset = TrafficLightDataset('dataset', transform=trans)
show_random_image()
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [4000, 1000, 1714])  #  [0.6, 0.15, 0.25]
dataset[0]['target'].argmax(), dataset[1]['target'].argmax(), dataset[2]['target'].argmax(), dataset[3]['target'].argmax(), dataset[4]['target'].argmax()
train_dataset[0]['target'].argmax(), train_dataset[1]['target'].argmax(), train_dataset[2]['target'].argmax(), train_dataset[3]['target'].argmax(), train_dataset[4]['target'].argmax()
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=4)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        

        self.features = nn.Sequential(

            nn.Conv2d(3, 64, 3, 1, 1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),  # 28*28*3 => 28*28*64

            

            nn.Conv2d(64, 128, 3, 1, 1),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 28*28*64 => 14*14*128

            

            nn.Conv2d(128, 256, 3, 1, 1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 14*14*128 => 7*7*256

        )

        output_ft = 256 * 7 * 7

        self.classifier = nn.Sequential(

            nn.Linear(output_ft, 1024),

            nn.ReLU(),

            nn.Linear(1024, 3),

        )

    

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)



        return x

def weights_init(m):

    if isinstance(m, nn.Conv2d):

        torch.nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:

            m.bias.data.fill_(0.01)

    if isinstance(m, nn.Linear):

        torch.nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:

            m.bias.data.fill_(0.01)

def lr_decay(iteration, init_lr=0.001, decay=0.5, cycle=None):

    if isinstance(cycle, int):

        iteration = iteration % cycle

    return init_lr / (1 + decay * iteration)

def save_net(name: str):

    if not os.path.isdir('models'):

        os.mkdir('models', 0o777)

    torch.save(net.state_dict(), 'models/' + name + '.pt')
experiences = {}
criterion = nn.BCEWithLogitsLoss()
def measure_accuracy(predictions, ground_truth):

    return (predictions.argmax(1) == ground_truth.argmax(1)).sum().item() / predictions.shape[0]

def train_net(net, n_epochs, print_every=10, lr=0.001, lrd=False, lrd_cycle=None):



    # prepare the net for training

    net.train()

    if not lrd:

        optimizer = optim.Adam(net.parameters(), lr=lr)

    losses = []

    accuracies = []

    

    val_losses = []

    val_accuracies = []



    for epoch in range(n_epochs):  # loop over the dataset multiple times

        

        net.train()

        if lrd:

            new_lr = lr_decay(epoch, lr, cycle=lrd_cycle)

            optimizer = optim.Adam(net.parameters(), lr=new_lr)

        

        running_loss = 0.0

        running_acc = 0.0

        

        # train on batches of data, assumes you already have train_loader

        for batch_i, data in enumerate(train_dataloader):

            # get the input images and their corresponding labels

            images = data['image']

            labels = data['target']



            # convert variables to floats for regression loss

            if torch.cuda.is_available():

                labels = labels.cuda().float()

                images = images.cuda().float()

            else:

                labels = labels.type(torch.FloatTensor)

                images = images.type(torch.FloatTensor)



            # forward pass to get outputs

            output = net(images)

            

            # calculate the loss between predicted and target keypoints

            loss = criterion(output, labels)



            # zero the parameter (weight) gradients

            optimizer.zero_grad()

            

            # backward pass to calculate the weight gradients

            loss.backward()



            # update the weights

            optimizer.step()

            

            # print loss statistics

            running_loss += loss.item()

            running_acc += measure_accuracy(output, labels)

            if (batch_i + 1) % print_every == 0:

                print('Epoch: {}, Batch: {}, Avg. Loss: {}, accuracy = {}'.format(epoch + 1, batch_i+1, running_loss/print_every, running_acc/print_every))

                losses.append(running_loss/print_every)

                running_loss = 0.0

                running_acc = 0.0



        net.eval()

        running_val_loss = 0.

        running_val_acc = 0.

        for batch_j, eval_data in enumerate(validation_dataloader):

            images = eval_data['image']

            labels = eval_data['target']



            # convert variables to floats for regression loss

            if torch.cuda.is_available():

                labels = labels.cuda().float()

                images = images.cuda().float()

            else:

                labels = labels.type(torch.FloatTensor)

                images = images.type(torch.FloatTensor)



            # forward pass to get outputs

            output = net(images)



            running_val_loss += criterion(output, labels)

            running_val_acc += measure_accuracy(output, labels)



        val_loss = running_val_loss / len(validation_dataloader)

        val_losses.append(val_loss)

        val_acc = running_val_acc / len(validation_dataloader)

        val_accuracies.append(val_acc)



        print('Epoch: {}, Avg. validation Loss: {}, validation accuracy = {}'.format(epoch + 1, val_loss, val_acc))

        print('-' * 85)



    print('Finished Training')

    return losses, accuracies, val_losses, val_accuracies

torch.cuda.empty_cache()
net = Network()

if torch.cuda.is_available():

    net.cuda()

net.apply(weights_init)

print(net)

n_epochs = 30 # start small, and increase when you've decided on your model structure and hyperparams



losses, accuracies, val_losses, val_acc = train_net(net, n_epochs, print_every=20, lr=0.0001 , lrd=True, lrd_cycle=40)

name = 'net_v4_cyclic_lrd_200'

experiences[name] = losses

# save_net(name)

!rm -rf dataset/