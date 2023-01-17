!nvidia-smi
import os

import time

import numpy as np

import pandas as pd

import torch

from matplotlib import pyplot as plt

from PIL import Image

from torch.optim import Adam, lr_scheduler

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

from torch.nn import Linear, CrossEntropyLoss

from torchvision.models import densenet201, resnet152, vgg19_bn 

from torchvision.transforms import Compose, RandomApply, RandomAffine, ColorJitter, Normalize, ToTensor
folders = {

    "plots": "plots",

    "models": "models",

    "results": "results"

}

for key in folders.keys():

    try:

        os.makedirs(folders[key])

    except FileExistsError:

        # if file exists, pass

        pass
class PCam(Dataset):

    """Patch Camelyon dataset."""



    def __init__(self, csv_file, root_dir, train=True, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with labels.

            root_dir (string): Root directory.

            train (boolean): Whether loading training or testing data. 

                            This is required to have same number of examples in each 

                            classification to be able to train better.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        if train:

            dataframe = pd.read_csv(os.path.join(root_dir, csv_file))

            min_value = dataframe['label'].value_counts().min()

            frames = []

            for label in dataframe['label'].unique():

                frames.append(dataframe[dataframe['label'] == label].sample(min_value))

            self.labels = pd.DataFrame().append(frames).sample(frac=1).reset_index(drop=True)

            self.data_folder = "train"

        else:

            self.labels = pd.read_csv(os.path.join(root_dir, csv_file))

            self.data_folder = "test"

        

        self.root_dir = root_dir

        self.transform = transform



    def __len__(self):

        return len(self.labels)



    def __getitem__(self, idx):

        image_name = os.path.join(self.root_dir,

                                "%s/%s.tif" % (self.data_folder, self.labels.iloc[idx, 0]))

        image = Image.open(image_name)

        image.thumbnail((40, 40), Image.ANTIALIAS)

        if self.transform is not None:

            image = self.transform(image)



        return self.labels.iloc[idx, 0], image, self.labels.iloc[idx, 1]
BATCH_SIZE = 32  # mini_batch size

MAX_EPOCH = 8  # maximum epoch to train

STEP_SIZE = 2  # decrease in learning rate after epochs

GAMMA = 0.1  # used in decreasing the gamma
train_transform = Compose([

    RandomAffine(45, translate=(0.15,0.15), shear=45),

    RandomApply([ColorJitter(saturation=0.5, hue=0.5)]),

    ToTensor(),

    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

test_transform = Compose(

    [ToTensor(),

     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # torchvision.transforms.Normalize(mean, std)



trainset = PCam(csv_file='train_labels.csv', root_dir='../input', train=True, transform=train_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)



testset = PCam(csv_file='sample_submission.csv', root_dir='../input', train=False, transform=test_transform)

testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
def eval_net(net, criterion, dataloader):

    correct = 0

    total = 0

    total_loss = 0

    net.eval()

    

    for data in dataloader:

        _, images, labels = data

        images, labels = Variable(images).cuda(), Variable(labels).cuda()

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels.data).sum().item()

        loss = criterion(outputs, labels)

        total_loss += loss.item()

    return total_loss / total, correct / total
def train_net(net, criterion, eval_criterion, optimizer, scheduler):



    train_loss_array = []

    test_loss_array = []

    train_accuracy_array = []

    test_accuracy_array = []



    print('Start training...')

    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        scheduler.step()

        net.train()

        running_loss = 0.0

        for i, data in enumerate(trainloader):

            _, inputs, labels = data

            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()



            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 500 == 499:    # print every 2000 mini-batches

                print('Step: %5d avg_batch_loss: %.5f' % (i + 1, running_loss / 500))

                running_loss = 0.0

        print('Finish training this EPOCH, start evaluating...')

        train_loss, train_acc = eval_net(net, eval_criterion, trainloader)

        test_loss, test_acc = eval_net(net, eval_criterion, testloader)

        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %

              (epoch+1, train_loss, train_acc, test_loss, test_acc))



        train_loss_array.append(train_loss)

        test_loss_array.append(test_loss)



        train_accuracy_array.append(train_acc)

        test_accuracy_array.append(test_acc)

    print('Finished Training')



    # plot loss

    plt.clf()

    plt.plot(list(range(1, MAX_EPOCH + 1)), train_loss_array, label='Train')

    plt.plot(list(range(1, MAX_EPOCH + 1)), test_loss_array, label='Test')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.title('Loss vs Epochs [%s]' % net.name)

    plt.savefig('./%s/loss-%s.png' % (folders['plots'], net.name))



    # plot accuracy

    plt.clf()

    plt.plot(list(range(1, MAX_EPOCH + 1)), train_accuracy_array, label='Train')

    plt.plot(list(range(1, MAX_EPOCH + 1)), test_accuracy_array, label='Test')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.title('Accuracy vs Epochs [%s]' % net.name)

    plt.savefig('./%s/accuracy-%s.png' % (folders['plots'], net.name))
def dump_results(dataloader, net):

    net.eval()

    results = pd.DataFrame()

    for data in dataloader:

        image_names, images, labels = data

        images, labels = Variable(images).cuda(), Variable(labels).cuda()

        outputs = net(images)

        _, predictions = torch.max(outputs.data, 1)

        results = results.append(pd.DataFrame({"id": image_names, "label": predictions.cpu().numpy()}))

    results.to_csv("%s/%s.csv" % (folders['results'], net.name), index=False)
start = time.time()

cur_net = densenet201(pretrained=True)

num_ftrs = cur_net.classifier.in_features

cur_net.classifier = Linear(num_ftrs, 2)

cur_net.name = "DenseNet201"

cur_net = cur_net.cuda()



cur_criterion = CrossEntropyLoss()

val_criterion = CrossEntropyLoss(reduction='sum')

cur_optimizer = Adam(cur_net.parameters(), lr=0.00007)

exp_lr_scheduler = lr_scheduler.StepLR(cur_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

train_net(cur_net, cur_criterion, val_criterion, cur_optimizer, exp_lr_scheduler)

dump_results(testloader, cur_net)

print("Time taken: %d secs" % int(time.time() - start))
start = time.time()

cur_net = resnet152(pretrained=True)

num_ftrs = cur_net.fc.in_features

cur_net.fc = Linear(num_ftrs, 2)

cur_net.name = "ResNet152"

cur_net = cur_net.cuda()



cur_criterion = CrossEntropyLoss()

val_criterion = CrossEntropyLoss(reduction='sum')

cur_optimizer = Adam(cur_net.parameters(), lr=0.00007)

exp_lr_scheduler = lr_scheduler.StepLR(cur_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

train_net(cur_net, cur_criterion, val_criterion, cur_optimizer, exp_lr_scheduler)

dump_results(testloader, cur_net)

print("Time taken: %d secs" % int(time.time() - start))
start = time.time()

cur_net = vgg19_bn(pretrained=True)

num_ftrs = cur_net.classifier._modules['6'].in_features

cur_net.classifier._modules['6'] = Linear(num_ftrs, 2)

cur_net.name = "VGG19"

cur_net = cur_net.cuda()



cur_criterion = CrossEntropyLoss()

val_criterion = CrossEntropyLoss(reduction='sum')

cur_optimizer = Adam(cur_net.parameters(), lr=0.00007)

exp_lr_scheduler = lr_scheduler.StepLR(cur_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

train_net(cur_net, cur_criterion, val_criterion, cur_optimizer, exp_lr_scheduler)

dump_results(testloader, cur_net)

print("Time taken: %d secs" % int(time.time() - start))