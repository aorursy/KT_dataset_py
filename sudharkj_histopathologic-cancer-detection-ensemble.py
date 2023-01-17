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

from torch.nn import Conv2d, Linear, CrossEntropyLoss

from torchvision.models import densenet201, resnet152, vgg19_bn 

from torchvision.transforms import Compose, RandomAffine, RandomApply, ColorJitter, Normalize, ToTensor
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

            # make the number of example in each classification equal

            dataframe = pd.read_csv(os.path.join(root_dir, csv_file))

            min_value = dataframe['label'].value_counts().min()

            frames = []

            for label in dataframe['label'].unique():

                frames.append(dataframe[dataframe['label'] == label].sample(min_value))

                # .sample(frac=1) shuffles the data

                # .reset_index(drop=True) do not add index while shuffling

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

        # reduce image size to be able to train fast

        image.thumbnail((40, 40), Image.ANTIALIAS)

        if self.transform is not None:

            image = self.transform(image)



        return self.labels.iloc[idx, 0], image, self.labels.iloc[idx, 1]
NUM_CLASSES = 2  # number of classes

BATCH_SIZE = 32  # mini_batch size

MAX_EPOCH = 10  # maximum epoch to train

STEP_SIZE = 2  # decrease in learning rate after epochs

LEARNING_RATE = 0.00007  # learning rate

GAMMA = 0.1  # used in decreasing the gamma
# other transformations are not included because they are included in these or those are not required in real life

train_transform = Compose([

    RandomAffine(45, translate=(0.15, 0.15), shear=45),

    RandomApply([ColorJitter(saturation=0.5, hue=0.5)]),

    ToTensor(),

    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

test_transform = Compose(

    [ToTensor(),

     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset = PCam(csv_file='train_labels.csv', root_dir='../input', train=True, transform=train_transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)



testset = PCam(csv_file='sample_submission.csv', root_dir='../input', train=False, transform=test_transform)

testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
def eval_ensemble(nets, criterion, dataloader):

    correct = 0

    total = 0

    total_loss = 0



    for data in dataloader:

        _, images, labels = data

        #         images, labels = Variable(images), Variable(labels)

        images, labels = Variable(images).cuda(), Variable(labels).cuda()



        predictions = torch.zeros([images.size(0), NUM_CLASSES]).cuda()

        for net in nets:

            net.eval()

            outputs = net(images)

            predictions = predictions.add(outputs)



        # predictions = predictions / len(nets)  # redundant division

        _, predicted = torch.max(predictions.data, 1)

        

        total += labels.size(0)

        correct += (predicted == labels.data).sum().item()

        

        loss = criterion(predictions, labels)

        total_loss += loss.item()

    return total_loss / total, correct / total
def train_ensemble(nets, optimizers, schedulers, criterion, eval_criterion):

    ensemble_name = 'ensemble'

    train_loss_array = []

    test_loss_array = []

    train_accuracy_array = []

    test_accuracy_array = []



    print('Start training...')

    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        for scheduler in schedulers:

            scheduler.step()

        

        running_loss = 0.0

        for i, data in enumerate(trainloader):

            _, images, labels = data

            #             inputs, labels = Variable(inputs), Variable(labels)

            images, labels = Variable(images).cuda(), Variable(labels).cuda()



            predictions = torch.zeros([images.size(0), NUM_CLASSES]).cuda()

            for net, optimizer in zip(nets, optimizers):

                net.train()

                optimizer.zero_grad()

                outputs = net(images)

                predictions = predictions.add(outputs)



            # predictions = predictions / len(nets)  # redundant division

            

            # back prop

            loss = criterion(predictions, labels)

            loss.backward()

            for optimizer in optimizers:

                optimizer.step()

            running_loss += loss.item()

            

            if i % 500 == 499:  # print every 2000 mini-batches

                print('Step: %5d avg_batch_loss: %.5f' % (i + 1, running_loss / 500))

                running_loss = 0.0

                

        print('Finish training this EPOCH, start evaluating...')

        train_loss, train_acc = eval_ensemble(nets, eval_criterion, trainloader)

        test_loss, test_acc = eval_ensemble(nets, eval_criterion, testloader)

        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %

              (epoch + 1, train_loss, train_acc, test_loss, test_acc))



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

    plt.savefig('./%s/loss-%s.png' % (folders['plots'], ensemble_name))



    # plot accuracy

    plt.clf()

    plt.plot(list(range(1, MAX_EPOCH + 1)), train_accuracy_array, label='Train')

    plt.plot(list(range(1, MAX_EPOCH + 1)), test_accuracy_array, label='Test')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.title('Accuracy vs Epochs [%s]' % net.name)

    plt.savefig('./%s/accuracy-%s.png' % (folders['plots'], ensemble_name))
def dump_ensemble_results(dataloader, nets):

    ensemble_name = 'ensemble'

    

    alphas = []

    for net in nets:

        net.eval()

        

        true_positives = 0

        for data in trainloader:

            _, images, labels = data

            images, labels = Variable(images).cuda(), Variable(labels).cuda()

            outputs = net(images)

            _, outputs = torch.max(outputs.data, 1)

            index = (labels == 1)

            true_positives += (outputs[index] == labels[index].data).sum().item()

        alphas.append(true_positives)

#     total_true_positives = sum(alphas)

    

    results = pd.DataFrame()

    for data in dataloader:

        image_names, images, labels = data

#         images, labels = Variable(images), Variable(labels)

        images, labels = Variable(images).cuda(), Variable(labels).cuda()

    

        predictions = torch.zeros([images.size(0), NUM_CLASSES]).cuda()

        for index, net in enumerate(nets):

            net.eval()

            outputs = net(images) * alphas[index]  # / total_true_positives

            predictions = predictions.add(outputs)



#         predictions = predictions / total_true_positives

        _, predictions = torch.max(predictions.data, 1)

        results = results.append(pd.DataFrame({"id": image_names, "label": predictions.cpu().numpy()}))

    results.to_csv("%s/%s.csv" % (folders['results'], ensemble_name), index=False)
start = time.time()

net_list = []

optimizer_list = []

scheduler_list = []



# DenseNet201

dense_net = densenet201()

dense_num_ftrs = dense_net.classifier.in_features

dense_net.classifier = Linear(dense_num_ftrs, NUM_CLASSES)

dense_net.name = "DenseNet201"

dense_net = dense_net.cuda()

dense_optimizer = Adam(dense_net.parameters(), lr=LEARNING_RATE)

dense_exp_lr_scheduler = lr_scheduler.StepLR(dense_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

net_list.append(dense_net)

optimizer_list.append(dense_optimizer)

scheduler_list.append(dense_exp_lr_scheduler)



# ResNet152

res_net = resnet152()

res_num_ftrs = res_net.fc.in_features

res_net.fc = Linear(res_num_ftrs, NUM_CLASSES)

res_net.name = "ResNet152"

res_net = res_net.cuda()

res_optimizer = Adam(res_net.parameters(), lr=LEARNING_RATE)

res_exp_lr_scheduler = lr_scheduler.StepLR(res_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

net_list.append(res_net)

optimizer_list.append(res_optimizer)

scheduler_list.append(res_exp_lr_scheduler)



# VGG19

vgg_net = vgg19_bn()

vgg_num_ftrs = vgg_net.classifier._modules['6'].in_features

vgg_net.classifier._modules['6'] = Linear(vgg_num_ftrs, NUM_CLASSES)

vgg_net.name = "VGG11"

vgg_net = vgg_net.cuda()

vgg_optimizer = Adam(vgg_net.parameters(), lr=LEARNING_RATE)

vgg_exp_lr_scheduler = lr_scheduler.StepLR(vgg_optimizer, step_size=STEP_SIZE, gamma=GAMMA)

net_list.append(vgg_net)

optimizer_list.append(vgg_optimizer)

scheduler_list.append(vgg_exp_lr_scheduler)



train_criterion = CrossEntropyLoss()

val_criterion = CrossEntropyLoss(reduction='sum')

train_ensemble(net_list, optimizer_list, scheduler_list, train_criterion, val_criterion)

print("Time taken: %d secs" % int(time.time() - start))
dump_ensemble_results(testloader, net_list)