%matplotlib inline

%reload_ext autoreload

%autoreload 2



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

from torchvision import transforms

from torch import nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

import torchvision.models as models

import matplotlib.pyplot as plt

import time

import math

import tqdm as tqdm

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    TRAIN_CSV = os.path.join(dirname, 'train.csv')

    TEST_CSV = os.path.join(dirname, 'test.csv')

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print(TRAIN_CSV)

print(TEST_CSV)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
train_df = pd.read_csv(TRAIN_CSV)

print(train_df.head())



test_df = pd.read_csv(TEST_CSV)

print(test_df.head())



train_labels = train_df['label']

print(train_labels)



train_df = train_df.drop(['label'], axis=1)

print(train_df.head())



print("Number of training example: {}".format(len(train_df)))

print("Number of testing example: {}".format(len(test_df)))
train_labels.value_counts()
plt.rcParams['figure.figsize'] = (8, 6)

plt.bar(train_labels.value_counts().index, train_labels.value_counts())

plt.xticks(np.arange(train_labels.nunique()))

plt.xlabel('Class')

plt.ylabel('Count')

plt.show()
transform = transforms.Compose([

    #transforms.ToPILImage(),

    #transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2)),

    transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))

])
class DS(Dataset):

    def __init__(self, imgs, labels=None, transform=None):

        self.imgs = imgs

        self.labels = labels

        self.transform = transform

        

    def __len__(self):

        return len(self.imgs)

    

    def __getitem__(self, index):

        image = self.imgs.iloc[index].values.astype(np.uint8).reshape((28, 28))

        if self.transform is not None:

            image = self.transform(image)

        if self.labels is not None:

            label = self.labels.iloc[index]

        else:

            return image       

        return image, label
features_train, features_test, targets_train, targets_test = train_test_split(train_df,train_labels, test_size = 0.1,

                                                                             random_state = 42)
trainset = DS(features_train, targets_train, transform)

validset = DS(features_test, targets_test, transform)

testset = DS(test_df, transform=transform)
bs = 64
train_loader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)

valid_loader = DataLoader(validset, batch_size=bs, shuffle=True, num_workers=4)

test_loader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=4)
def plot_img(imgs, labels, preds=None, is_pred=False, columns = 4):

    assert(len(imgs) > 0 and len(labels) > 0 and len(imgs) == len(labels))

    fig = plt.figure(figsize=(8,8))

    rows = math.ceil(len(imgs)/columns) 

    

    for i in range(1, columns*rows +1):

        fig.add_subplot(rows, columns, i)

        if is_pred:

            plt.title("Pred:: " + str(labels[i-1].item()) + "/Act:: " + str(train_labels[i-1].item()))

        else:

            plt.title("Pred:: " + str(labels[i-1].item()))

        plt.axis('off')

        plt.imshow(imgs[i-1].numpy().reshape(28,28), cmap='gray')

    plt.show()
imgs, labels = next(iter(train_loader))

plot_img(imgs[:20], labels[:20])
def accuracy(output, target, is_test=False):

    global total

    global correct

    batch_size = target.size(0)

    total += batch_size

    

    _, pred = torch.max(output, 1)

    if is_test:

        preds.extend(pred)

    correct += (pred == target).sum()

    return 100 * float(correct) / float(total)
class AvgStats(object):

    def __init__(self):

        self.reset()

        

    def reset(self):

        self.losses =[]

        self.precs =[]

        self.its = []

        

    def append(self, loss, prec, it):

        self.losses.append(loss)

        self.precs.append(prec)

        self.its.append(it)
def save_checkpoint(model, is_best, filename='./checkpoint.pth.tar'):

    """Save checkpoint if a new best is achieved"""

    if is_best:

        torch.save(model.state_dict(), filename)  # save checkpoint

    else:

        print ("=> Validation Accuracy did not improve")
def load_checkpoint(model, filename = './checkpoint.pth.tar'):

    sd = torch.load(filename, map_location=lambda storage, loc: storage)

    names = set(model.state_dict().keys())

    for n in list(sd.keys()):

        if n not in names and n+'_raw' in names:

            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]

            del sd[n]

    model.load_state_dict(sd)
class CLR(object):

    def __init__(self, optim, bn, base_lr=1e-7, max_lr=100):

        self.base_lr = base_lr

        self.max_lr = max_lr

        self.optim = optim

        self.bn = bn - 1

        ratio = self.max_lr/self.base_lr

        self.mult = ratio ** (1/self.bn)

        self.best_loss = 1e9

        self.iteration = 0

        self.lrs = []

        self.losses = []

        

    def calc_lr(self, loss):

        self.iteration +=1

        if math.isnan(loss) or loss > 4 * self.best_loss:

            return -1

        if loss < self.best_loss and self.iteration > 1:

            self.best_loss = loss

            

        mult = self.mult ** self.iteration

        lr = self.base_lr * mult

        

        self.lrs.append(lr)

        self.losses.append(loss)

        

        return lr

        

    def plot(self, start=10, end=-5):

        plt.xlabel("Learning Rate")

        plt.ylabel("Losses")

        plt.plot(self.lrs[start:end], self.losses[start:end])

        plt.xscale('log')

        

        

    def plot_lr(self):

        plt.xlabel("Iterations")

        plt.ylabel("Learning Rate")

        plt.plot(self.lrs)

        plt.yscale('log')
def update_lr(optimizer, lr):

    for g in optimizer.param_groups:

        g['lr'] = lr
def update_mom(optimizer, mom):

    for g in optimizer.param_groups:

        g['momentum'] = mom
class AdaptiveConcatPool2d(nn.Module):

    def __init__(self, sz=1):

        super().__init__()

        self.adavgp = nn.AdaptiveAvgPool2d(sz)

        self.adamaxp = nn.AdaptiveMaxPool2d(sz)

        

    def forward(self, x):

        x = torch.cat([self.adavgp(x), self.adamaxp(x)], 1)

        x = x.view(x.size(0),-1)

        return x
class CustomClassifier(nn.Module):

    def __init__(self, in_features, intermed_bn= 512, out_features=10, dout=0.25):

        super().__init__()

        self.fc_bn0 = nn.BatchNorm1d(in_features)

        self.dropout0 = nn.Dropout(dout)

        self.fc0 = nn.Linear(in_features, intermed_bn, bias=True)

        self.fc_bn1 = nn.BatchNorm1d(intermed_bn, momentum=0.01)

        self.dropout1 = nn.Dropout(dout * 2)

        self.fc1 = nn.Linear(intermed_bn, out_features, bias=True)

        

    def forward(self, x):

        x = self.fc_bn0(x)

        x = self.dropout0(x)

        x = F.relu(self.fc0(x))

        x = self.fc_bn1(x)

        x = self.dropout1(x)

        x = self.fc1(x)

        return x
class OneCycle(object):

    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt= 10, div=10, use_cosine=False):

        self.nb = nb

        self.div = div

        self.high_lr = max_lr

        self.low_mom = momentum_vals[1]

        self.high_mom = momentum_vals[0]

        self.use_cosine = use_cosine

        if self.use_cosine:

            self.prcnt = 0

        else:

            self.prcnt = prcnt

        self.iteration = 0

        self.lrs = []

        self.moms = []

        if self.use_cosine:

            self.step_len =  int(self.nb / 4)

        else:

            self.step_len =  int(self.nb * (1- prcnt/100)/2)

        

    def calc(self):

        if self.use_cosine:

            lr = self.calc_lr_cosine()

            mom = self.calc_mom_cosine()

        else:

            lr = self.calc_lr()

            mom = self.calc_mom()

        self.iteration += 1

        return (lr, mom)

        

    def calc_lr(self):

        if self.iteration ==  0:

            self.lrs.append(self.high_lr/self.div)

            return self.high_lr/self.div

        elif self.iteration == self.nb:

            self.iteration = 0

            self.lrs.append(self.high_lr/self.div)

            return self.high_lr/self.div

        elif self.iteration > 2 * self.step_len:

            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)

            #lr = self.high_lr * ( 1 - 0.99 * ratio)/self.div

            lr = (self.high_lr / self.div) * (1- ratio * (1 - 1/self.div))

        elif self.iteration > self.step_len:

            ratio = 1- (self.iteration -self.step_len)/self.step_len

            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div

        else :

            ratio = self.iteration/self.step_len

            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div

        self.lrs.append(lr)

        return lr



    def calc_mom(self):

        if self.iteration == 0:

            self.moms.append(self.high_mom)

            return self.high_mom

        elif self.iteration == self.nb:

            self.iteration = 0

            self.moms.append(self.high_mom)

            return self.high_mom

        elif self.iteration > 2 * self.step_len:

            mom = self.high_mom

        elif self.iteration > self.step_len:

            ratio = (self.iteration -self.step_len)/self.step_len

            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)

        else :

            ratio = self.iteration/self.step_len

            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)

        self.moms.append(mom)

        return mom



    def calc_lr_cosine(self):

        if self.iteration ==  0:

            self.lrs.append(self.high_lr/self.div)

            return self.high_lr/self.div

        elif self.iteration == self.nb:

            self.iteration = 0

            self.lrs.append(self.high_lr/self.div)

            return self.high_lr/self.div

        elif self.iteration > self.step_len:

            ratio = (self.iteration -self.step_len)/(self.nb - self.step_len)

            lr = (self.high_lr/self.div) + 0.5 * (self.high_lr - self.high_lr/self.div) * (1 + math.cos(math.pi * ratio))

        else :

            ratio = self.iteration/self.step_len

            lr = self.high_lr - 0.5 * (self.high_lr - self.high_lr/self.div) * (1 + math.cos(math.pi * ratio))

        self.lrs.append(lr)

        return lr



    def calc_mom_cosine(self):

        if self.iteration == 0:

            self.moms.append(self.high_mom)

            return self.high_mom

        elif self.iteration == self.nb:

            self.iteration = 0

            self.moms.append(self.high_mom)

            return self.high_mom

        elif self.iteration > self.step_len:

            ratio = (self.iteration -self.step_len)/(self.nb - self.step_len)

            mom = self.high_mom - 0.5 * (self.high_mom - self.low_mom) * (1 + math.cos(math.pi * ratio))

        else :

            ratio = self.iteration/self.step_len

            mom = self.low_mom + 0.5 * (self.high_mom - self.low_mom) * (1 + math.cos(math.pi * ratio))

        self.moms.append(mom)

        return mom
train_stats = AvgStats()

test_stats = AvgStats()
total = 0

correct = 0
train_loss = 0

test_loss = 0

best_acc = 0

trn_losses = []

trn_accs = []

val_losses = []

val_accs = []

preds = []
def train(model, epoch=0, use_cycle = False):

    model.train()

    global best_acc

    global trn_accs, trn_losses

    is_improving = True

    counter = 0

    running_loss = 0.

    avg_beta = 0.98

            

    for i, (input, target) in enumerate(train_loader):

        bt_start = time.time()

        input, target = input.to(device), target.to(device)

        var_ip, var_tg = Variable(input), Variable(target)

        

        if use_cycle:    

            lr, mom = onecycle.calc()

            update_lr(optimizer, lr)

            update_mom(optimizer, mom)

        

        output = model(var_ip)

        loss = criterion(output, var_tg)

            

        running_loss = avg_beta * running_loss + (1-avg_beta) *loss.item()

        smoothed_loss = running_loss / (1 - avg_beta**(i+1))

        

        trn_losses.append(smoothed_loss)

            

        # measure accuracy and record loss

        prec = accuracy(output.data, target)

        trn_accs.append(prec)



        train_stats.append(smoothed_loss, prec, time.time()-bt_start)

        if prec > best_acc :

            best_acc = prec

            save_checkpoint(model, True)



        # compute gradient and do SGD step

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()           
def test(model):

    with torch.no_grad():

        model.eval()

        global val_accs, val_losses

        running_loss = 0.

        avg_beta = 0.98

        for i, (input, target) in enumerate(valid_loader):

            bt_start = time.time()

            input, target = input.to(device), target.to(device)

            var_ip, var_tg = Variable(input), Variable(target)

            output = model(var_ip)

            loss = criterion(output, var_tg)



            running_loss = avg_beta * running_loss + (1-avg_beta) *loss.item()

            smoothed_loss = running_loss / (1 - avg_beta**(i+1))



            # measure accuracy and record loss

            prec = accuracy(output.data, target, is_test=True)

            test_stats.append(loss.item(), prec, time.time()-bt_start)



            val_losses.append(smoothed_loss)

            val_accs.append(prec)

            

        return running_loss
def fit(model, scheduler= None, use_cycle = False):

    print("Epoch\tTrain loss\tValidn loss\tTrain acc\tValidn acc")

    for j in range(epoch):

        train(model, j, use_cycle=use_cycle)

        loss = test(model)

        if scheduler is not None:

            scheduler.step(loss)

        print("{}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}"

              .format(j+1, trn_losses[-1], val_losses[-1], trn_accs[-1], val_accs[-1]))
model = models.resnet18(pretrained=True)

model
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model.avgpool = AdaptiveConcatPool2d()

model.fc = CustomClassifier(in_features=model.fc.in_features*2, out_features=10)
for param in model.parameters():

    param.require_grad = False

    

for param in model.fc.parameters():

    param.require_grad = True

    

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
clr = CLR(optimizer, len(train_loader))
save_checkpoint(model, True, 'before_clr.pth.tar')
def lr_find(model):

    t = tqdm.tqdm(train_loader, leave=False, total=len(train_loader))

    running_loss = 0.

    avg_beta = 0.98

    model.train()

    for i, (input, target) in enumerate(t):

        input, target = input.to(device), target.to(device)

        var_ip, var_tg = Variable(input), Variable(target)

        output = model(var_ip)

        loss = criterion(output, var_tg)



        running_loss = avg_beta * running_loss + (1-avg_beta) *loss.item()

        smoothed_loss = running_loss / (1 - avg_beta**(i+1))

        t.set_postfix(loss=smoothed_loss)



        lr = clr.calc_lr(smoothed_loss)

        if lr == -1 :

            break

        update_lr(optimizer, lr)



        # compute gradient and do SGD step

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
lr_find(model)
clr.plot()
load_checkpoint(model, 'before_clr.pth.tar')
epoch = 40
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
#sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, min_lr=1e-4)
onecycle = OneCycle(int(len(train_loader) * epoch /bs), 1e-2, use_cosine=True)
fit(model, scheduler=None, use_cycle=True)
save_checkpoint(model, True, 'before_clr_unfreeze.pth.tar')
for param in model.parameters():

    param.require_grad = True
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
clr = CLR(optimizer, len(train_loader), base_lr=1e-30)
lr_find(model)
clr.plot(start=0, end=-1)
load_checkpoint(model, 'before_clr_unfreeze.pth.tar')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-30, momentum=0.9, weight_decay=1e-4)
#sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, min_lr=1e-24)
onecycle = OneCycle(int(len(train_loader) * epoch /bs), 1e-29, use_cosine=True)
fit(model, scheduler=None, use_cycle=True)
ep_losses = []

for i in range(0, len(train_stats.losses), len(train_loader)):

    if i != 0 :

        ep_losses.append(train_stats.losses[i])

        

ep_lossesv = []

for i in range(0, len(test_stats.losses), len(valid_loader)):

    if(i != 0):

        ep_lossesv.append(test_stats.losses[i])
ep_accs = []

for i in range(0, len(train_stats.precs), len(train_loader)):

    if i != 0 :

        ep_accs.append(train_stats.precs[i])

        

ep_accsv = []

for i in range(0, len(test_stats.precs), len(valid_loader)):

    if(i != 0):

        ep_accsv.append(test_stats.precs[i])
plt.xlabel("Iterations")

plt.ylabel("Accuracy")

plt.yticks(np.arange(0, 100))

plt.plot(ep_accs, 'r', label='Train')

plt.plot(ep_accsv, 'b', label='Valid')

plt.legend()
plt.xlabel("Iterations")

plt.ylabel("Loss")

plt.yticks(np.arange(0, 100, step=10))

plt.plot(ep_losses, 'r', label='Train')

plt.plot(ep_lossesv, 'b', label='Valid')

plt.legend()
preds = []
with torch.no_grad():

    model.eval()

    for i, input in enumerate(test_loader):

        input = input.to(device)

        var_ip = Variable(input)

        output = model(var_ip)

        _, pred = torch.max(output, 1)

        preds.extend(pred.tolist())
len(preds)
preds
import csv
!ls
sb = open('final_sub.csv', 'w', newline='')

writer = csv.writer(sb)

writer.writerow(["ImageId", "Label"])



for i, pred in enumerate(preds):

    writer.writerow([i+1, pred])

    

sb.close()
from IPython.display import FileLink

FileLink('final_sub.csv')