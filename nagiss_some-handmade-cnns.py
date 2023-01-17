import os

from time import time

from random import randint

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np



import torch

import torch.nn as nn

import torch.nn.functional as F



device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device={device}")



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    

from pathlib import Path



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))

test_tasks = sorted(os.listdir(test_path))

print(len(training_tasks), len(evaluation_tasks), len(test_tasks))
cmap = colors.ListedColormap(

    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,

# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

plt.figure(figsize=(5, 2), dpi=200)

plt.imshow([list(range(10))], cmap=cmap, norm=norm)

plt.xticks(list(range(10)))

plt.yticks([])

plt.show()



def plot_task(task):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Test-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    

    plt.tight_layout()

    plt.show()

    

def get_data(task_filename):

    with open(task_filename, 'r') as f:

        task = json.load(f)

    return task



num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

color2num = {c: n for n, c in enumerate(num2color)}
def check(task, pred_func):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        t_pred = pred_func(t_in)

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        axs[2][fig_num].imshow(t_pred, cmap=cmap, norm=norm)

        axs[2][fig_num].set_title(f'Train-{i} pred')

        axs[2][fig_num].set_yticks(list(range(t_pred.shape[0])))

        axs[2][fig_num].set_xticks(list(range(t_pred.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        t_pred = pred_func(t_in)

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Test-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        axs[2][fig_num].imshow(t_pred, cmap=cmap, norm=norm)

        axs[2][fig_num].set_title(f'Test-{i} pred')

        axs[2][fig_num].set_yticks(list(range(t_pred.shape[0])))

        axs[2][fig_num].set_xticks(list(range(t_pred.shape[1])))

        fig_num += 1
class ArcDataset(torch.utils.data.Dataset):

    def __init__(self, task=None, mode="train", augment=False):

        if task is not None:

            assert mode in ["train", "test"]

            self.mode = mode

            self.task = task[mode]

        self.augment = augment



    def __len__(self):

        return len(self.task)

    

    def __getitem__(self, index):

        t = self.task[index]

        t_in = torch.tensor(t["input"])

        t_out = torch.tensor(t["output"])

        t_in, t_out = self.preprocess(t_in, t_out)

        return t_in, t_out

    

    def preprocess(self, t_in, t_out):

        if self.augment:

            t_in, t_out = self._random_rotate(t_in, t_out)

        t_in = self._one_hot_encode(t_in)

        t_out = self._one_hot_encode(t_out)

        return t_in, t_out

    

    def _one_hot_encode(self, x):

        return torch.eye(10)[x].permute(2, 0, 1)

    

    def _random_rotate(self, t_in, t_out):

        t_in_shape = t_in.shape

        t_out_shape = t_out.shape

        t_in = t_in.reshape(-1, *t_in_shape[-2:])

        t_out = t_out.reshape(-1, *t_out_shape[-2:])

        r = randint(0, 7)

        if r%2 == 0:

            t_in = t_in.permute(0, 2, 1)

            t_out = t_out.permute(0, 2, 1)

        r //= 2

        if r%2 == 0:

            t_in = t_in[:, :, torch.arange(t_in.shape[-1]-1, -1, -1)]

            t_out = t_out[:, :, torch.arange(t_out.shape[-1]-1, -1, -1)]

        r //= 2

        if r%2 == 0:

            t_in = t_in[:, torch.arange(t_in.shape[-2]-1, -1, -1), :]

            t_out = t_out[:, torch.arange(t_out.shape[-2]-1, -1, -1), :]

        t_in = t_in.reshape(*t_in_shape[:-2], *t_in.shape[-2:])

        t_out = t_out.reshape(*t_out_shape[:-2], *t_out.shape[-2:])

        return t_in, t_out

    

def device_collate(batch):

    return tuple(map(lambda x: torch.stack(x).to(device), zip(*batch)))
def hinge_loss(y_pred, y_true):

    loss = y_pred.clone()

    loss[y_true>0.5] = 1-loss[y_true>0.5]

    loss[loss<0] = 0

    return loss.sum(0).mean()
task = get_data(str(training_path / training_tasks[330]))

plot_task(task)
class Task330Net(nn.Module):

    def __init__(self):

        super(Task330Net, self).__init__()

        siz = 16

        self.conv1 = nn.Conv2d(10, siz, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = torch.nn.Dropout2d(p=0.1)

        self.conv1x1 = nn.Conv2d(siz, 10, 1)



    def forward(self, x):

        x2 = self.conv1(x)

        x2 = self.relu(x2)

        x2 = self.dropout(x2)

        x2 = self.conv1x1(x2)

        x = x + x2  # skip connection

        return x
train_dataset = ArcDataset(task, mode="train", augment=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=device_collate)

valid_dataset = ArcDataset(task, mode="test", augment=False)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=device_collate)
net = Task330Net().to(device)

criterion = hinge_loss

#optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

t0 = time()



for param in net.named_parameters():

    print(f"{param[0]:>15} {list(param[1].shape)}")



for epoch in range(5000):

    train_loss = valid_loss = 0.0

    train_loss_denom = valid_loss_denom = 0

    

    ####################

    # train

    ####################

    net.train()

    for i, (feature, target) in enumerate(train_dataloader):

        outputs = net(feature)

        loss = criterion(outputs, target)



        # backprop

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    

        # record

        train_loss += loss.item()

        train_loss_denom += feature.shape[0]



    train_loss /= train_loss_denom



    ####################

    # eval

    ####################

    net.eval()

    with torch.no_grad():

        for i, (feature, target) in enumerate(valid_dataloader):

            feature = feature.to(device)

            target = target.to(device)

            

            outputs = net(feature)

            loss = criterion(outputs, target)



            # record

            valid_loss += loss.item()

            valid_loss_denom += feature.shape[0]



    valid_loss /= valid_loss_denom





    if epoch%100==0:

        print(f"epoch {epoch:4d}  |  train_loss: {train_loss:5.6f}  valid_loss: {valid_loss:5.6f}  |  time: {time()-t0:7.1f} sec")



#         if best_loss > valid_loss:

#             best_loss = valid_loss

#             filename = f"./work/trained_weight/{MODEL_NAME}_epoch{epoch:03d}_loss{valid_loss:.3f}.pth"

#             torch.save(net.state_dict(), filename)
def task_train330(x, net):

    def one_hot_decode(x):

        return x.argmax(0)

    net.eval()

    with torch.no_grad():

        x = torch.tensor(x).to(device)

        y_dummy = x.clone()

        dataset = ArcDataset(augment=False)

        x = dataset.preprocess(x, y_dummy)[0].unsqueeze(0)

        y = net(x).detach()

        y = one_hot_decode(y.squeeze(0))

    y = y.to("cpu").numpy()

    return y

    



check(task, lambda x: task_train330(x, net))
task = get_data(str(training_path / training_tasks[301]))

plot_task(task)
class Task301Net(nn.Module):

    def __init__(self):

        super(Task301Net, self).__init__()

        siz = 16

        self.conv1 = nn.Conv2d(10, siz, 3, padding=1)

        self.conv2 = nn.Conv2d(siz, siz, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = torch.nn.Dropout2d(p=0.1)

        self.conv1x1 = nn.Conv2d(siz, 10, 1)



    def forward(self, x):

        x2 = self.conv1(x)

        x2 = self.relu(x2)

        x2 = self.conv2(x2)

        x2 = self.relu(x2)

        x2 = self.dropout(x2)

        x2 = self.conv1x1(x2)

        x = x + x2  # skip connection

        return x
train_dataset = ArcDataset(task, mode="train", augment=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=device_collate)

valid_dataset = ArcDataset(task, mode="test", augment=False)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=device_collate)
net = Task301Net().to(device)

criterion = hinge_loss

#optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)



t0 = time()



for param in net.named_parameters():

    print(f"{param[0]:>15} {list(param[1].shape)}")



for epoch in range(5000):

    train_loss = valid_loss = 0.0

    train_loss_denom = valid_loss_denom = 0

    

    ####################

    # train

    ####################

    net.train()

    for i, (feature, target) in enumerate(train_dataloader):

        outputs = net(feature)

        loss = criterion(outputs, target)



        # backprop

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    

        # record

        train_loss += loss.item()

        train_loss_denom += feature.shape[0]



    train_loss /= train_loss_denom



    ####################

    # eval

    ####################

    net.eval()

    with torch.no_grad():

        for i, (feature, target) in enumerate(valid_dataloader):

            feature = feature.to(device)

            target = target.to(device)

            

            outputs = net(feature)

            loss = criterion(outputs, target)



            # record

            valid_loss += loss.item()

            valid_loss_denom += feature.shape[0]



    valid_loss /= valid_loss_denom





    if epoch%100==0:

        print(f"epoch {epoch:4d}  |  train_loss: {train_loss:5.6f}  valid_loss: {valid_loss:5.6f}  |  time: {time()-t0:7.1f} sec")



#         if best_loss > valid_loss:

#             best_loss = valid_loss

#             filename = f"./work/trained_weight/{MODEL_NAME}_epoch{epoch:03d}_loss{valid_loss:.3f}.pth"

#             torch.save(net.state_dict(), filename)
def task_train301(x, net):

    def one_hot_decode(x):

        return x.argmax(0)

    net.eval()

    with torch.no_grad():

        x = torch.tensor(x).to(device)

        y_dummy = x.clone()

        dataset = ArcDataset(augment=False)

        x = dataset.preprocess(x, y_dummy)[0].unsqueeze(0)

        y = net(x).detach()

        y = one_hot_decode(y.squeeze(0))

    y = y.to("cpu").numpy()

    return y

    



check(task, lambda x: task_train301(x, net))
task = get_data(str(training_path / training_tasks[343]))

plot_task(task)
class Task343Net(nn.Module):

    def __init__(self):

        super(Task343Net, self).__init__()

        siz = 16

        self.conv1 = nn.Conv2d(10, siz, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = torch.nn.Dropout2d(p=0.1)

        self.conv1x1 = nn.Conv2d(siz, 10, 1)



    def forward(self, x):

        x2 = self.conv1(x)

        x2 = self.relu(x2)

        x2 = self.dropout(x2)

        x2 = self.conv1x1(x2)

        x = x + x2  # skip connection

        return x
train_dataset = ArcDataset(task, mode="train", augment=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=device_collate)

valid_dataset = ArcDataset(task, mode="test", augment=False)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=device_collate)
net = Task343Net().to(device)

criterion = hinge_loss

#optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

t0 = time()



for param in net.named_parameters():

    print(f"{param[0]:>15} {list(param[1].shape)}")



for epoch in range(5000):

    train_loss = valid_loss = 0.0

    train_loss_denom = valid_loss_denom = 0

    

    ####################

    # train

    ####################

    net.train()

    for i, (feature, target) in enumerate(train_dataloader):

        outputs = net(feature)

        loss = criterion(outputs, target)



        # backprop

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    

        # record

        train_loss += loss.item()

        train_loss_denom += feature.shape[0]



    train_loss /= train_loss_denom



    ####################

    # eval

    ####################

    net.eval()

    with torch.no_grad():

        for i, (feature, target) in enumerate(valid_dataloader):

            feature = feature.to(device)

            target = target.to(device)

            

            outputs = net(feature)

            loss = criterion(outputs, target)



            # record

            valid_loss += loss.item()

            valid_loss_denom += feature.shape[0]



    valid_loss /= valid_loss_denom





    if epoch%100==0:

        print(f"epoch {epoch:4d}  |  train_loss: {train_loss:5.6f}  valid_loss: {valid_loss:5.6f}  |  time: {time()-t0:7.1f} sec")



#         if best_loss > valid_loss:

#             best_loss = valid_loss

#             filename = f"./work/trained_weight/{MODEL_NAME}_epoch{epoch:03d}_loss{valid_loss:.3f}.pth"

#             torch.save(net.state_dict(), filename)
def task_train343(x, net):

    def one_hot_decode(x):

        return x.argmax(0)

    net.eval()

    with torch.no_grad():

        x = torch.tensor(x).to(device)

        y_dummy = x.clone()

        dataset = ArcDataset(augment=False)

        x = dataset.preprocess(x, y_dummy)[0].unsqueeze(0)

        y = net(x).detach()

        y = one_hot_decode(y.squeeze(0))

    y = y.to("cpu").numpy()

    return y

    



check(task, lambda x: task_train343(x, net))
task = get_data(str(training_path / training_tasks[368]))

plot_task(task)
class Task368Net(nn.Module):

    def __init__(self):

        super(Task368Net, self).__init__()

        siz = 16

        self.conv1 = nn.Conv2d(10, siz, 2, padding=1)

        self.conv2 = nn.Conv2d(siz, siz, 2, padding=0)

        self.conv3 = nn.Conv2d(siz, siz, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = torch.nn.Dropout2d(p=0.1)

        self.conv4 = nn.Conv2d(siz, 10, 1)



    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.conv3(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv4(x)

        return x

    

# class Task368Net(nn.Module):  # does not work

#     def __init__(self):

#         super(Task368Net, self).__init__()

#         siz = 16

#         self.conv1 = nn.Conv2d(10, siz, 3, padding=1)

#         self.relu = nn.ReLU(inplace=True)

#         self.dropout = torch.nn.Dropout2d(p=0.1)

#         self.conv1x1 = nn.Conv2d(siz, 10, 1)



#     def forward(self, x):

#         x2 = self.conv1(x)

#         x2 = self.relu(x2)

#         x2 = self.dropout(x2)

#         x2 = self.conv1x1(x2)

#         x = x + x2  # skip connection

#         return x
train_dataset = ArcDataset(task, mode="train", augment=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=device_collate)

valid_dataset = ArcDataset(task, mode="test", augment=False)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=device_collate)
net = Task368Net().to(device)

criterion = hinge_loss

#optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

t0 = time()



for param in net.named_parameters():

    print(f"{param[0]:>15} {list(param[1].shape)}")



for epoch in range(5000):

    train_loss = valid_loss = 0.0

    train_loss_denom = valid_loss_denom = 0

    

    ####################

    # train

    ####################

    net.train()

    for i, (feature, target) in enumerate(train_dataloader):

        outputs = net(feature)

        loss = criterion(outputs, target)



        # backprop

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    

        # record

        train_loss += loss.item()

        train_loss_denom += feature.shape[0]



    train_loss /= train_loss_denom



    ####################

    # eval

    ####################

    net.eval()

    with torch.no_grad():

        for i, (feature, target) in enumerate(valid_dataloader):

            feature = feature.to(device)

            target = target.to(device)

            

            outputs = net(feature)

            loss = criterion(outputs, target)



            # record

            valid_loss += loss.item()

            valid_loss_denom += feature.shape[0]



    valid_loss /= valid_loss_denom





    if epoch%100==0:

        print(f"epoch {epoch:4d}  |  train_loss: {train_loss:5.6f}  valid_loss: {valid_loss:5.6f}  |  time: {time()-t0:7.1f} sec")



#         if best_loss > valid_loss:

#             best_loss = valid_loss

#             filename = f"./work/trained_weight/{MODEL_NAME}_epoch{epoch:03d}_loss{valid_loss:.3f}.pth"

#             torch.save(net.state_dict(), filename)
def task_train368(x, net):

    def one_hot_decode(x):

        return x.argmax(0)

    net.eval()

    with torch.no_grad():

        x = torch.tensor(x).to(device)

        y_dummy = x.clone()

        dataset = ArcDataset(augment=False)

        x = dataset.preprocess(x, y_dummy)[0].unsqueeze(0)

        y = net(x).detach()

        y = one_hot_decode(y.squeeze(0))

    y = y.to("cpu").numpy()

    return y

    



check(task, lambda x: task_train368(x, net))
task = get_data(str(training_path / training_tasks[351]))

plot_task(task)
class Task351Net(nn.Module):

    def __init__(self):

        super(Task351Net, self).__init__()

        siz = 16

        self.conv1 = nn.Conv2d(10, siz, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = torch.nn.Dropout2d(p=0.1)

        self.conv1x1 = nn.Conv2d(siz, 10, 1)



    def forward(self, x):

        x2 = self.conv1(x)

        x2 = self.relu(x2)

        x2 = self.dropout(x2)

        x2 = self.conv1x1(x2)

        x = x + x2  # skip connection

        return x

    

train_dataset = ArcDataset(task, mode="train", augment=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=device_collate)

valid_dataset = ArcDataset(task, mode="test", augment=False)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=device_collate)
net = Task351Net().to(device)

criterion = hinge_loss

#optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

t0 = time()



for param in net.named_parameters():

    print(f"{param[0]:>15} {list(param[1].shape)}")



for epoch in range(5000):

    train_loss = valid_loss = 0.0

    train_loss_denom = valid_loss_denom = 0

    

    ####################

    # train

    ####################

    net.train()

    for i, (feature, target) in enumerate(train_dataloader):

        outputs = net(feature)

        loss = criterion(outputs, target)



        # backprop

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    

        # record

        train_loss += loss.item()

        train_loss_denom += feature.shape[0]



    train_loss /= train_loss_denom



    ####################

    # eval

    ####################

    net.eval()

    with torch.no_grad():

        for i, (feature, target) in enumerate(valid_dataloader):

            feature = feature.to(device)

            target = target.to(device)

            

            outputs = net(feature)

            loss = criterion(outputs, target)



            # record

            valid_loss += loss.item()

            valid_loss_denom += feature.shape[0]



    valid_loss /= valid_loss_denom





    if epoch%100==0:

        print(f"epoch {epoch:4d}  |  train_loss: {train_loss:5.6f}  valid_loss: {valid_loss:5.6f}  |  time: {time()-t0:7.1f} sec")

def task_train351(x, net):

    def one_hot_decode(x):

        return x.argmax(0)

    net.eval()

    with torch.no_grad():

        x = torch.tensor(x).to(device)

        y_dummy = x.clone()

        dataset = ArcDataset(augment=False)

        x = dataset.preprocess(x, y_dummy)[0].unsqueeze(0)

        y = net(x).detach()

        y = one_hot_decode(y.squeeze(0))

    y = y.to("cpu").numpy()

    return y

    



check(task, lambda x: task_train351(x, net))