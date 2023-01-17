import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import math
from torch import nn
from torch.nn import functional as F
import torchvision
import os
from torch.autograd import Variable
def strokes_to_img(in_strokes):
    in_strokes = eval(in_strokes)
    # make an agg figure
    fig, ax = plt.subplots()
    for x,y in in_strokes:
        ax.plot(x, y, linewidth=12.) #  marker='.',
    ax.axis('off')
    fig.canvas.draw()
    
    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    return (cv2.resize(X, (96, 96)) / 255.)[::-1]
class_files = os.listdir("../input/train_simplified/")
classes = {x[:-4]:i for i, x in enumerate(class_files)}
to_class = {i:x[:-4].replace(" ", "_") for i, x in enumerate(class_files)}
classes
dfs = [pd.read_csv("../input/train_simplified/" + x, nrows=10000)[["word", "drawing"]] for x in class_files]
df = pd.concat(dfs)
del dfs
df[:5]
n_samples = df.shape[0]
batch_size = 64

pick_order = np.arange(n_samples)
pick_per_epoch = n_samples // batch_size

def train_gen():
    while True:  # Infinity loop
        np.random.shuffle(pick_order)
        for i in range(pick_per_epoch):
            c_pick = pick_order[i*batch_size: (i+1)*batch_size]
            dfs = df.iloc[c_pick]
            out_imgs = list(map(strokes_to_img, dfs["drawing"]))
            X = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
            y = np.array([classes[x] for x in dfs["word"]])
            yield X, y
dataloaders = train_gen()
x,y = next(iter(dataloaders))
x.shape
def display_img(n):
    for i in range(n):
        plt.subplot(2,n//2,i+1)
        plt.imshow(x[i])
        plt.axis('off')
plt.show()
    
display_img(20)
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

device = torch.device('cuda')

criterion = nn.NLLLoss()
lr = 0.001
weight_decay = 5e-4
gamma=0.1
stepsize=60
epochs = 40
num_classes=340

class Model(nn.Module):
    def __init__(self, pretrained_model):
        super(Model,self).__init__()
        self.pretrained_model = pretrained_model
        self.base = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.linear = nn.Linear(512,340)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.linear(f)
        return 

ft_model = torchvision.models.resnet18(pretrained=True)
model = Model(ft_model)
opt = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)
scheduler = CyclicLR(opt,gamma=gamma,step_size=stepsize)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_gpu=True, num_epochs=25, mixup = False, alpha = 0.1):
    print("MIXUP".format(mixup))
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.batch_step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #for data in tqdm(dataloaders):
            for i, data in enumerate(dataloaders, 0):
                # get the inputs
                inputs, labels = data
                #augementation using mixup
                #if phase == 'train' and mixup:
                #    inputs = mixup_batch(inputs, alpha)
                # wrap them in Variable
                inputs = torch.from_numpy(inputs)
                labels = torch.from_numpy(labels)
                inputs = inputs.permute(0,3,1,2)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
import time
import tqdm
use_cuda=True
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model_ft = train_model(model=model,dataloaders=dataloaders,scheduler=scheduler,dataset_sizes=3400000,criterion=criterion, optimizer=opt)
x,y = next(iter(dataloaders))
x = torch.from_numpy(x)
x = x.permute(0,3,1,2)
k = model(x)
x.shape
