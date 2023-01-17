!pip3 install numpy pandas opencv-python torchvision Pillow tqdm albumentations
!sudo apt-get install libxrender1
!nvidia-smi --query-gpu=utilization.memory  -u --format=csv
import zipfile



with zipfile.ZipFile('./r5.zip', 'r') as zip_ref:

  zip_ref.extractall('./dataset3/train/')
from __future__ import division

from abc import ABCMeta, abstractproperty



import torch

# Кирилл baka <3

def _to_value(other):

    if isinstance(other, Metric):

        other = other.value

    return other





class Metric(metaclass=ABCMeta):



    @abstractproperty

    def value(self):

        raise NotImplementedError



    def __gt__(self, other):

        return self.value > _to_value(other)



    def __lt__(self, other):

        return self.value < _to_value(other)



    def __le__(self, other):

        return self.value <= _to_value(other)



    def __ge__(self, other):

        return self.value >= _to_value(other)







class Accuracy(Metric):



    def __init__(self, top_k=1):

        self.top_k = top_k

        self.correct = 0

        self.count = 0



    def update(self, output: torch.Tensor, target: torch.Tensor):

        assert output.size(0) == target.size(0)



        with torch.no_grad():

            _, pred = output.topk(self.top_k, 1, True, True)

            pred = pred.t()

            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_k = correct[:self.top_k].view(-1).float().sum(0, keepdim=True).item()



        self.correct += correct_k

        self.count += output.size(0)



    @property

    def value(self):

        return 100 * self.correct / self.count



    def __str__(self):

        return '{:.2f}%'.format(self.value)





class Average(Metric):



    def __init__(self):

        self.sum = 0

        self.count = 0



    def update(self, value, number=1):

        self.sum += value * number

        self.count += number



    @property

    def value(self):

        if self.count == 0:

            return float('inf')

        else:

            return self.sum / self.count



    def __str__(self):

        return '{:.4f}'.format(self.value)
import os



import torch



from tqdm import tqdm, trange



from metrics import Accuracy, Average



class Trainer:

    def __init__(self, model, optimizer, criterion, train_loader, valid_loader, test_loader, scheduler, device, epochs, save_dir):

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.train_loader = train_loader

        self.valid_loader = valid_loader

        self.test_loader = test_loader

        self.scheduler = scheduler

        self.device = device

        self.epochs = epochs

        self.save_dir = save_dir



        self.epoch = 1

        self.best_acc = 0



    def fit(self):

        epochs = range(self.epoch, self.epochs + 1)

        for self.epoch in epochs:

            self.scheduler.step()



            train_loss, train_acc = self.train()

            valid_loss, valid_acc = self.evaluate()



            self.save(os.path.join(self.save_dir, 'checkpoint.pth'))



            if valid_acc > self.best_acc:

                self.best_acc = valid_acc.value

                self.save(os.path.join(self.save_dir, 'best.pth'))



            print(f'EPOCH: {self.epoch}: train loss: {train_loss}, train acc: {train_acc}, '

                                   f'valid loss: {valid_loss}, valid acc: {valid_acc}, '

                                   f'best valid acc: {self.best_acc:.2f}')



    def train(self):

        self.model.train()



        train_loss = Average()

        train_acc = Accuracy()



        train_loader = tqdm(self.train_loader, desc='Train')

        for x, y in train_loader:

            x = x.to(self.device)

            y = y.to(self.device)



            output = self.model(x)

            loss = self.criterion(output, y)



            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()



            train_loss.update(loss.item(), number=x.size(0))

            train_acc.update(output, y)



            train_loader.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}.')



        return train_loss, train_acc



    def evaluate(self):

        self.model.eval()



        valid_loss = Average()

        valid_acc = Accuracy()



        with torch.no_grad():

            valid_loader = tqdm(self.valid_loader, desc='Validate')

            for x, y in valid_loader:

                x = x.to(self.device)

                y = y.to(self.device)



                output = self.model(x)

                loss = self.criterion(output, y)



                valid_loss.update(loss.item(), number=x.size(0))

                valid_acc.update(output, y)



                valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')



        return valid_loss, valid_acc



    def save(self, f):

        self.model.eval()



        checkpoint = {

            'model': self.model.state_dict(),

            'optimizer': self.optimizer.state_dict(),

            'scheduler': self.scheduler.state_dict(),

            'epoch': self.epoch,

            'best_acc': self.best_acc

        }



        dirname = os.path.dirname(f)

        if dirname:

            os.makedirs(dirname, exist_ok=True)



        torch.save(checkpoint, f)



    def resume(self, f):

        checkpoint = torch.load(f, map_location=self.device)



        self.model.load_state_dict(checkpoint['model'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.scheduler.load_state_dict(checkpoint['scheduler'])



        self.epoch = checkpoint['epoch'] + 1

        self.best_acc = checkpoint['best_acc']

        

    def submit(self):

        self.model.to(device)

        self.model.eval()

        pred_list = []

        names_list = []

        for images, image_names in self.test_loader:

            with torch.no_grad():

                images = images.to(device)

                output = self.model(images)

                pred = F.softmax(output)

                pred = torch.argmax(pred, dim=1).cpu().numpy()

                pred_list += [p.item() for p in pred]

                names_list += [name for name in image_names]



        sample_submission = pd.read_csv('./ndataset/sample_submission.csv')

        sample_submission.image_name = names_list

        sample_submission.label = pred_list

        sample_submission.to_csv('./submission' + str(round(self.best_acc, 2)) + '.csv', index=False)

        
import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import torch.backends.cudnn as cudnn

import torchvision

import torchvision.transforms as transforms

import os

import numpy as np

import pandas as pd

import random

import cv2

import time

from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split

import albumentations

from albumentations.pytorch import ToTensorV2 as AT



# from efficientnet_pytorch import EfficientNet



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



BATCH = 64

IMAGE_SIZE = 224

CLASSES = 8

WORKERS = 2
PATH = './dataset3/'

train_path = PATH + "train/"

test_path = PATH + "test/"

train_list = os.listdir(train_path)

test_list = os.listdir(test_path)

print(len(train_list), len(test_list))





data_transforms = albumentations.Compose([

    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),

    albumentations.ChannelDropout(p=0.3),

    albumentations.ChannelShuffle(p=0.3),

    # albumentations.ToGray(p=1, always_apply=True),

    albumentations.Downscale(p=0.3),

    albumentations.ShiftScaleRotate(),

    albumentations.Normalize(),

    AT()

    ])

data_transforms_test = albumentations.Compose([

    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),

    # albumentations.ToGray(p=1, always_apply=True),

    albumentations.Normalize(),

    AT()

    ])





class ChartsDataset(Dataset):



    def __init__(self, path, img_list, transform=None, mode='train'):

        self.path = path

        self.img_list = img_list

        self.transform = transform

        self.mode = mode



    def __len__(self):

        return len(self.img_list)



    def __getitem__(self, idx):

        image_name = self.img_list[idx]



        if image_name.split(".")[1] == "gif":

            gif = cv2.VideoCapture(self.path + image_name)

            _, image = gif.read()

        else:

            image = cv2.imread(self.path + image_name)



        try:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except:

            print(image_name)

            raise EnvironmentError()



        label = 0

        if "bar_chart" in image_name:

            label = 1

        elif "diagram" in image_name:

            label = 2

        elif "flow_chart" in image_name:

            label = 3

        elif "graph" in image_name:

            label = 4

        elif "growth_chart" in image_name:

            label = 5

        elif "pie_chart" in image_name:

            label = 6

        elif "table" in image_name:

            label = 7

        else:

            label = 0



        if self.transform:

            augmented = self.transform(image=image)

            image = augmented["image"]



        if self.mode == "train":

            return image, label

        else:

            return image, image_name





trainset = ChartsDataset(train_path, train_list,  transform=data_transforms)

testset = ChartsDataset(test_path, test_list,  transform=data_transforms_test, mode="test")



valid_size = int(len(train_list) * 0.1)

train_set, valid_set = torch.utils.data.random_split(trainset,

                                    (len(train_list)-valid_size, valid_size))



trainloader = torch.utils.data.DataLoader(train_set, pin_memory=True,

                                          batch_size=BATCH, shuffle=True)



validloader = torch.utils.data.DataLoader(valid_set, pin_memory=True,

                                          batch_size=BATCH, shuffle=True)



testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH,

                                         num_workers=WORKERS)
import math

import torch

from torch.optim.optimizer import Optimizer





class AdamW(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0, amsgrad=False):

        if not 0.0 <= betas[0] < 1.0:

            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:

            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay, amsgrad=amsgrad)

        #super(AdamW, self).__init__(params, defaults)

        super().__init__(params, defaults)



    def step(self, closure=None):

        """Performs a single optimization step.

        Arguments:

            closure (callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data

                if grad.is_sparse:

                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                amsgrad = group['amsgrad']



                state = self.state[p]



                # State initialization

                if len(state) == 0:

                    state['step'] = 0

                    # Exponential moving average of gradient values

                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values

                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    if amsgrad:

                        # Maintains max of all exp. moving avg. of sq. grad. values

                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if amsgrad:

                    max_exp_avg_sq = state['max_exp_avg_sq']

                beta1, beta2 = group['betas']



                state['step'] += 1



                # Decay the first and second moment running average coefficient

                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:

                    # Maintains the maximum of all 2nd moment running avg. till now

                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                    # Use the max. for normalizing running avg. of gradient

                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])

                else:

                    denom = exp_avg_sq.sqrt().add_(group['eps'])



                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1



                p.data.mul_(1 - group['weight_decay']).addcdiv_(-step_size, exp_avg, denom)



        return loss

    

from torch.optim import Optimizer

from torch.optim.lr_scheduler import _LRScheduler

import math

import torch





class ReduceMaxLROnRestart:

    def __init__(self, ratio=0.75):

        self.ratio = ratio



    def __call__(self, eta_min, eta_max):

        return eta_min, eta_max * self.ratio





class ExpReduceMaxLROnIteration:

    def __init__(self, gamma=1):

        self.gamma = gamma



    def __call__(self, eta_min, eta_max, iterations):

        return eta_min, eta_max * self.gamma ** iterations





class CosinePolicy:

    def __call__(self, t_cur, restart_period):

        return 0.5 * (1. + math.cos(math.pi *

                                    (t_cur / restart_period)))





class ArccosinePolicy:

    def __call__(self, t_cur, restart_period):

        return (math.acos(max(-1, min(1, 2 * t_cur

                                      / restart_period - 1))) / math.pi)





class TriangularPolicy:

    def __init__(self, triangular_step=0.5):

        self.triangular_step = triangular_step



    def __call__(self, t_cur, restart_period):

        inflection_point = self.triangular_step * restart_period

        point_of_triangle = (t_cur / inflection_point

                             if t_cur < inflection_point

                             else 1.0 - (t_cur - inflection_point)

                             / (restart_period - inflection_point))

        return point_of_triangle





class CyclicLRWithRestarts(_LRScheduler):



    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,

                 t_mult=2, last_epoch=-1, verbose=False,

                 policy="cosine", policy_fn=None, min_lr=1e-7,

                 eta_on_restart_cb=None, eta_on_iteration_cb=None,

                 gamma=1.0, triangular_step=0.5):



        if not isinstance(optimizer, Optimizer):

            raise TypeError('{} is not an Optimizer'.format(

                type(optimizer).__name__))



        self.optimizer = optimizer



        if last_epoch == -1:

            for group in optimizer.param_groups:

                group.setdefault('initial_lr', group['lr'])

                group.setdefault('minimum_lr', min_lr)

        else:

            for i, group in enumerate(optimizer.param_groups):

                if 'initial_lr' not in group:

                    raise KeyError("param 'initial_lr' is not specified "

                                   "in param_groups[{}] when resuming an"

                                   " optimizer".format(i))



        self.base_lrs = [group['initial_lr'] for group

                         in optimizer.param_groups]



        self.min_lrs = [group['minimum_lr'] for group

                        in optimizer.param_groups]



        self.base_weight_decays = [group['weight_decay'] for group

                                   in optimizer.param_groups]



        self.policy = policy

        self.eta_on_restart_cb = eta_on_restart_cb

        self.eta_on_iteration_cb = eta_on_iteration_cb

        if policy_fn is not None:

            self.policy_fn = policy_fn

        elif self.policy == "cosine":

            self.policy_fn = CosinePolicy()

        elif self.policy == "arccosine":

            self.policy_fn = ArccosinePolicy()

        elif self.policy == "triangular":

            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)

        elif self.policy == "triangular2":

            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)

            self.eta_on_restart_cb = ReduceMaxLROnRestart(ratio=0.5)

        elif self.policy == "exp_range":

            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)

            self.eta_on_iteration_cb = ExpReduceMaxLROnIteration(gamma=gamma)



        self.last_epoch = last_epoch

        self.batch_size = batch_size

        self.epoch_size = epoch_size



        self.iteration = 0

        self.total_iterations = 0



        self.t_mult = t_mult

        self.verbose = verbose

        self.restart_period = math.ceil(restart_period)

        self.restarts = 0

        self.t_epoch = -1

        self.epoch = -1



        self.eta_min = 0

        self.eta_max = 1



        self.end_of_period = False

        self.batch_increments = []

        self._set_batch_increment()



    def _on_restart(self):

        if self.eta_on_restart_cb is not None:

            self.eta_min, self.eta_max = self.eta_on_restart_cb(self.eta_min,

                                                                self.eta_max)



    def _on_iteration(self):

        if self.eta_on_iteration_cb is not None:

            self.eta_min, self.eta_max = self.eta_on_iteration_cb(self.eta_min,

                                                                  self.eta_max,

                                                                  self.total_iterations)



    def get_lr(self, t_cur):

        eta_t = (self.eta_min + (self.eta_max - self.eta_min)

                 * self.policy_fn(t_cur, self.restart_period))



        weight_decay_norm_multi = math.sqrt(self.batch_size /

                                            (self.epoch_size *

                                             self.restart_period))



        lrs = [min_lr + (base_lr - min_lr) * eta_t for base_lr, min_lr

               in zip(self.base_lrs, self.min_lrs)]

        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi

                         for base_weight_decay in self.base_weight_decays]



        if (self.t_epoch + 1) % self.restart_period < self.t_epoch:

            self.end_of_period = True



        if self.t_epoch % self.restart_period < self.t_epoch:

            if self.verbose:

                print("Restart {} at epoch {}".format(self.restarts + 1,

                                                      self.last_epoch))

            self.restart_period = math.ceil(self.restart_period * self.t_mult)

            self.restarts += 1

            self.t_epoch = 0

            self._on_restart()

            self.end_of_period = False



        return zip(lrs, weight_decays)



    def _set_batch_increment(self):

        d, r = divmod(self.epoch_size, self.batch_size)

        batches_in_epoch = d + 2 if r > 0 else d + 1

        self.iteration = 0

        self.batch_increments = torch.linspace(0, 1, batches_in_epoch).tolist()



    def step(self):

        self.last_epoch += 1

        self.t_epoch += 1

        self._set_batch_increment()

        self.batch_step()



    def batch_step(self):

        try:

            t_cur = self.t_epoch + self.batch_increments[self.iteration]

            self._on_iteration()

            self.iteration += 1

            self.total_iterations += 1

        except (IndexError):

            raise StopIteration("Epoch size and batch size used in the "

                                "training loop and while initializing "

                                "scheduler should be the same.")



        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,

                                                   self.get_lr(t_cur)):

            param_group['lr'] = lr

            param_group['weight_decay'] = weight_decay
torch.backends.cudnn.benchmark = True

from efficientnet_pytorch import EfficientNet



EPOCHS = 24



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)



# model = torchvision.models.resnet18(pretrained=True, progress=True)



# for param in model.parameters():

#     param.requires_grad = False



# in_features = model.fc.in_features

# model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

# model.fc = nn.Linear(in_features, 8)



# model = EfficientNet.from_pretrained("efficientnet-b3", advprop=True, num_classes=8, in_channels=3)

model = EfficientNet.from_pretrained("efficientnet-b1", advprop=False, num_classes=8, in_channels=3)



# for param in model.parameters():

#     param.requires_grad = False



for name, child in model.named_children():

    if name in ['_swish', '_fc', '_dropout', '_avg_pooling', '_bn1', '_conv_head', '_blocks']:

        if name == '_blocks':

            for block in child[4:]:

                for param in block.parameters():

                    param.requires_grad = True

        else:        

            for param in child.parameters():

                param.requires_grad = True

    else:

        for param in child.parameters():

            param.requires_grad = False



model = model.to(device)

# model = nn.DataParallel(model)



criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0, eps=1e-8, amsgrad=False)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3, weight_decay=1e-4, momentum=0.9, nesterov=True);

# optimizer = optim.RMSprop(model.parameters(), lr=0.0125, weight_decay=0, eps=1e-3, momentum=0.9)

# optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5, eps=1e-8, amsgrad=False)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98738885893)

# scheduler = CyclicLRWithRestarts(optimizer, BATCH, EPOCHS, restart_period=5, t_mult=1.2, policy="cosine")



trainer = Trainer(model, optimizer, criterion, trainloader, validloader, testloader, scheduler, device, EPOCHS, './_ e1 sgd nogr grad2 l1 0')
for name, child in model.named_children():

    print(name)

    

for param in model.parameters():

    print(param.requires_grad)
file = './_ e1 sgd nogr grad2 l1 0/checkpoint.pth'

trainer.save_dir = os.path.dirname(file)

trainer.resume(file)
trainer.fit()
trainer.submit()