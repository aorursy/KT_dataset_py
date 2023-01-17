import csv

import math

import numpy as np

import os

import pandas as pd



import torch

from torch import nn

from torch.optim.optimizer import Optimizer, required

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T, utils, models, datasets
class Config:

    split_dataset = True

    valid_frac = 0.1

    

    model_path = '/kaggle/saves/kannada.pth'

    train_valid_csv = "/kaggle/input/Kannada-MNIST/train.csv"

    train_csv = "/kaggle/saves/train_.csv"

    valid_csv = "/kaggle/saves/valid_.csv"

    test_csv = "/kaggle/input/Kannada-MNIST/test.csv"

    output_csv = "submission.csv"



    train_transforms = T.Compose([

        T.ToPILImage(),

        T.RandomPerspective(),

        T.RandomRotation(15),

        T.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2),

        T.RandomHorizontalFlip(p=0.5),

        T.ToTensor(),

        T.Normalize((0.1307,), (0.3081,))

    ])



    valid_transforms = T.Compose([

        T.ToTensor(),

        T.Normalize((0.1307,), (0.3081,))

    ])



    device = torch.device("cuda:0")

    model = models.mobilenet_v2

#     print(model)

    criterion = nn.CrossEntropyLoss()

    epochs = 30

    batch_size = 32



    log_interval = 337

    min_loss = 10**9

if Config.split_dataset:

    if not os.path.exists("/kaggle/saves"):

        os.makedirs("/kaggle/saves")

    

    train_valid = pd.read_csv(Config.train_valid_csv)

    valid = train_valid.sample(frac=Config.valid_frac)

    train = train_valid.drop(valid.index)

    valid.to_csv(Config.valid_csv, index=False)

    train.to_csv(Config.train_csv, index=False)

    

    for root, dirs, files in os.walk("/kaggle"):

        print(root, dirs, files)
class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):

        if not 0.0 <= lr:

            raise ValueError("Invalid learning rate: {}".format(lr))

        if not 0.0 <= eps:

            raise ValueError("Invalid epsilon value: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:

            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:

            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        

        self.degenerated_to_sgd = degenerated_to_sgd

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):

            for param in params:

                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):

                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = group['buffer'][int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    elif self.degenerated_to_sgd:

                        step_size = 1.0 / (1 - beta1 ** state['step'])

                    else:

                        step_size = -1

                    buffered[2] = step_size



                # more conservative since it's an approximated value

                if N_sma >= 5:

                    if group['weight_decay'] != 0:

                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)

                    p.data.copy_(p_data_fp32)

                elif step_size > 0:

                    if group['weight_decay'] != 0:

                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                    p.data.copy_(p_data_fp32)



        return loss

class CsvImages(Dataset):

    def __init__(self, csv_file, has_labels=True, transform=None):

        self.df = pd.read_csv(csv_file)

        self.has_labels = has_labels

        self.transform = transform



    def __len__(self):

        return self.df.shape[0]



    def __getitem__(self, i):

        sample = self.df.loc[i]

        if self.has_labels:

            label = sample.pop('label')

        else:

            id = sample.pop('id')

        sample = sample.to_numpy(np.uint8).reshape(28, 28, 1)



        if self.transform is not None:

            sample = self.transform(sample)



        if self.has_labels:

            return sample, label

        else:

            return sample, id
def save(model, valid_loss):

    print(f'Saving model with best validation loss = {valid_loss:.7f} at {Config.model_path}\n')

    torch.save(model.state_dict(), Config.model_path)

    

def load(model):

    print(f'Loading best version of the model, with validation loss = {Config.min_loss:.7f}\n')

    model.load_state_dict(torch.load(Config.model_path))

    model.eval()



def train(model, criterion, device, train_loader, optimizer, epoch):

    model.train()

    sum_loss = 0

    correct = 0

    num_batches = len(train_loader)

    print(f'\n\nTrain Epoch: {epoch}\n')

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        sum_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()

        optimizer.step()

        if batch_idx % Config.log_interval == 0 and batch_idx != 0:

            progress = 100. * (batch_idx+1) // len(train_loader)

            num_examples = (batch_idx+1) * Config.batch_size

            avg_acc = 100. * correct / num_examples

            avg_loss = sum_loss / (batch_idx+1)

            print(f'[{batch_idx}/{num_batches} ({progress}%)]\tAverage loss: {avg_loss:.7f}\tAccuracy: {correct}/{num_examples} ({avg_acc:.2f}%)\n', end='\r')





def valid(model, criterion, device, valid_loader):

    model.eval()

    valid_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in valid_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            valid_loss += criterion(output, target).item()

            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()



    valid_loss /= len(valid_loader.dataset)



    val_acc = 100. * correct / len(valid_loader.dataset)

    print(f'\nValid set: Average loss: {valid_loss:.7f}, Accuracy: {correct}/{len(valid_loader.dataset)} ({val_acc:.2f}%)\n')

    if valid_loss < Config.min_loss:

        Config.min_loss = valid_loss

        save(model, valid_loss)

    

def test(model, device, test_loader, output_csv):

    model.eval()

    with open(output_csv, mode='w') as out_csv:

        fieldnames = ['id', 'label']

        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)

        writer.writeheader()



        with torch.no_grad():

            for data, ids in test_loader:

                data = data.to(device)

                output = model(data)

                preds = output.argmax(dim=1).detach().cpu().numpy()

                for id, pred in zip(ids, preds):

                    writer.writerow({'id': id.item(), 'label': pred})



    print(f'\n Finished predicting labels on test dataset. Wrote output csv at {Config.output_csv}')
train_set = CsvImages(Config.train_csv, transform=Config.train_transforms)

train_loader = DataLoader(train_set, batch_size=Config.batch_size,

                          shuffle=True, num_workers=4)



valid_set = CsvImages(Config.valid_csv, transform=Config.valid_transforms)

valid_loader = DataLoader(valid_set, batch_size=Config.batch_size,

                          shuffle=True, num_workers=4)



test_set = CsvImages(Config.test_csv, has_labels=False, transform=Config.valid_transforms)

test_loader = DataLoader(test_set, batch_size=Config.batch_size,

                         shuffle=True, num_workers=4)



model = Config.model(pretrained=False, progress=True, num_classes=10)

model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

model.classifier = nn.Sequential(

    nn.Dropout(p=0.2),

    nn.Linear(1280, 256),

    nn.Linear(256, 10)

)

model = model.to(Config.device)

optimizer = RAdam(model.parameters())

print(model)
for epoch in range(1, Config.epochs + 1):

        train(model, Config.criterion, Config.device, train_loader, optimizer, epoch)

        valid(model, Config.criterion, Config.device, valid_loader)
load(model)

test(model, Config.device, test_loader, Config.output_csv)