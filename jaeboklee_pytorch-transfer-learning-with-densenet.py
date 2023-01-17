#module import



import os

import numpy as np

import pandas as pd

import torch

import torchvision

import torch.nn as nn

import cv2

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torchvision import transforms, utils

import copy

import tqdm

import sys



%matplotlib inline
# variables and hyperparameters



train_dir = '../input/train_simplified'

test_dir = '../input'

test_file = 'test_simplified.csv'



train_nrows = 2000

val_nrows = 100



train_files = os.listdir(train_dir)

train_files = sorted(train_files)



w2i_dict = {fn[:-4].replace(' ','_'): i for i, fn in enumerate(train_files)}

i2w_dict = {v: k for k, v in w2i_dict.items()}
# Make training / validation Datasets



class DoodleDataset(Dataset):

    def __init__(self, file, dir, mode='train', nrows = None, skiprows=None, transform = None):

        self.file = file

        self.dir = dir

        self.mode = mode

        self.nrows = nrows

        self.transform = transform

        # only use 'nrows' to reduce memory usage / skiprows is used when we make validation set

        self.pd_frame = pd.read_csv(os.path.join(self.dir, self.file), usecols=['drawing'], nrows=nrows, skiprows=skiprows)

        if self.mode == 'train':

            self.label = w2i_dict[file[:-4].replace(' ','_')]

        

    @staticmethod

    # used in __getitem__ function / converts points into line

    def _point2line(strokes):

        # one channel as a grayscale

        img = np.zeros((256,256), dtype = np.uint8)

        for stroke in strokes:

            x_coords = stroke[0]

            y_coords = stroke[1]

            for i in range(len(stroke[0])-1):

                # draw line in color 255

                cv2.line(img, (x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1]), color = 255, thickness = 2)

        return img

    

    def __len__(self):

        return len(self.pd_frame)

    

    def __getitem__(self, idx):

        # string to type variable

        strokes = eval(self.pd_frame.drawing[idx])

        sample = self._point2line(strokes)

        if self.transform:

            sample = self.transform(sample)

        if self.mode=='train':

            # after transform, return images as a numpy array

            sample = sample.numpy()

            return sample.astype('float32'), self.label

        else:

            # add batch size

            sample = sample.numpy()

            return sample.astype('float32')



        

data_transform = transforms.Compose([

    # numpy ndarray to PIL image

    transforms.ToPILImage(),

    transforms.RandomHorizontalFlip(),

    transforms.Resize((128,128)),

    transforms.ToTensor()

])



val_data_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((128,128)),

    transforms.ToTensor()

])





doodles = ConcatDataset([DoodleDataset(fn, train_dir, nrows=train_nrows, transform = data_transform) for fn in train_files])

doodles_val = ConcatDataset([DoodleDataset(fn, train_dir, nrows=val_nrows, transform = val_data_transform, skiprows=range(1, train_nrows+1)) for fn in train_files])
# Make dataloader



dataloader = DataLoader(doodles, batch_size = 16, shuffle = True, num_workers = 4)

valloader = DataLoader(doodles_val, batch_size = 16, shuffle = True, num_workers = 4)
# check if dataset and dataloader are formed well



samples, labels = iter(dataloader).next()

plt.figure(figsize=(16,24))

grid_imgs = torchvision.utils.make_grid(samples[:24])

np_grid_imgs = grid_imgs.numpy()

# in tensor, image is (batch, height, width), so you have to transpose it to (height, width, batch) in numpy to show it.

plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
#score function



def mapk(output, target, k=3):

    with torch.no_grad():

        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i in range(k):

            correct[i] = correct[i]*(k-i)

        score = correct[:k].view(-1).float().sum(0, keepdim=True)

        score.mul_(1.0 / (k * batch_size))

        return score
import math

import torch

from torch.optim.optimizer import Optimizer, required



class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(RAdam, self).__init__(params, defaults)



        # step_buffer stores [step, N_sma, step_size]

        self.step_buffer = [None, None, None]

        self.N_sma_max = 2 / (1-betas[1]) -1



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):

        loss = None if closure is None else closure()



        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue



                # basic variables

                p_data_fp32 = p.data.float()

                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 1

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['step'] += 1

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                beta1_t, beta2_t = beta1 ** state['step'], beta2 ** state['step']

                bias_correction1, bias_correction2 = 1 - beta1_t, 1 - beta2_t



                # get gradients

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')    



                # update exponential average and exponential average squared

                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)



                # get N_sma and store in step_buffer

                if state['step'] == self.step_buffer[0]:

                    N_sma, step_size = self.step_buffer[1], self.step_buffer[2]

                else:

                    N_sma = self.N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    # step size reflects rectification and bias_correction term in advance

                    if N_sma >= 5:

                        rectification = math.sqrt( ((N_sma-4)*(N_sma-2)*self.N_sma_max) / ((self.N_sma_max-4)*(self.N_sma_max-2)*N_sma) )

                        step_size = math.sqrt(bias_correction2) * rectification / bias_correction1

                    else:

                        step_size = 1.0 / bias_correction1

                    

                    self.step_buffer[0] = state['step']

                    self.step_buffer[1] = N_sma

                    self.step_buffer[2] = step_size



                

                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # update parameter

                if N_sma >= 5:            

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)

                else:

                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)



                p.data.copy_(p_data_fp32)



        return loss

# transfer learning with densenet121 structure



device = 'cuda'



# densenet input image is RGB channel color image, so you have to reduce channel

def squeeze_weights(m):

    m.weight.data = m.weight.data.sum(dim=1)[:,None]

    m.in_channels = 1

model = torchvision.models.densenet121(pretrained=True)    



'''

if you have small portion of data, only training classifier is a good way.

for param in model.parameters():

    param.requires_grad = False

'''

num_ftrs = model.classifier.in_features

model.classifier = nn.Linear(num_ftrs, 340)

model.features.conv0.apply(squeeze_weights)





model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = RAdam(model.parameters(), lr=0.005)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30000,60000,90000], gamma=0.3)
# Training procedure



epochs = 3

itr = 1

p_itr = 5000

total_loss, score = 0,0

val_score = 0.0

best_model_wts = copy.deepcopy(model.state_dict())

best_score = 0.0

loss_list = []

score_list = []





for epoch in range(epochs):

    model.train()

    for samples, labels in dataloader:

        samples, labels = samples.to(device), labels.to(device)    

        optimizer.zero_grad()

        output = model(samples)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        score += mapk(output, labels)[0].item()

        scheduler.step()

        

        if itr%p_itr==0:

            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, MAP@3: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, score/p_itr))

            loss_list.append(total_loss/p_itr)

            score_list.append(score/p_itr)

            total_loss, score = 0,0

        itr += 1

        

    # use validation set for finding well updated weights

    model.eval()

    for samples, labels in valloader:

        with torch.no_grad():

            samples = samples.to(device)

            labels = labels.to(device)

            output = model(samples)

            val_score += mapk(output, labels)[0].item()

    print('[Epoch {}/{}] Validation Score: {:.4f}'.format(epoch+1, epochs, val_score/len(valloader)))

    if val_score > best_score:

        best_score = val_score

        best_model_wts = copy.deepcopy(model.state_dict())

    val_score = 0

    

    

# best performed weights are chosen

model.load_state_dict(best_model_wts)
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot(loss_list, label='loss', color = 'g')

ax2.plot(score_list, label='score', color = 'b')



ax1.set_ylabel('loss')

ax2.set_ylabel('score')



plt.title('training loss and score')

plt.show()
filename_pth = 'ckpt_densenet121.pth'

torch.save(model.state_dict(), filename_pth)



test_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((128,128)),

    transforms.ToTensor()

])



testset = DoodleDataset(test_file, test_dir, mode='test', transform = test_transform)

testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)
model.eval()

labels = np.empty((0,3))

for x in tqdm.tqdm_notebook(testloader):

    with torch.no_grad():

        x = x.to(device)

        output = model(x)

        _, pred = output.topk(3,1,True,True)

        labels = np.concatenate([labels, pred.cpu()], axis=0)

    

submission = pd.read_csv('../input/test_simplified.csv', index_col='key_id')

submission.drop(['countrycode','drawing'], axis=1, inplace=True)

submission['word']=''

for i, label in enumerate(labels):

    submission.word.iloc[i] = " ".join([i2w_dict[l] for l in label])

    

submission.to_csv('preds_densenet121.csv')


from IPython.display import FileLinks

FileLinks('.') # input argument is specified folder
