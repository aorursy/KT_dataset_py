import torch

import numpy as np

import pandas as pd

import os



import torch.nn.functional as F

import torch.optim as optim

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

import torchvision.utils as utils

import matplotlib.pyplot as plt

from PIL import Image

import pickle

%matplotlib inline
TRAIN_DIR = '../input/humpback-whale-identification/train'

TEST_DIR = '../input/humpback-whale-identification/test'
os.listdir('../input/data-preprocessing-and-serialization')
train_file = '../input/data-preprocessing-and-serialization/processed_data'

test_file = '../input/data-preprocessing-and-serialization/test_data'

val_file = '../input/validation-data/val_data'
class W_dataset(Dataset):

    def __init__(self, data_file, transform=None):

        # data_file: handle to the preprocessed data file

        self.data_file = data_file

        self.transform = transform

        self.train_dict = pickle.load(self.data_file)

        

    def __len__(self):

        return len(self.train_dict['labels'])

    def __getitem__(self, idx):

        img_np = self.train_dict['data'][idx]

        label = self.train_dict['labels'][idx]

        x = self.transform(img_np)

        return (x, label)
transform = T.Compose([

            T.ToTensor(),

            T.Normalize((0.51401635, 0.55264414, 0.59649817), (0.26610398, 0.2555096,  0.25559797))

])
f = open(train_file, 'rb')

train_dset = W_dataset(f, transform=transform)

train_loader = DataLoader(train_dset, batch_size=64, shuffle=True)

f.close()

print(len(train_dset))
max(train_dset.train_dict['labels'])
for t, (x,y) in enumerate(train_loader):

    print(t, x.size(), y.size())

    if t>2:

        break
def display_batch(batch):

    grid = utils.make_grid(batch, nrow=4)

    plt.figure(figsize=(10,10))

    plt.imshow(grid.numpy().transpose(1,2,0))

for x, label in train_loader:

    display_batch(x)

    break
USE_GPU = True



dtype = torch.float32 # we will be using float throughout this tutorial



if USE_GPU and torch.cuda.is_available():

    device = torch.device('cuda')

else:

    device = torch.device('cpu')



# Constant to control how frequently we print train loss

print_every = 100



print('using device:', device)



def flatten(x):

    N = x.shape[0] # read in N, C, H, W

    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image



def test_flatten():

    x = torch.arange(12).view(2, 1, 3, 2)

    print('Before flattening: ', x)

    print('After flattening: ', flatten(x))



test_flatten()



# We need to wrap `flatten` function in a module in order to stack it

# in nn.Sequential

class Flatten(nn.Module):

    def forward(self, x):

        return flatten(x)

def check_accuracy_part34(loader, model):

    print("Checking accuracy on training set: ")  

    num_correct = 0

    num_samples = 0

    model.eval()  # set model to evaluation mode

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU

            y = y.to(device=device, dtype=torch.long)

            scores = model(x)

            _, preds = scores.max(1)

            num_correct += (preds == y).sum()

            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples

        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
def train_part34(model, optimizer, loader, epochs=1):

    """

    Inputs:

    - model: A PyTorch Module giving the model to train.

    - optimizer: An Optimizer object we will use to train the model

    - epochs: (Optional) A Python integer giving the number of epochs to train for

    

    Returns: Nothing, but prints model accuracies during training.

    """

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.80, last_epoch=-1)

    

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    for e in range(1, epochs+1):

        loss_history = []

        t_history = []

        print('Start of epoch: ', e)

        for t, (x, y) in enumerate(loader):

            model.train()  # put model to training mode

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU

            y = y.to(device=device, dtype=torch.long)



            scores = model(x)

            loss = F.cross_entropy(scores, y)



            # Zero out all of the gradients for the variables which the optimizer

            # will update.

            optimizer.zero_grad()



            # This is the backwards pass: compute the gradient of the loss with

            # respect to each  parameter of the model.

            loss.backward()

            if t%50 == 0:

                loss_history.append(float(loss))

                t_history.append(t)

            

            # Actually update the parameters of the model using the gradients

            # computed by the backwards pass.

            optimizer.step()



            if t % print_every == 0:

                print('Iteration %d, loss = %.4f' % (t, loss.item()))

                print()

                #print(loss_history)

                #print(t_history)

        if e==epochs:

            check_accuracy_part34(loader, model)

        plt.plot(t_history, loss_history, 'o-')

        plt.show()

        for param_group in optimizer.param_groups:

             print(param_group['lr'])

             break

        scheduler.step()

        for param_group in optimizer.param_groups:

             print(param_group['lr'])

             break
model = None

optimizer = None

channel_1 = 16

channel_2 = 16

channel_3 = 128

num_units_1 = 8192

num_units_2 = 128

bias = False

num_classes = 5005

dropout_prob = 0.0



model = nn.Sequential(

        # Conv_1

            nn.Conv2d(3, channel_1, (5,5), padding=2, bias=bias),

            nn.ReLU(),

            nn.BatchNorm2d(num_features= channel_1),

            nn.MaxPool2d(kernel_size=(2,2), stride=2),

            nn.Dropout2d(p=dropout_prob),

        # Conv_2

            nn.Conv2d(channel_1, channel_2, (3,3), padding=1, bias=bias),

            nn.ReLU(),

            nn.BatchNorm2d(num_features= channel_2),

            nn.MaxPool2d(kernel_size=(2,2), stride=2),

            nn.Dropout2d(p=dropout_prob),

            Flatten(),

        # Linear_1

            nn.Linear(channel_2*16*32, num_units_1, bias=bias),

            nn.ReLU(),

            nn.BatchNorm1d(num_features= num_units_1),

        # output

            nn.Linear(num_units_1, num_classes, bias=True)

)



optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

train_part34(model, optimizer, train_loader, epochs=6)
# find out threshold for new_whale class

'''def select_threshold(loader, model, thresh):

    print("Finding the best threshold value: ")  

    num_correct = 0

    num_samples = 0

    model.eval()  # set model to evaluation mode

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU

            y = y.to(device=device, dtype=torch.long)

            scores = model(x)

            probs = F.softmax(scores, dim=1)

            max_value, preds = probs.max(1)

            preds[max_value <= thresh] = 0

            num_correct += (preds == y).sum()

            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples

        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    return acc

thresh_values = np.linspace(0.2, 0.8)

best_acc = -1

for thresh in thresh_values:

    acc = select_threshold(val_loader, model, thresh)

    print('acc = ', acc)

    print('thresh = ', thresh)

    print()

    if acc > best_acc:

        best_acc = acc

        best_thresh = thresh

print('Best threshold value found to be: ', best_thresh)'''
# save the model

torch.save(model.state_dict(), 'model_state_dict')
class Test_dataset(Dataset):

    def __init__(self, data_file, transform=None):

        # data_file: handle to the preprocessed data file

        self.data_file = data_file

        self.transform = transform

        self.test_dict = pickle.load(self.data_file)

        

    def __len__(self):

        return len(self.test_dict['data'])

    def __getitem__(self, idx):

        img_np = self.test_dict['data'][idx]

        img_name = self.test_dict['img_names'][idx]

        x = self.transform(img_np)

        return (x, img_name)

    

f = open(test_file, 'rb')

test_dset = Test_dataset(f, transform=transform)

test_loader = DataLoader(test_dset, batch_size=64, shuffle=False)

f.close()
for t, (x,y) in enumerate(test_loader):

    print(t, x.size(), len(y))

    if t>2:

        break
def gen_test_csv(model, loader):

    thresh = 1.0

    columns = ['Image', 'Id']

    test_df = pd.DataFrame(columns=columns)

    # map whale ids to labels

    train_df = pd.read_csv('../input/humpback-whale-identification/train.csv')

    IDs = list(train_df.Id.unique())

    IDs.sort() 

    print('Number of whale ids = ', len(IDs))

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    model.eval()

    with torch.no_grad():

        for t, (x, img_name) in enumerate(loader):

            x = x.to(device=device, dtype=dtype)

            scores = model(x)

            #scores = scores.cpu().numpy()

            # argsort sorts in ascending order. So, we take the last five elements of each row.

            # Calculate softmax

            probs = F.softmax(scores, dim=1)

            probs, indices = torch.topk(probs, 5, dim=1)  # top 5 probabilities

            #whale_classes = np.argsort(scores, axis=1)[:,-5:]

            whale_Ids = []

            for i in range(scores.shape[0]):

                whale_Ids.append([])

                new_whale = False    # has new whale been put in

                for j in range(5):

                    #if j==0:

                        #whale_Ids[i].append('new_whale')

                    if probs[i, j] < thresh and not new_whale:

                        label = 0

                        new_whale = True

                    else:

                        label = indices[i, j]

                    Id = IDs[label]

                    whale_Ids[i].append(Id)

            for i in range(len(whale_Ids)):

                whale_Ids[i] = '\n'.join(whale_Ids[i])

            whale_imgs = list(img_name)

            pred = {'Image':whale_imgs, 'Id':whale_Ids}

            test_df = test_df.append(pd.DataFrame(pred), ignore_index=True, sort=False)

    test_df.to_csv('submission.csv', index=False)

    return test_df



test_df = gen_test_csv(model, test_loader)

            
test_df