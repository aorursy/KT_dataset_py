import os

import time

import math

import random

import numpy as np 

import pandas as pd 

from collections import Counter



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader



import torchvision

import torchvision.transforms as T



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
### read dataset 

#train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

train_x = train[list(train.columns)[1:]].values

train_y = train['label'].values



## normalize and reshape the predictors  

train_x = train_x / 255



## create train and validation datasets

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)



## reshape the inputs

train_x = train_x.reshape(-1, 784)

val_x = val_x.reshape(-1, 784)



# make dataset 

train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))

val_dataset = TensorDataset(torch.tensor(val_x), torch.tensor(val_y))
X_reduced = TSNE(n_components=2).fit_transform(val_x)

palette = np.array(sns.color_palette("hls", 10))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=palette[val_y])

plt.show()
def my_collate(batch):

    data = [item[0].view(1, 28,28) for item in batch]

    data = torch.stack(data).float()



    target = [item[1] for item in batch]

    target = torch.LongTensor(target)

    return data, target
class convEmbedding(nn.Module):

    def __init__(self, c_dim = 1):

        super().__init__()



        self.embedding = nn.Sequential(

            nn.Conv2d(c_dim, 32, 3),

            nn.ReLU(),

            nn.Conv2d(32, 32, 3),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.2),

            

            nn.Conv2d(32, 64, 3),

            nn.ReLU(),

            nn.Conv2d(64, 64, 3),

            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Dropout(0.2),



            nn.Flatten(),



            nn.Linear(1024, 512),

            nn.ReLU(),

            nn.Dropout(0.2),



            nn.Linear(512, 10)

        )





    def forward(self, input_tensor):

        x = self.embedding(input_tensor)

        return F.normalize(x, p =2, dim = 1)

    

model = convEmbedding()
def get_anchor_positive_mask(labels):

    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    """

    # Check that i and j are distinct

    indices_equal = torch.eye(labels.size(0)).bool()

    indices_not_equal = ~indices_equal



    # Check if labels[i] == labels[j]

    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)



    return labels_equal & indices_not_equal





def get_anchor_negative_mask(labels):

    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    """

    # Check if labels[i] != labels[k]

    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)



    return ~(labels.unsqueeze(0) == labels.unsqueeze(1))



def get_triplet_mask(labels):

    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    """

    mask_anchor_positive = get_anchor_positive_mask(labels)

    mask_anchor_negative = get_anchor_negative_mask(labels)



    return mask_anchor_positive.unsqueeze(2) & mask_anchor_negative.unsqueeze(1)
def batch_triplet_loss(embeddings, labels, hparams):

    

    pairwise_dist = torch.cdist(embeddings, embeddings, p = 1)

    

    if hparams["mode"] == 'batch hard':

        ##  for each anchor, select the hardest positive and the hardest negative among the batch

        ## get the hardest positive

        mask_anchor_positive = get_anchor_positive_mask(labels).float()

        anchor_positive_dist = pairwise_dist * mask_anchor_positive

        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)



        ## get the hardest negative

        mask_anchor_negative = get_anchor_negative_mask(labels).float()

        anchor_negative_dist = pairwise_dist + 999. * (1.0 - mask_anchor_negative)

        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        

        triplet_loss = hardest_positive_dist - hardest_negative_dist + hparams["margin"] ## (batch_size, 1)

        

    elif hparams["mode"] == 'batch all':

        ## select 1) all the valid triplets, and 2) average the loss on the hard and semi-hard triplets

        anchor_positive_dist = pairwise_dist.unsqueeze(2) ## (batch_size, batch_size, 1)

        anchor_negative_dist = pairwise_dist.unsqueeze(1) ## (batch_size, 1, batch_size)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + hparams["margin"] ## broadcasting, all compibations of (a, p, n)

        ## 1) select valid triplets

        valid_mask = get_triplet_mask(labels).float()

        triplet_loss = valid_mask * triplet_loss ## (batch_size, batch_size, batch_size)

    else:

        raise TypeError("invalid mode")

        

    # remove easy triplets

    triplet_loss[triplet_loss < 0] = 0

    return torch.mean(triplet_loss)
def timeSince(since):

    now = time.time()

    s = now - since

    m = math.floor(s / 60)

    s -= m * 60

    return '%dm %ds' % (m, s)



def train_one_batch(model, x, y, hparams):

    x = x.to(device)



    batch_embeddings = model(x)

    loss = batch_triplet_loss(batch_embeddings, y, hparams)

    

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()
hparams = {"learning_rate": 1e-3, "margin": 0.5, "mode": "batch hard", "batch_size":  16, "epochs": 50} 



trainloader = DataLoader(train_dataset, batch_size = hparams["batch_size"], collate_fn = my_collate, drop_last = True)

steps_per_epoch  = train_dataset.__len__()// hparams["batch_size"]



model = convEmbedding().to(device)

optimizer = torch.optim.Adam(model.parameters(), hparams['learning_rate'])



  

start = time.time()

train_losses = []

model.train()

for epoch in range(1, hparams["epochs"]+1):

    print('-' * 10)

    print('Epoch {}/{}\t{} batches'.format(epoch, hparams["epochs"], steps_per_epoch))

    

    curr_loss = []

    for step, (x, y) in enumerate(trainloader):

        loss = train_one_batch(model, x, y, hparams)

        curr_loss.append(loss)

        print('\rprogress {:6.1f} %\tloss {:8.4f}'.format(100*(step+1)/steps_per_epoch, np.mean(curr_loss)), end = "")

        

    train_losses.append(np.mean(curr_loss))

    print('\rprogress {:6.1f} %\tloss {:8.4f}'.format(100*(step+1)/steps_per_epoch, np.mean(curr_loss)))

    print('{}'.format(timeSince(start)))

## Check if we can get at least one pair of anchor-positive using this batch size

#for step, (x, y) in enumerate(trainloader):

#    print(Counter(y.numpy()))

    
model.eval()



val_loader = DataLoader(val_dataset, batch_size = 128, collate_fn = my_collate)
X = []; labels = []

for x, label in val_loader:

    batch_embedings = model(x.to(device))

    X.append(batch_embedings.cpu().detach().numpy())

    labels.append(label.numpy())

    print('\rprogress {:6.1f} %'.format(100*(step+1)/steps_per_epoch), end = "")



X = np.concatenate(X)

labels = np.concatenate(labels)



X_reduced = TSNE(n_components=2).fit_transform(X)

palette = np.array(sns.color_palette("hls", 10))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=palette[labels])

plt.show()