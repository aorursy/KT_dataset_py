# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import torch.nn as nn

import torch.nn.functional as F





class VAE(nn.Module):

    def __init__(self, dim):

        super().__init__()

        

        self.dim=dim

        

        self.encoder = nn.Sequential(

            nn.Linear(self.dim, 200),

            nn.ReLU(True),

            nn.Linear(200, 4)

        )



        self.decoder = nn.Sequential(

            nn.Linear(2, 200),

            nn.ReLU(True),

            nn.Linear(200, self.dim),

            nn.Sigmoid()

        )



    def forward(self, x):

        # Run Encoder

        mu, log_var = self.encoder(x).chunk(2, dim=1)



        # Re-parametrize

        sigma = (log_var * .5).exp()

        z = mu + sigma * torch.randn_like(sigma)

        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        kl_div = kl_div / x.size(0)  # mean over batch



        # Run Decoder

        x_prime = self.decoder(z)

        return x_prime, kl_div
xr = pd.read_csv('/kaggle/input/lish-moa/train_features.csv', index_col='sig_id')



num_cols = xr.columns[xr.columns.str.contains('-')]



model = VAE(len(num_cols)).cuda()



model(torch.randn(100, len(num_cols)).cuda())
ttxr = torch.tensor(xr[num_cols].values).float().cuda()
x_, kl = model(ttxr)
optim = torch.optim.AdamW(model.parameters(), lr=0.001)
ttxr


#loss = F.binary_cross_entropy(x_,ttxr) + kl
F.mse_loss(x_,ttxr), kl
for i in range(20000):

    x_, kl = model(ttxr)

    loss = F.binary_cross_entropy(x_,(ttxr)) + kl

    #loss = F.mse_loss(x_,torch.exp(ttxr)) + kl#*10

    optim.zero_grad()

    loss.backward()

    optim.step()

    

    if i%200==0:

        print(loss)
with torch.no_grad():

    zr_ = model.encoder(ttxr)
yr = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')



yrm = yr.melt('sig_id')

yrm = yrm[yrm.value!=0].set_index('sig_id')



xr['x'] = zr_[:,0].cpu()

xr['y'] = zr_[:,1].cpu()



yrm = xr.join(yrm)



yrm['variable'] = yrm.variable.fillna('None')



import matplotlib.cm as cm



yrm['colors'] = yrm.variable.apply(lambda x:cm.rainbow(hash(x)%256))







yrm.variable
from matplotlib import pyplot as plt



plt.figure(figsize=(20,10))



plt.scatter(yrm.x, yrm.y, s=10, c=yrm.colors, alpha=1)



xe = pd.read_csv('/kaggle/input/lish-moa/test_features.csv', index_col='sig_id')

with torch.no_grad():

    ttxe = torch.tensor(xe[num_cols].values).float().cuda()

    ze_ = model.encoder(ttxe)

    

xed = xe.copy()

xed['x'] = ze_[:,0].cpu()

xed['y'] = ze_[:,1].cpu()
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)



neighbors.fit(yrm[['x','y']], yrm.values)



_,ind = neighbors.kneighbors(xed[['x','y']], n_neighbors=1)



xed['colors'] = yrm.iloc[ind.squeeze()]['colors'].values





plt.figure(figsize=(20,10))

plt.scatter(xed.x, xed.y, s=10, c=xed.colors)
from collections import Counter



_,ind = neighbors.kneighbors(xed[['x','y']], n_neighbors=20)



labels = [[yrm.iloc[i].variable for i in ii]  for ii in ind.squeeze()] 



submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv', index_col='sig_id')

submission[:] = 0

for label, (i, row) in zip(labels, submission.iterrows()):

    

    var, count = Counter(label).most_common()[0]

    if count <= 4:

        continue

    if var == 'None':

        continue

    submission.loc[i,var] = 1.0

    

    print(var, count)
submission.to_csv('submission.csv')
