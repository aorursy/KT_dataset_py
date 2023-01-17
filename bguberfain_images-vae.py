import numpy as np
import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
%matplotlib inline
batch_size = 64
epochs = 50
num_base_filters = 16
ndims = 2
fimages_train = glob.glob('../input/train_images/*.png')
fimages_test = glob.glob('../input/test_images/*.png')
fimages_unlabel = glob.glob('../input/unlabeled_images/*.png')
len(fimages_train), len(fimages_test), len(fimages_unlabel)
class ImageDataset(Dataset):
    def __init__(self, fimages):
        super(ImageDataset, self).__init__()
        self.fimages = fimages
        
    def __len__(self):
        return len(self.fimages)
        
    def __getitem__(self, index):
        im = io.imread(self.fimages[index]).astype(np.float32)
        im /= 255

        return im
dataset = ImageDataset(fimages_train + fimages_test + fimages_unlabel)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# class VAE(nn.Module):
#     def __init__(self, ndims):
#         super(VAE, self).__init__()
        
#         self.cnn_encoder = nn.Sequential(
#             nn.Conv2d(3, 8, 4, stride=2, padding=1),
#             nn.ReLU(),

#             nn.Conv2d(8, 16, 4, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),

#             nn.Conv2d(16, 32, 4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.Conv2d(32, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 128, 4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.Conv2d(128, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
        
#         self.cnn_decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),

#             nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),

#             nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
# #             nn.BatchNorm2d(3),
#             nn.Sigmoid()
#         )
        
#         self.cnn_decode_max = nn.MaxPool2d(3, return_indices=True)
#         self.cnn_encode_unpool = nn.MaxUnpool2d(3)

#         self.fc_mu = nn.Linear(256, ndims)
#         self.fc_logvar = nn.Linear(256, ndims)
#         self.fc_decode = nn.Linear(ndims, 256)

#     def encode(self, x, only_mean=True):
#         # CNN encode
#         cnn_out = self.cnn_encoder(x)
        
#         # Global max pooling
#         h1, h1_max = self.cnn_decode_max(cnn_out)
        
#         # Prepare for fully conencted
#         h1 = h1.view(-1, 256)
        
#         mu = self.fc_mu(h1)
        
#         if only_mean:
#             return mu
#         else:
#             return mu, self.fc_logvar(h1), h1_max

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5*logvar)
#             eps = Variable(torch.randn(*tuple(std.size()))).cuda()
#             return eps.mul(std).add_(mu)
#         else:
#             return mu

#     def decode(self, z, h1_max):
#         h2 = F.relu(self.fc_decode(z)).view(-1, 256, 1, 1)
        
#         cnn_in = self.cnn_encode_unpool(h2, h1_max)
        
#         return self.cnn_decoder(cnn_in)

#     def forward(self, x):
#         mu, logvar, h1_max = self.encode(x, only_mean=False)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z, h1_max), mu, logvar
def num_filters(idx): return num_base_filters * 2 ** idx
list(map(num_filters, [0, 1, 2, 3, 4, 5]))
class VAE(nn.Module):
    def __init__(self, ndims):
        super(VAE, self).__init__()
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, num_filters(0), 4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(num_filters(0), num_filters(1), 4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters(1)),
            nn.ReLU(),

            nn.Conv2d(num_filters(1), num_filters(2), 4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters(2)),
            nn.ReLU(),

            nn.Conv2d(num_filters(2), num_filters(3), 4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters(3)),
            nn.ReLU(),

            nn.Conv2d(num_filters(3), num_filters(4), 4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters(4)),
            nn.ReLU(),

            nn.Conv2d(num_filters(4), num_filters(5), 3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters(5)),
            nn.ReLU()
        )
        
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters(5), num_filters(4), 6, stride=1, padding=1),
            nn.BatchNorm2d(num_filters(4)),
            nn.ReLU(),
            
            nn.ConvTranspose2d(num_filters(4), num_filters(3), 4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters(3)),
            nn.ReLU(),

            nn.ConvTranspose2d(num_filters(3), num_filters(2), 4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters(2)),
            nn.ReLU(),

            nn.ConvTranspose2d(num_filters(2), num_filters(1), 4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters(1)),
            nn.ReLU(),

            nn.ConvTranspose2d(num_filters(1), 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.fc_mu = nn.Linear(num_filters(5) * 3 * 3, ndims)
        self.fc_logvar = nn.Linear(num_filters(5) * 3 * 3, ndims)
        self.fc_decode = nn.Linear(ndims, num_filters(5) * 3 * 3)

    def encode(self, x, only_mean=True):
        # CNN encode
        cnn_out = self.cnn_encoder(x)
        
        # Flatten for dense layers
        h1 = cnn_out.view(-1, num_filters(5) * 3 * 3)
        
        mu = self.fc_mu(h1)
        
        if only_mean:
            return mu
        else:
            return mu, self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = Variable(torch.randn(*tuple(std.size()))).cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        cnn_in = F.relu(self.fc_decode(z)).view(-1, num_filters(5), 3, 3)
        
        return self.cnn_decoder(cnn_in)

    def forward(self, x):
        mu, logvar = self.encode(x, only_mean=False)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
def exp_lr_scheduler(optimizer, epoch, init_lr=1e-2, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
#     BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    MSE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE, KLD
model = VAE(ndims).cuda()
optimizer = optim.Adam(model.parameters())
model
for epoch in range(epochs):
    print('Epoch: {} of {}'.format(epoch+1, epochs))
    optimizer = exp_lr_scheduler(optimizer, epoch, init_lr=5e-3, lr_decay_epoch=20)
    model.train()
    total_samples = 0
    total_loss = 0
    with tqdm(total=len(dataloader.dataset), file=sys.stdout) as bar:
        for batch_idx, data in enumerate(dataloader):
            data = Variable(data).permute(0, 3, 1, 2).cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss_mse, loss_kld = loss_function(recon_batch, data, mu, logvar)
            loss = loss_mse + 0.1 * loss_kld
            loss.backward()
            total_loss += loss.cpu().data[0]
            optimizer.step()
            
            total_samples += data.shape[0]
            
            bar.update(data.shape[0])
            bar.set_description('loss: {:7.2f} {:.5f} {:.5f}'.format(total_loss/total_samples, loss_mse.cpu().data[0], loss_kld.cpu().data[0]))
            
            del data, recon_batch, mu, logvar, loss
# Sample some data
data = next(iter(dataloader))
data_x = Variable(data).permute(0, 3, 1, 2).cuda()
# Forward pass
model.eval()
recon_batch, mu, logvar = model(data_x)
data_encode = model.encode(data_x).cpu().data.numpy()
recon_batch = recon_batch.permute(0, 2, 3, 1).cpu().data.numpy()
if data_encode.shape[1] == 2:
    plt.scatter(*data_encode.T);
im = 24
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(data[im])
ax[1].imshow(recon_batch[im]);

