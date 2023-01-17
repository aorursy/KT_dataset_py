import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

import os.path

from tqdm import tqdm_notebook as tqdm
img_limit = 20000



id2names = {}

name2id = {}

with open('../input/celeba-cropped/identity_CelebA.txt') as f:

    line = f.readline()

    cnt = 0

    while(line):

        name, id = line.split()

        name2id[name] = id

        if id not in id2names:

            id2names[id] = [name]

        else:

            id2names[id].append(name)

        cnt+=1

        if cnt>=img_limit:

            break

        line = f.readline()
# person = list(id2names['2937'])   # 2880

# plt.figure(figsize=(20, 5*len(person)))    

# for i,fname in enumerate(person):

#     im = Image.open('../input/celeba-dataset/img_align_celeba/img_align_celeba/'+fname)

#     plt.subplot(len(person),1,i+1)

#     plt.imshow(im)

# plt.show()
mean_embs = np.zeros((len(id2names.keys()),512), dtype=float)

idx = 0

for k in id2names.keys():

    count = 0

    mean_emb = np.zeros((1,512))

    for fname in id2names[k]:

        if os.path.exists('../input/celeba-cropped/emb/emb/'+fname.replace('.jpg','.npy')):

            mean_emb += np.load('../input/celeba-cropped/emb/emb/'+fname.replace('.jpg','.npy'))

            count+=1

    mean_embs[idx,:] = mean_emb / count

    idx+=1

print(idx)
mean_embs[1,:]

#list(id2names.keys())[1]
person = list(id2names['2937'])   # 2880



num_emb = 0

mean_emb = np.zeros((1,512))

for fname in person:

    if os.path.exists('../input/celeba-cropped/emb/emb/'+fname.replace('.jpg','.npy')):

        emb = np.load('../input/celeba-cropped/emb/emb/'+fname.replace('.jpg','.npy'))

        num_emb+=1

        mean_emb+=emb

print(num_emb)

print(mean_emb/num_emb)
# mean_emb0 = []

# for fname in person:

#     if os.path.exists('../input/celeba-cropped/mean_emb/mean_emb/'+fname.replace('.jpg','.npy')):

#         mean_emb0 = np.load('../input/celeba-cropped/mean_emb/mean_emb/'+fname.replace('.jpg','.npy'))

#         break
import torch.optim as optim

import torch.utils.data

import torchvision

import torchvision.datasets as dset

import torchvision.transforms as transforms

import torchvision.utils as vutils

from torch.autograd import Variable

from torch import nn, optim

import torch.nn.functional as F

from torchvision import datasets, transforms

from torchvision.utils import save_image

import matplotlib.image as mpimg

import torchvision

import torchvision.datasets as dset

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

import torchvision.utils as vutils

from torch.autograd import Variable

import random
def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()
import json

with open('../input/celeba-cropped/bboxes_celeba.json') as file:  

    bboxes_json = json.load(file)
idx = 666

key = list(bboxes_json.keys())[idx]

print(key)

box = [round(x) for x in bboxes_json[key]]

plt.imshow(Image.open('../input/celeba-dataset/img_align_celeba/img_align_celeba/'+key).crop(box))
SIZE = 128

folder_w_imgs = '../input/celeba-dataset/img_align_celeba/img_align_celeba/'



class faces_with_mean_emb(Dataset):

    def __init__(self, imgs, mean_embs, bboxes_json, ident_list, name2id, transform=None):

        self.imgs = imgs

        self.transform = transform

        self.bboxes = bboxes_json

        self.cropped = self.get_cropped()

        self.mean_embs = mean_embs

        self.ident = ident_list

        self.name2id = name2id

            

    def __len__(self):

        return len(self.imgs)

    

    def get_cropped(self):

        

        required_transforms = torchvision.transforms.Compose([

                torchvision.transforms.Resize(SIZE),

                torchvision.transforms.CenterCrop(SIZE),

        ])

        

        cropped_list = []

        for imname in self.imgs:

            im = Image.open(folder_w_imgs + imname)

            if imname in self.bboxes.keys():

                box = [round(x) for x in self.bboxes[imname]]

                im = im.crop(box)

            im_t = required_transforms(im)

            cropped_list.append(im_t)

                

        return cropped_list

    

    def __getitem__(self, idx):

        im = self.cropped[idx]

        if self.transform:

            im = self.transform(im)

        emb = self.mean_embs[ident_list.index(name2id[imgs[idx]])]

        return np.asarray(im), torch.from_numpy(emb).float()
imgs = list(name2id.keys())

ident_list = list(id2names.keys())



batch_size = 32



transform = transforms.Compose([#transforms.RandomHorizontalFlip(p=0.1),

                                #transforms.RandomApply(random_transforms, p=0),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



train_data = faces_with_mean_emb(imgs, mean_embs, bboxes_json, ident_list, name2id, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=False,

                                           batch_size=batch_size, num_workers=4)

real_batch = next(iter(train_loader))

plt.figure(figsize=(8,8))

plt.axis("off")

plt.title("Training Images")

plt.imshow(np.transpose(vutils.make_grid(real_batch[0], padding=2, normalize=True).cpu(),(1,2,0)))
class PixelwiseNorm(nn.Module):

    def __init__(self):

        super(PixelwiseNorm, self).__init__()



    def forward(self, x, alpha=1e-8):

        """

        forward pass of the module

        :param x: input activations volume

        :param alpha: small number for numerical stability

        :return: y => pixel normalized activations

        """

        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]

        y = x / y  # normalize the input x volume

        return y
from torch.nn.utils import spectral_norm



class Encoder(nn.Module):

    def __init__(self, latent_dim=50, channels=3):

        super(Encoder, self).__init__()

        self.channels = channels

        self.latent_dim = latent_dim

    

        def convlayer_enc(n_input, n_output, k_size=4, stride=2, padding=1, bn=False, add=False):

            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]

            if bn:

                block.append(nn.BatchNorm2d(n_output))

            block.append(nn.LeakyReLU(0.2, inplace=True))

            if add:

                block.append(nn.Conv2d(n_output, n_output, kernel_size=3, stride=1, padding=1, bias=False)) # add depth

                block.append(nn.LeakyReLU(0.2, inplace=True))

            return block

    

        self.encoder = nn.Sequential(

                *convlayer_enc(self.channels, 32, 4, 2, 1),                # (32,64,64)

                *convlayer_enc(32, 64, 4, 2, 1, add=True),                             #(64, 32, 32)

                *convlayer_enc(64, 128, 4, 2, 1, add=True),                         # (128, 16, 16)

                *convlayer_enc(128, 256, 4, 2, 1, bn=True, add=True),               # (256, 8, 8)

                *convlayer_enc(256, 512, 4, 2, 1, bn=True),               # (512, 4, 4)

                #nn.Conv2d(512, self.latent_dim, 4, 1, 1, bias=False),   # (latent_dim, 4, 4)

                #nn.LeakyReLU(0.2, inplace=True)

            )

        self.linear_layer = nn.Linear(512*4*4,self.latent_dim)

        

    def forward(self, x):

        x = self.encoder(x)

        x = self.linear_layer(x.view(x.size(0),-1))

        return x

    

class Decoder(nn.Module):

    def __init__(self, nz, nchannels, nfeats):

        super(Decoder, self).__init__()



        # input is Z, going into a convolution

        #self.conv1 = spectral_norm(nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False))

        self.conv1 = nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False)

        #self.bn1 = nn.BatchNorm2d(nfeats * 8)

        # state size. (nfeats*8) x 4 x 4

        

        #self.conv2 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False))

        self.conv2 = nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False)

        #self.bn2 = nn.BatchNorm2d(nfeats * 8)

        # state size. (nfeats*8) x 8 x 8

        

        #self.conv3 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False))

        self.conv3 = nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False)

        #self.bn3 = nn.BatchNorm2d(nfeats * 4)

        # state size. (nfeats*4) x 16 x 16

        

        #self.conv4 = spectral_norm(nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False))

        self.conv4 = nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False)

        #self.bn4 = nn.BatchNorm2d(nfeats * 2)

        # state size. (nfeats * 2) x 32 x 32

        

        #self.conv5 = spectral_norm(nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False))

        self.conv5 = nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False)

        #self.bn5 = nn.BatchNorm2d(nfeats)

        # state size. (nfeats) x 64 x 64

        

        self.conv6 = nn.ConvTranspose2d(nfeats, nfeats, 4, 2, 1, bias=False)

        # nf x 128 x 128

        self.conv7 = spectral_norm(nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False))

        # state size. (nchannels) x 64 x 64

        self.pixnorm = PixelwiseNorm()

        self.add_depth3 = nn.Conv2d(nfeats*4, nfeats*4, kernel_size=3, stride=1, padding=1, bias=False)

        self.add_depth4 = nn.Conv2d(nfeats*2, nfeats*2, kernel_size=3, stride=1, padding=1, bias=False)

        self.add_depth5 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)

    

    

        

    def forward(self, x):

#         x = F.leaky_relu(self.bn1(self.conv1(x)))

#         x = F.leaky_relu(self.bn2(self.conv2(x)))

#         x = F.leaky_relu(self.bn3(self.conv3(x)))

#         x = F.leaky_relu(self.bn4(self.conv4(x)))

#         x = F.leaky_relu(self.bn5(self.conv5(x)))

        x = F.leaky_relu(self.conv1(x))

        x = F.leaky_relu(self.conv2(x))

        #x = self.pixnorm(x)

        x = F.leaky_relu(self.conv3(x))

        x = F.leaky_relu(self.add_depth3(x))

        #x = self.pixnorm(x)

        x = F.leaky_relu(self.conv4(x))

        x = F.leaky_relu(self.add_depth4(x))

        #x = self.pixnorm(x)

        x = F.leaky_relu(self.conv5(x))

        x = F.leaky_relu(self.add_depth5(x))

        

        x = F.leaky_relu(self.conv6(x))

        x = self.pixnorm(x)

        x = torch.tanh(self.conv7(x))

        return x
latent_dim = 20

channels= 3

emb_size = 512

beta1 = 0.5



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





enc = Encoder(latent_dim, channels).to(device)

dec = Decoder(latent_dim+emb_size, channels, 32).to(device)

criterion = nn.MSELoss()

#optim_enc = optim.Adam(enc.parameters(), lr=0.0002, betas=(beta1, 0.999))

optim_enc = optim.Adam(enc.parameters())

#optim_dec = optim.Adam(dec.parameters(), lr=0.0002, betas=(beta1, 0.999))

optim_dec = optim.Adam(dec.parameters())



epochs = 300

steps = 0

for epoch in range(epochs):

    for ii, (x,emb) in tqdm(enumerate(train_loader), total=len(train_loader)):

        enc.zero_grad()

        dec.zero_grad()



        x = x.to(device)

        emb = emb.to(device)

        z = enc(x)

        z = torch.cat((z,emb), dim=1)

        z = z.unsqueeze(2).unsqueeze(3)

        x_hat = dec(z)



        loss = criterion(x_hat, x)

        loss.backward()

        optim_enc.step()

        optim_dec.step()

        

        steps+=1

        

        if steps%1000==0: print('epoch: %d, loss: %.5f' % (epoch, loss.item()))
torch.save(enc.state_dict(),'enc1.pth')

torch.save(dec.state_dict(),'dec1.pth')
optim_enc = optim.Adam(enc.parameters(), lr=0.0005, betas=(beta1, 0.999))

optim_dec = optim.Adam(dec.parameters(), lr=0.0005, betas=(beta1, 0.999))

epochs = 300

steps = 0

for epoch in range(epochs):

    for ii, (x,emb) in tqdm(enumerate(train_loader), total=len(train_loader)):

        enc.zero_grad()

        dec.zero_grad()



        x = x.to(device)

        emb = emb.to(device)

        z = enc(x)

        z = torch.cat((z,emb), dim=1)

        z = z.unsqueeze(2).unsqueeze(3)

        x_hat = dec(z)



        loss = criterion(x_hat, x)

        loss.backward()

        optim_enc.step()

        optim_dec.step()

        

        steps+=1

        

        if steps%1000==0: print('epoch: %d, loss: %.5f' % (epoch, loss.item()))
torch.save(enc.state_dict(),'enc2.pth')

torch.save(dec.state_dict(),'dec2.pth')
optim_enc = optim.Adam(enc.parameters(), lr=0.0001, betas=(beta1, 0.999))

optim_dec = optim.Adam(dec.parameters(), lr=0.0001, betas=(beta1, 0.999))

epochs = 300

steps = 0

for epoch in range(epochs):

    for ii, (x,emb) in tqdm(enumerate(train_loader), total=len(train_loader)):

        enc.zero_grad()

        dec.zero_grad()



        x = x.to(device)

        emb = emb.to(device)

        z = enc(x)

        z = torch.cat((z,emb), dim=1)

        z = z.unsqueeze(2).unsqueeze(3)

        x_hat = dec(z)



        loss = criterion(x_hat, x)

        loss.backward()

        optim_enc.step()

        optim_dec.step()

        

        steps+=1

        

        if steps%1000==0: print('epoch: %d, loss: %.5f' % (epoch, loss.item()))
torch.save(enc.state_dict(),'enc3.pth')

torch.save(dec.state_dict(),'dec3.pth')
it = iter(train_loader)

real_batch = next(it)

x1 = real_batch[0][10].unsqueeze(0)

x2 = real_batch[0][11].unsqueeze(0)

emb1 = real_batch[1][11].unsqueeze(0)

emb2 = real_batch[1][10].unsqueeze(0)





x1 = x1.to(device)

emb1 = emb1.to(device)

z = enc(x1)



z = torch.cat((z,emb1), dim=1)

z = z.unsqueeze(2).unsqueeze(3)

x_hat1 = dec(z).to("cpu").clone().detach().squeeze(0)

x1 = x1.squeeze(0).to("cpu")



x2 = x2.to(device)

emb2 = emb2.to(device)

z = enc(x2)



z = torch.cat((z,emb2), dim=1)

z = z.unsqueeze(2).unsqueeze(3)

x_hat2 = dec(z).to("cpu").clone().detach().squeeze(0)

x2 = x2.squeeze(0).to("cpu")



print(x1.shape,x2.shape,x_hat1.shape, x_hat2.shape)



plt.imshow(np.transpose(x1,(1,2,0)))

plt.show()

plt.imshow(np.transpose(x_hat1,(1,2,0)))

plt.show()

plt.imshow(np.transpose(x2,(1,2,0)))

plt.show()

plt.imshow(np.transpose(x_hat2,(1,2,0)))

plt.show()
