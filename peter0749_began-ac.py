import os
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

if not os.path.exists('./previews'):
    os.makedirs('./previews')
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_class = 10
n_class_remap = 16
batch_size = 64
n_dim = 32
n_feature = 128
n_d_latent = 128
n_ch = 1
g_feature_map_b = 64
d_feature_map_b = 64
'''
fold_dataset = datasets.ImageFolder('./cat_b_128/', 
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomAffine(5, translate=(0.05,0.05), scale=(0.9,1.1), shear=2, resample=2, fillcolor=tuple([127]*n_ch)),
                           transforms.Resize([n_dim]*2, interpolation=2),
                           transforms.ToTensor(), # normalize to [0,1]
                           transforms.Normalize([0.5]*n_ch, [0.5]*n_ch) # [0,1] -> [-1,+1]
                       ]))
'''
fold_dataset = datasets.MNIST('./mnist_data', download=True, train=False, transform=transforms.Compose([
                           transforms.Pad(2), # 28 -> 32
                           transforms.ToTensor(), # normalize to [0,1]
                           transforms.Normalize([0.5]*n_ch, [0.5]*n_ch) # [0,1] -> [-1,+1]
                       ]))

print(fold_dataset.__getitem__(100)[0].shape)
plt.imshow(np.squeeze(np.clip(np.array(fold_dataset.__getitem__(100)[0]).transpose(1,2,0)*127.5+127.5,0,255).astype(np.uint8)))
plt.show()
data_loader = torch.utils.data.DataLoader(
        fold_dataset,
        batch_size=batch_size, shuffle=True, num_workers=4)
print(n_dim, n_feature)
def inf_data_gen():
    while True:
        for data, label in data_loader:
            yield data, label
gen = inf_data_gen()
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)

def one_hot(ids, n_class):
    if len(ids.size())==2:
        return ids
    ohe = torch.FloatTensor(ids.size(0), n_class)
    ids = ids.view(-1,1)
    ohe.zero_()
    ohe.scatter_(1, ids, 1)
    return ohe

class ConvolutioBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True, down=False, relu=True, leaky=False):
        super(ConvolutioBlock, self).__init__()
        
        conv_block = []
        conv_block += [nn.Conv2d(in_ch, out_ch, 3, stride=2 if down else 1, padding=1, bias=False)]
        if norm:
            conv_block += [nn.InstanceNorm2d(out_ch)]
        if relu:
            conv_block += [ nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True) ]

        self.conv_block = nn.Sequential(*conv_block)
        self.conv_block.apply(weights_init)
    def forward(self, x):
        return self.conv_block(x)

# ref SRResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1, bias=False),
                        nn.InstanceNorm2d(in_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1, bias=False),
                        nn.InstanceNorm2d(in_ch)  
                     ]

        self.conv_block = nn.Sequential(*conv_block)
        self.conv_block.apply(weights_init)

    def forward(self, x):
        return x + self.conv_block(x)

class UpConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True, relu=True, leaky=False):
        super(UpConvolution, self).__init__()
        
        conv_block = [nn.PixelShuffle(2)]
        conv_block += [nn.Conv2d(in_ch//4, out_ch, 3, stride=1, padding=1, bias=False)]
        if norm:
            conv_block += [nn.InstanceNorm2d(out_ch)]
        if relu:
            conv_block += [ nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True) ]

        self.conv_block = nn.Sequential(*conv_block)
        self.conv_block.apply(weights_init)

    def forward(self, x):
        return self.conv_block(x)

class C(nn.Module):
    def __init__(self, N_DIM=32, N_CH=3, BASE_FEATURE_N=32, N_CLASS=1, N_BOTTLENECK=128):
        super(C, self).__init__()
        self.n_dim = N_DIM
        self.n_ch = N_CH
        self.n_class = N_CLASS
        self.base_f = BASE_FEATURE_N
        self.n_bottleneck = N_BOTTLENECK
        self.head_conv = nn.Conv2d(self.n_ch, self.base_f, 3, stride=1, padding=1, bias=False)
        weights_init(self.head_conv)
        self.convs1 = nn.Sequential(*[
            ConvolutioBlock( self.base_f, self.base_f,   norm=True, down=True, relu=True, leaky=True ), # /2
            ConvolutioBlock( self.base_f, self.base_f*2, norm=True, down=True, relu=True, leaky=True ), # /4
            ConvolutioBlock( self.base_f*2, self.base_f*4, norm=True, down=True, relu=True, leaky=True ), # /8
            ConvolutioBlock( self.base_f*4, self.base_f*8, norm=True, down=True, relu=True, leaky=True ), # /16
        ])
        self.fc1 = nn.Linear(self.base_f*8*(self.n_dim//16)**2, self.n_bottleneck, bias=False)
        self.fc2 = nn.Linear(self.n_bottleneck, self.base_f*8*(self.n_dim//16)**2, bias=False)
        self.classifier = nn.Linear(self.n_bottleneck, self.n_class, bias=True)
        weights_init(self.fc1)
        weights_init(self.fc2)
        weights_init(self.classifier)
        self.convs2 = nn.Sequential(*[
            UpConvolution(self.base_f*8, self.base_f*4, norm=True, relu=True, leaky=True), # 2x
            UpConvolution(self.base_f*4, self.base_f*2, norm=True, relu=True, leaky=True), # 4x
            UpConvolution(self.base_f*2, self.base_f, norm=True, relu=True, leaky=True), # 8x
            UpConvolution(self.base_f, self.base_f, norm=True, relu=True, leaky=True), # 16x
        ])
        self.tail_conv = nn.Conv2d(self.base_f, self.n_ch, 3, stride=1, padding=1, bias=False)
        weights_init(self.tail_conv)
        
    def forward(self, x, label):
        
        o = x
        
        x = self.head_conv(x)
        x = self.convs1(x)
        x = x.view(x.size(0), self.base_f*8*(self.n_dim//16)**2)
        x = self.fc1(x)
        
        c = self.classifier(x)
        bce_loss = nn.functional.cross_entropy(c, label, reduction='mean')
        
        x = self.fc2(x)
        x = x.view(x.size(0), self.base_f*8, self.n_dim//16, self.n_dim//16)
        x = self.convs2(x)
        
        x = self.tail_conv (x)
        x = torch.tanh(x)
        return torch.abs(x-o).mean() , bce_loss

class G(nn.Module):
    def __init__(self, N_DIM, N_FEATURE, N_CH, BASE_FEATURE_N=32, N_CLASS=1, N_EMB=1):
        super(G, self).__init__()
        self.n_dim = N_DIM
        self.n_ch  = N_CH
        self.n_class = N_CLASS
        self.base_f = BASE_FEATURE_N
        self.n_feature = N_FEATURE
        self.n_emb = N_EMB
        self.aux   = nn.Linear(self.n_class, self.n_emb, bias=False)
        weights_init(self.aux)
        self.latent_map = nn.Linear(self.n_emb+self.n_feature, self.base_f*8*((self.n_dim//32)**2)) 
        weights_init(self.latent_map)
        self.convs = nn.Sequential(*[
            ResidualBlock(self.base_f*8),
            UpConvolution(self.base_f*8, self.base_f*4, norm=True, relu=True), # 2x
            UpConvolution(self.base_f*4, self.base_f*4, norm=True, relu=True), # 4x
            UpConvolution(self.base_f*4, self.base_f*2, norm=True, relu=True), # 8x
            UpConvolution(self.base_f*2, self.base_f*2, norm=True, relu=True), # 16x
            UpConvolution(self.base_f*2, self.base_f, norm=True, relu=True),   # 32x
        ])
        self.tail_conv = nn.Conv2d(self.base_f, self.n_ch, 3, stride=1, padding=1, bias=False)
        weights_init(self.tail_conv)
        
    def forward(self, x, label):
        label_emb = self.aux(label)
        x = torch.cat((x, label_emb), 1)
        x = self.latent_map(x)
        x = x.view(x.size(0), self.base_f*8, self.n_dim//32, self.n_dim//32)
        x = self.convs(x)
        x = self.tail_conv (x)
        x = torch.tanh(x)
        return x
def plot2dir(directory='./previews', imgs=None, iter_n=0):
    imgs = np.clip(np.round((np.concatenate(tuple(imgs.transpose(0,2,3,1)), axis=0)+1)*127.5), 0, 255).astype(np.uint8) # (?, 28, 28)
    cv2.imwrite('{}/{:08d}.jpg'.format(directory, iter_n), np.squeeze(imgs[...,::-1])) # RGB->BGR
    
seed = 3 # debug!!!
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
G_net = G(n_dim, n_feature, n_ch, g_feature_map_b, n_class, n_class_remap).to(device)
C_net = C(n_dim, n_ch, d_feature_map_b, n_class, n_d_latent).to(device)
opt_C = optim.Adam(C_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_G = optim.Adam(G_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
from tqdm import tqdm_notebook

iterations = 5000
preview_iter = 100
# max_preview_imgs = 5
d_iter = 1
std = 1.0
min_k = 0.05
k = torch.FloatTensor([0]).to(device)
lambda_k = torch.FloatTensor([0.001]).to(device) 
lambda_c = torch.FloatTensor([0.1]).to(device) 
gamma = torch.FloatTensor([0.75]).to(device) 

for ite in tqdm_notebook(range(1, iterations+1)):
    start_train_ts = time.time()
    # train D:
    G_net.eval()
    C_net.train()
    d_loss_mean = []
    c_loss_mean = []
    g_loss_mean = 0.0
    for _ in range(d_iter):
        opt_C.zero_grad()
        real, label = next(gen)
        real = real.to(device)
        label_ohe = one_hot(label, n_class).to(device)
        label = label.to(device)
        sample = torch.randn(real.size(0), n_feature, device=device).clamp(-2,2) * std
        fake   = G_net(sample, label_ohe).detach() # not to touch G_net
        d_loss_real, BCE_loss_real = C_net(real, label)
        BCE_loss_real = lambda_c.detach() * BCE_loss_real
        d_loss_fake, not_used = C_net(fake, label)
        d_loss_fake = -k.detach() * d_loss_fake
        d_loss = d_loss_real + d_loss_fake
        (d_loss + BCE_loss_real).backward()
        opt_C.step()
        d_loss_mean.append(d_loss.item())
        c_loss_mean.append(BCE_loss_real.item())
    d_loss_mean = np.mean(d_loss_mean)
    c_loss_mean = np.mean(c_loss_mean)
    D_update_ts = time.time()
    # train G:
    real, label = next(gen)
    real = real.to(device)
    label_ohe = one_hot(label, n_class).to(device)
    label = label.to(device)
    G_net.train()
    C_net.eval()
    opt_G.zero_grad()
    opt_C.zero_grad()
    sample = torch.randn(real.size(0), n_feature, device=device).clamp(-2,2) * std
    generated = G_net(sample, label_ohe)
    g_loss, g_bce_loss = C_net(generated, label)
    g_bce_loss = lambda_c.detach() * g_bce_loss
    (g_loss + g_bce_loss).backward()
    opt_G.step()
    g_loss = g_loss.item()
    g_bce_loss = g_bce_loss.item()
    G_update_ts = time.time()
    k_delta = gamma.detach()*d_loss_real-g_loss
    M_global = (d_loss_real + torch.abs(k_delta)).item()
    k = (k + lambda_k.detach()*k_delta).clamp(min=min_k, max=1)
    if ite%preview_iter==0:
        print('[{}/{}] G:{:.4f}, D:{:.4f}, C_g:{:4f}, C_d:{:4f}, M:{:4f}, k:{:4f} -- elapsed_G: {:.4f}s -- elapsed_D: {:.4f}s'.format(ite, iterations, g_loss, d_loss_mean, g_bce_loss, c_loss_mean, M_global, k.item(), (G_update_ts-D_update_ts), (D_update_ts-start_train_ts) ))
        
        imgs = []
        for c in range(n_class):
            sample = torch.randn(1, n_feature, device=device).clamp(-2,2) * std
            label  = one_hot(torch.LongTensor([c], device="cpu"), n_class).to(device)
            generated = G_net(sample, label).detach().cpu().numpy()[0]
            imgs.append(generated)
        imgs = np.asarray(imgs)
        
        plot2dir('./previews', imgs, ite)
        torch.save(G_net.state_dict(), './checkpoints/iter-{:d}-G.ckpt'.format(ite))
        torch.save(C_net.state_dict(), './checkpoints/iter-{:d}-D.ckpt'.format(ite))
