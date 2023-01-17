import os

import math

import random

import itertools

import numpy as np

from scipy import io

from PIL import Image

import torch

import torch.nn as nn

from torch.nn.utils import spectral_norm

import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torchvision.utils as vutils

from tqdm import tqdm_notebook as tqdm

from pathlib import Path
class SelfAttentionNaive(nn.Module):

    def __init__(self, nf, nh=False):

        super(SelfAttentionNaive, self).__init__()

        if not nh:

            nh = max(nf//8, 1)

        self.f = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))

        self.g = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))

        self.h = spectral_norm(nn.Conv2d(nf, nf, 1, bias=False))

        self.gamma = nn.Parameter(torch.zeros(1))

        self.nh = nh

        self.nf = nf

    def forward(self, x):

        fx = self.f(x).view(x.size(0), self.nh, x.size(2)*x.size(3))

        gx = self.g(x).view(x.size(0), self.nh, x.size(2)*x.size(3))

        hx = self.h(x).view(x.size(0), self.nf, x.size(2)*x.size(3))

        s = fx.transpose(-1,-2).matmul(gx)

        b = F.softmax(s, dim=1)

        o = hx.matmul(b)

        return o.view_as(x) * self.gamma + x



class SelfAttention(nn.Module):

    def __init__(self, nf, nh=False):

        super(SelfAttention, self).__init__()

        if not nh:

            nh = max(nf//8, 1)

        self.f = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))

        self.g = spectral_norm(nn.Conv2d(nf, nh, 1, bias=False))

        self.h = spectral_norm(nn.Conv2d(nf, nf//2, 1, bias=False))

        self.o = spectral_norm(nn.Conv2d(nf//2, nf, 1, bias=False))

        self.gamma = nn.Parameter(torch.zeros(1))

        self.nh = nh

        self.nf = nf

    def forward(self, x):

        fx = self.f(x).view(x.size(0), self.nh, x.size(2)*x.size(3))

        gx = self.g(x)

        gx = F.max_pool2d(gx, kernel_size=2)

        gx = gx.view(x.size(0), self.nh, x.size(2)*x.size(3)//4)

        s = gx.transpose(-1,-2).matmul(fx)

        s = F.softmax(s, dim=1)

        hx = self.h(x)

        hx = F.max_pool2d(hx, kernel_size=2)

        hx = hx.view(x.size(0), self.nf//2, x.size(2)*x.size(3)//4)

        ox = hx.matmul(s).view(x.size(0), self.nf//2, x.size(2), x.size(3))

        ox = self.o(ox)

        return ox * self.gamma + x

    

class _resDiscriminator128(nn.Module):

    def __init__(self, nIn=3, nf=64, selfAtt=False):

        super(_resDiscriminator128, self).__init__()

        self.blocs = []

        self.sc = []

        # first bloc

        self.bloc0 = nn.Sequential(spectral_norm(nn.Conv2d(nIn, nf, 3, 1, 1, bias=True)),

                                   nn.ReLU(),

                                   spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),

                                   nn.AvgPool2d(2),)

        self.sc0 = nn.Sequential(nn.AvgPool2d(2),

                                 spectral_norm(nn.Conv2d(nIn, nf, 1, bias=True)),)

        if selfAtt:

            self.selfAtt = SelfAttention(nf)

        else:

            self.selfAtt = nn.Sequential()

        # Down blocs

        for i in range(4):

            nfPrev = nf

            nf = nf*2

            self.blocs.append(nn.Sequential(nn.ReLU(),

                                            spectral_norm(nn.Conv2d(nfPrev, nf, 3, 1, 1, bias=True)),

                                            nn.ReLU(),

                                            spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),

                                            nn.AvgPool2d(2),))

            self.sc.append(nn.Sequential(nn.AvgPool2d(2),

                                         spectral_norm(nn.Conv2d(nfPrev, nf, 1, bias=True)),))

        # Last Bloc

        self.blocs.append(nn.Sequential(nn.ReLU(),

                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),

                                        nn.ReLU(),

                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))))

        self.sc.append(nn.Sequential())

        self.dense = nn.Linear(nf, 1)

        self.blocs = nn.ModuleList(self.blocs)

        self.sc = nn.ModuleList(self.sc)

    def forward(self, x):

        x = self.selfAtt(self.bloc0(x) + self.sc0(x))

        for k in range(len(self.blocs)):

            x = self.blocs[k](x) + self.sc[k](x)

        x = x.sum(3).sum(2)

        return self.dense(x)



class _resEncoder128(nn.Module):

    def __init__(self, nIn=3, nf=64, nOut=8):

        super(_resEncoder128, self).__init__()

        self.blocs = []

        self.sc = []

        # first bloc

        self.blocs.append(nn.Sequential(spectral_norm(nn.Conv2d(nIn, nf, 3, 1, 1, bias=True)),

                                        nn.ReLU(),

                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),

                                        nn.AvgPool2d(2),))

        self.sc.append(nn.Sequential(nn.AvgPool2d(2),

                                     spectral_norm(nn.Conv2d(nIn, nf, 1, bias=True)),))

        # Down blocs

        for i in range(4):

            nfPrev = nf

            nf = nf*2

            self.blocs.append(nn.Sequential(nn.ReLU(),

                                            spectral_norm(nn.Conv2d(nfPrev, nf, 3, 1, 1, bias=True)),

                                            nn.ReLU(),

                                            spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),

                                            nn.AvgPool2d(2),))

            self.sc.append(nn.Sequential(nn.AvgPool2d(2),

                                         spectral_norm(nn.Conv2d(nfPrev, nf, 1, bias=True)),))

        # Last Bloc

        self.blocs.append(nn.Sequential(nn.ReLU(),

                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)),

                                        nn.ReLU(),

                                        spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))))

        self.sc.append(nn.Sequential())

        self.dense = nn.Linear(nf, nOut)

        self.blocs = nn.ModuleList(self.blocs)

        self.sc = nn.ModuleList(self.sc)

    def forward(self, x):

        for k in range(len(self.blocs)):

            x = self.blocs[k](x) + self.sc[k](x)

        x = x.sum(3).sum(2)

        return self.dense(x)

    

class _resMaskedGenerator128(nn.Module):

    def __init__(self, nf=64, nOut=3, nc=8, selfAtt=False):

        super(_resMaskedGenerator128, self).__init__()

        if selfAtt:

            self.selfAtt = SelfAttention(nf*2)

        else:

            self.selfAtt = nn.Sequential()

        self.dense = nn.Linear(nc, 4*4*nf*16)

        self.convA = []

        self.convB = []

        self.normA = []

        self.normB = []

        self.gammaA = []

        self.gammaB = []

        self.betaA = []

        self.betaB = []

        self.sc = []

        nfPrev = nf*16

        nfNext = nf*16

        for k in range(5):

            self.convA.append(nn.Sequential(nn.Upsample(scale_factor=2),

                                            spectral_norm(nn.Conv2d(nfPrev + 1, nfNext, 3, 1, 1, bias=False)),))

            self.convB.append(spectral_norm(nn.Conv2d(nfNext, nfNext, 3, 1, 1, bias=True )))

            self.normA.append(nn.InstanceNorm2d(nfPrev, affine=False))

            self.normB.append(nn.InstanceNorm2d(nfNext, affine=False))

            self.gammaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))

            self.gammaB.append(nn.Conv2d(nc, nfNext, 1, bias=True))

            self.betaA.append(nn.Conv2d(nc, nfPrev, 1, bias=True))

            self.betaB.append(nn.Conv2d(nc, nfNext, 1, bias=True))

            self.sc.append(nn.Sequential(nn.Upsample(scale_factor=2),

                                         spectral_norm(nn.Conv2d(nfPrev, nfNext, 1, bias=True))))

            nfPrev = nfNext

            nfNext = nfNext // 2

        self.convA = nn.ModuleList(self.convA)

        self.convB = nn.ModuleList(self.convB)

        self.normA = nn.ModuleList(self.normA)

        self.normB = nn.ModuleList(self.normB)

        self.gammaA =nn.ModuleList(self.gammaA)

        self.gammaB =nn.ModuleList(self.gammaB)

        self.betaA = nn.ModuleList(self.betaA)

        self.betaB = nn.ModuleList(self.betaB)

        self.sc = nn.ModuleList(self.sc)

        self.normOut = nn.InstanceNorm2d(nf, affine=False)

        self.gammaOut = nn.Conv2d(nc, nf, 1, bias=True)

        self.betaOut = nn.Conv2d(nc, nf, 1, bias=True)

        self.convOut = spectral_norm(nn.Conv2d(nf, nOut, 3, 1, 1))

        self.convOut = spectral_norm(nn.Conv2d(nf + 1, nOut, 3, 1, 1))

        ##############################

    def forward(self, m, z, c):

        ######### Upsample ###########

        x = self.dense(z.view(z.size(0),z.size(1))).view(z.size(0), -1, 4, 4)

        mask_ratio = m.size(-1) // 4

        for k in range(5):

            if k == 4:

                x = self.selfAtt(x)

            h = self.convA[k](torch.cat((F.relu(self.normA[k](x) * self.gammaA[k](c) + self.betaA[k](c)),

                                         F.avg_pool2d(m, kernel_size=mask_ratio)), 1))

            h = self.convB[k](F.relu(self.normB[k](h) * self.gammaB[k](c) + self.betaB[k](c)))

            x = h + self.sc[k](x)

            mask_ratio = mask_ratio // 2

        x = self.convOut(torch.cat((F.relu(self.normOut(x) * self.gammaOut(c) + self.betaOut(c)),

                                    m), 1))

        x = torch.tanh(x)

        return x * m

    

class _downConv(nn.Module):

    def __init__(self, nIn=3, nf=128, spectralNorm=False):

        super(_downConv, self).__init__()

        self.mods = nn.Sequential(nn.ReflectionPad2d(3),

                                  spectral_norm(nn.Conv2d(nIn, nf//4, 7, bias=False)) if spectralNorm else nn.Conv2d(nIn, nf//4, 7, bias=False),

                                  nn.InstanceNorm2d(nf//4, affine=True),

                                  nn.ReLU(),

                                  spectral_norm(nn.Conv2d(nf//4, nf//2, 3, 2, 1, bias=False)) if spectralNorm else nn.Conv2d(nf//4, nf//2, 3, 2, 1, bias=False),

                                  nn.InstanceNorm2d(nf//2, affine=True),

                                  nn.ReLU(),

                                  spectral_norm(nn.Conv2d(nf//2, nf, 3, 2, 1, bias=False)) if spectralNorm else nn.Conv2d(nf//2, nf, 3, 2, 1, bias=False),

                                  nn.InstanceNorm2d(nf, affine=True),

                                  nn.ReLU(),

        )

    def forward(self, x):

        return self.mods(x)

class _resBloc(nn.Module):

    def __init__(self, nf=128, spectralNorm=False):

        super(_resBloc, self).__init__()

        self.blocs = nn.Sequential(spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)) if spectralNorm else nn.Conv2d(nf, nf, 3, 1, 1, bias=False),

                                   nn.InstanceNorm2d(nf, affine=True),

                                   nn.ReLU(),

                                   spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True)) if spectralNorm else nn.Conv2d(nf, nf, 3, 1, 1, bias=True),

        )

        self.activationF = nn.Sequential(nn.InstanceNorm2d(nf, affine=True),

                                         nn.ReLU(),

        )

    def forward(self, x):

        return self.activationF(self.blocs(x) + x)

class _upConv(nn.Module):

    def __init__(self, nOut=3, nf=128, spectralNorm=False):

        super(_upConv, self).__init__()

        self.mods = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),

                                  spectral_norm(nn.Conv2d(nf, nf//2, 3, 1, 1, bias=False)) if spectralNorm else nn.Conv2d(nf, nf//2, 3, 1, 1, bias=False),

                                  nn.InstanceNorm2d(nf//2, affine=True),

                                  nn.ReLU(),

                                  nn.Upsample(scale_factor=2, mode='nearest'),

                                  spectral_norm(nn.Conv2d(nf//2, nf//4, 3, 1, 1, bias=False)) if spectralNorm else nn.Conv2d(nf//2, nf//4, 3, 1, 1, bias=False),

                                  nn.InstanceNorm2d(nf//4, affine=True),

                                  nn.ReLU(),

                                  nn.ReflectionPad2d(3),

                                  spectral_norm(nn.Conv2d(nf//4, nOut, 7, bias=True)) if spectralNorm else nn.Conv2d(nf//4, nOut, 7, bias=True),

        )

    def forward(self, x):

        return self.mods(x)

class _netEncM(nn.Module):

    def __init__(self, sizex=128, nIn=3, nMasks=2, nRes=5, nf=128, temperature=1):

        super(_netEncM, self).__init__()

        self.nMasks = nMasks

        sizex = sizex // 4 

        self.cnn = nn.Sequential(*([_downConv(nIn, nf)] +

                                   [_resBloc(nf=nf) for i in range(nRes)]))

        self.psp = nn.ModuleList([nn.Sequential(nn.AvgPool2d(sizex),

                                                nn.Conv2d(nf,1,1),

                                                nn.Upsample(size=sizex, mode='bilinear')),

                                  nn.Sequential(nn.AvgPool2d(sizex//2, sizex//2),

                                                nn.Conv2d(nf,1,1),

                                                nn.Upsample(size=sizex, mode='bilinear')),

                                  nn.Sequential(nn.AvgPool2d(sizex//3, sizex//3),

                                                nn.Conv2d(nf,1,1),

                                                nn.Upsample(size=sizex, mode='bilinear')),

                                  nn.Sequential(nn.AvgPool2d(sizex//6, sizex//6),

                                                nn.Conv2d(nf,1,1),

                                                nn.Upsample(size=sizex, mode='bilinear'))])

        self.out = _upConv(1 if nMasks == 2 else nMasks, nf+4)

        self.temperature = temperature

    def forward(self, x):

        f = self.cnn(x)

        m = self.out(torch.cat([f] + [pnet(f) for pnet in self.psp], 1))

        if self.nMasks == 2:

            m = torch.sigmoid(m / self.temperature)

            m = torch.cat((m, (1-m)), 1)

        else:

            m = F.softmax(m / self.temperature, dim=1)

        return m



class _netGenX(nn.Module):

    def __init__(self, sizex=128, nOut=3, nc=8, nf=64, nMasks=2, selfAtt=False):

        super(_netGenX, self).__init__()

        if sizex != 128:

            raise NotImplementedError

        self.net = nn.ModuleList([_resMaskedGenerator128(nf=nf, nOut=nOut, nc=nc, selfAtt=selfAtt) for k in range(nMasks)])

        self.nMasks = nMasks

    def forward(self, masks, c):

        masks = masks.unsqueeze(2)

        y = []

        for k in range(self.nMasks):

            y.append(self.net[k](masks[:,k], c[:,k], c[:,k]).unsqueeze(1))

        return torch.cat(y,1)

    

class _netRecZ(nn.Module):

    def __init__(self, sizex=128, nIn=3, nc=5, nf=64, nMasks=2):

        super(_netRecZ, self).__init__()

        if sizex == 128:

            self.net = _resEncoder128(nIn=nIn, nf=nf, nOut=nc*nMasks)

        elif sizex == 64:

            self.net = _resEncoder64(nIn=nIn, nf=nf, nOut=nc*nMasks)

        self.nc = nc

        self.nMasks = nMasks

    def forward(self, x):

        c = self.net(x)

        return c.view(c.size(0), self.nMasks, self.nc, 1 , 1)
random.seed(2019)

torch.manual_seed(2019)

device = torch.device("cuda:0")

cudnn.benchmark = True

SIZE_X = 128

N_X = 3

N_Z = 64

N_F = 64

N_RES_M = 3 # number of residual blocks

N_MASKS = 2 # number of masks

initOrthoGain = 0.8

wdecay = 1e-4

wrecZ = 5

lrG = 1e-4

lrM = 1e-5

lrD = 1e-4

lrZ = 1e-4

nIteration = 20000

ITER_STEPS = 500 # steps per iteration

N_TEST = 5

BATCH_SIZE = 20

dataroot = Path('../input/oxford-flower-segmentations/')
class FlowersDataset(torch.utils.data.Dataset):

    def __init__(self, dataPath, sets='train', transform=transforms.ToTensor()):

        super(FlowersDataset, self).__init__()

        self.files =  io.loadmat(os.path.join(dataPath, "setid.mat"))

        if sets == 'train':

            self.files = self.files.get('tstid')[0]

        elif sets == 'val':

            self.files = self.files.get('valid')[0]

        else:

            self.files = self.files.get('trnid')[0]

        self.transform = transform

        self.datapath = dataPath

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        imgname = "image_%05d.jpg" % self.files[idx]

        segname = "segmim_%05d.jpg" % self.files[idx]

        img = self.transform(Image.open(os.path.join(self.datapath, "jpg", "jpg", imgname)))

        seg = np.array(Image.open(os.path.join(self.datapath, "segmim", "segmim", segname)))

        seg = 1 - ((seg[:,:,0:1] == 0) + (seg[:,:,1:2] == 0) + (seg[:,:,2:3] == 254))

        seg = (seg * 255).astype('uint8').repeat(3,axis=2)

        seg = self.transform(Image.fromarray(seg))[:1]

        return img * 2 - 1, seg
trainset = FlowersDataset(dataPath=dataroot,

                                   sets='train',

                                   transform=transforms.Compose([transforms.Resize(SIZE_X, Image.NEAREST),

                                                                 transforms.CenterCrop(SIZE_X),

                                                                 transforms.ToTensor(),

                                   ]),)

testset = FlowersDataset(dataPath=dataroot,

                              sets='test',

                              transform=transforms.Compose([transforms.Resize(SIZE_X, Image.NEAREST),

                                                            transforms.CenterCrop(SIZE_X),

                                                            transforms.ToTensor(),

                              ]),)

valset = FlowersDataset(dataPath=dataroot,

                             sets='val',

                             transform=transforms.Compose([transforms.Resize(SIZE_X, Image.NEAREST),

                                                           transforms.CenterCrop(SIZE_X),

                                                           transforms.ToTensor(),

                             ]),)
def weights_init_ortho(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):

        nn.init.orthogonal_(m.weight, initOrthoGain)
netEncM = _netEncM(sizex=SIZE_X, nIn=N_X, nMasks=N_MASKS, nRes=N_RES_M, nf=N_F, temperature=1).to(device)

netGenX = _netGenX(sizex=SIZE_X, nOut=N_X, nc=N_Z, nf=N_F, nMasks=N_MASKS, selfAtt=True).to(device)

netDX = _resDiscriminator128(nIn=N_X, nf=N_F, selfAtt=True).to(device)
netEncM.apply(weights_init_ortho)

netGenX.apply(weights_init_ortho)

netDX.apply(weights_init_ortho)
optimizerEncM = torch.optim.Adam(netEncM.parameters(), lr=lrM, betas=(0, 0.9), weight_decay=wdecay, amsgrad=False)

optimizerGenX = torch.optim.Adam(netGenX.parameters(), lr=lrG, betas=(0, 0.9), amsgrad=False)

optimizerDX = torch.optim.Adam(netDX.parameters(), lr=lrD, betas=(0, 0.9), amsgrad=False)
if wrecZ > 0:

    netRecZ = _netRecZ(sizex=SIZE_X, nIn=N_X, nc=N_Z, nf=N_Z, nMasks=N_MASKS).to(device)

    netRecZ.apply(weights_init_ortho)

    optimizerRecZ = torch.optim.Adam(netRecZ.parameters(), lr=lrZ, betas=(0, 0.9), amsgrad=False)
def evaluate(netEncM, loader, device, nMasks=2):

    sumScoreAcc = 0

    sumScoreIoU = 0

    nbIter = 0

    if nMasks > 2:

        raise NotImplementedError

    for xLoad, mLoad in loader:

        xData = xLoad.to(device)

        mData = mLoad.to(device)

        mPred = netEncM(xData)

        sumScoreAcc += torch.max(((mPred[:,:1] >= .5).float() == mData).float().mean(-1).mean(-1),

                                 ((mPred[:,:1] <  .5).float() == mData).float().mean(-1).mean(-1)).mean().item()

        sumScoreIoU += torch.max(

            ((((mPred[:,:1] >= .5).float() + mData) == 2).float().sum(-1).sum(-1) /

             (((mPred[:,:1] >= .5).float() + mData) >= 1).float().sum(-1).sum(-1)),

            ((((mPred[:,:1] <  .5).float() + mData) == 2).float().sum(-1).sum(-1) /

             (((mPred[:,:1] <  .5).float() + mData) >= 1).float().sum(-1).sum(-1))).mean().item()

        nbIter += 1

    return sumScoreAcc / nbIter, sumScoreIoU / nbIter
x_test, m_test = next(iter(torch.utils.data.DataLoader(testset, batch_size=N_TEST, shuffle=True, num_workers=4, drop_last=True)))



x_test = x_test.to(device)



z_test = torch.randn((N_TEST, N_MASKS, N_Z, 1, 1), device=device)

zn_test = torch.randn((N_TEST, N_Z, 1, 1), device=device)



img_m_test = m_test[:,:1].float()

for n in range(N_TEST):

    img_m_test[n] = (img_m_test[n] / img_m_test[n].max()) * 2 - 1



out_X = torch.full((N_MASKS, N_TEST+1, N_TEST+5, N_X, SIZE_X, SIZE_X), -1).to(device)

out_X[:,1:,0] = x_test

out_X[:,1:,1] = img_m_test



valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

genData = iter(trainloader)

disData = iter(trainloader)
iteration = 0

bestValIoU = 0

pbar = tqdm(total=ITER_STEPS)

while iteration <= nIteration:

    pbar.update(1)

    ########################## Get Batch #############################

    try:

        xLoadG, mLoadG = next(genData)

    except StopIteration:

        genData = iter(trainloader)

        xLoadG, mLoadG = next(genData)

    try:

        xLoadD, mLoadD = next(disData)

    except StopIteration:

        disData = iter(trainloader)

        xLoadD, mLoadD = next(disData)

    xData = xLoadG.to(device)

    mData = mLoadG.to(device)

    xReal = xLoadD.to(device)

    zData = torch.randn((xData.size(0), N_MASKS, N_Z, 1, 1), device=device)

    ########################## Reset Nets ############################

    netEncM.zero_grad()

    netGenX.zero_grad()

    netDX.zero_grad()

    netEncM.train()

    netGenX.train()

    netDX.train()

    if wrecZ > 0:

        netRecZ.zero_grad()

        netRecZ.train()

    dStep = (iteration % 4 == 0)

    gStep = (iteration % 4 == 0)

    #########################  AutoEncode X #########################

    if gStep:

        mEnc = netEncM(xData)

        hGen = netGenX(mEnc, zData)

        xGen = (hGen + ((1 - mEnc.unsqueeze(2)) * xData.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))

        dGen = netDX(xGen)

        lossG = - dGen.mean()

        if wrecZ > 0:

            zRec = netRecZ(hGen.sum(1))

            err_recZ = ((zData - zRec) * (zData - zRec)).mean()

            lossG += err_recZ * wrecZ

        lossG.backward()

        optimizerEncM.step()

        optimizerGenX.step()

        if wrecZ > 0:

            optimizerRecZ.step()

    if dStep:

        netDX.zero_grad()

        with torch.no_grad():

            mEnc = netEncM(xData)

            hGen = netGenX(mEnc, zData)

            xGen = (hGen + ((1 - mEnc.unsqueeze(2)) * xData.unsqueeze(1))).view(hGen.size(0) * hGen.size(1), hGen.size(2), hGen.size(3), hGen.size(4))

        dPosX = netDX(xReal)

        dNegX = netDX(xGen)

        err_dPosX = (-1 + dPosX)

        err_dNegX = (-1 - dNegX)

        err_dPosX = ((err_dPosX < 0).float() * err_dPosX).mean()

        err_dNegX = ((err_dNegX < 0).float() * err_dNegX).mean()

        (-err_dPosX - err_dNegX).backward()

        optimizerDX.step()

    iteration += 1

    if iteration % ITER_STEPS == 0:

        pbar.close()

        netEncM.eval()

        netGenX.eval()

        netDX.eval()

        if wrecZ > 0:

            netRecZ.eval()

        with torch.no_grad():

            mEnc_test = netEncM(x_test)

            out_X[:,1:,3] = mEnc_test.transpose(0,1).unsqueeze(2)*2-1

            out_X[:,1:,2] = ((out_X[:,1:,3] < 0).float() * -1) + (out_X[:,1:,3] > 0).float()

            out_X[:,1:,4] = (netGenX(mEnc_test, z_test) + ((1 - mEnc_test.unsqueeze(2)) * x_test.unsqueeze(1))).transpose(0,1)

            for k in range(N_MASKS):

                for i in range(N_TEST):

                    zx_test = z_test.clone()

                    zx_test[:, k] = zn_test[i]

                    out_X[k, 1:, i+5] = netGenX(mEnc_test, zx_test)[:,k] + ((1 - mEnc_test[:,k:k+1]) * x_test)

            scoreAccTrain, scoreIoUTrain = evaluate(netEncM, trainloader, device, N_MASKS)

            scoreAccVal, scoreIoUVal = evaluate(netEncM, valloader, device, N_MASKS)

            print("train:", scoreAccTrain, scoreIoUTrain)

            print("val:", scoreAccVal, scoreIoUVal)

            try:

                with open( 'train.dat', 'a') as f:

                    f.write(str(iteration) + ' ' + str(scoreAccTrain) + ' ' + str(scoreIoUTrain) + '\n')

            except:

                print("Cannot save in train.dat")

            try:

                with open( 'val.dat', 'a') as f:

                    f.write(str(iteration) + ' ' + str(scoreAccVal) + ' ' + str(scoreIoUVal) + '\n')

            except:

                print("Cannot save in val.dat")

            try:

                vutils.save_image(out_X.view(-1,N_X,SIZE_X, SIZE_X), "out_%05d.png" % iteration, normalize=True, range=(-1,1), nrow=N_TEST+5)

            except:

                print("Cannot save output")

        netEncM.zero_grad()

        netGenX.zero_grad()

        netDX.zero_grad()

        stateDic = {

            'netEncM': netEncM.state_dict(),

            'netGenX': netGenX.state_dict(),

            'netDX': netDX.state_dict(),

            'optimizerEncM': optimizerEncM.state_dict(),

            'optimizerGenX': optimizerGenX.state_dict(),

            'optimizerDX': optimizerDX.state_dict(),

        }

        if wrecZ > 0:

            netRecZ.zero_grad()

            stateDic['netRecZ'] = netRecZ.state_dict()

            stateDic['optimizerRecZ'] = optimizerRecZ.state_dict(),

        try:

            torch.save(stateDic, 'state.pth')

        except:

            print("Cannot save checkpoint")

        if bestValIoU < scoreIoUVal:

            bestValIoU = scoreIoUVal

            try:

                torch.save(stateDic,  'best.pth')

            except:

                print("Cannot save best")

        

        pbar = tqdm(total=ITER_STEPS)

        netEncM.train()

        netGenX.train()

        netDX.train()

        if wrecZ > 0:

            netRecZ.train()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams["figure.figsize"] = (15, 10)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

plt.rcParams['image.cmap'] = 'gray' # grayscale looks better

from skimage.util import montage

montage_rgb = lambda x, **kwargs: np.stack([montage(x[:, :, :, i], **kwargs) for i in range(x.shape[3])], -1)
out_image = out_X.view(-1,N_X,SIZE_X, SIZE_X).to('cpu').detach().numpy().swapaxes(1, 3).swapaxes(1,2)

plt.imshow(montage_rgb(out_image, grid_shape=(13, 10)))