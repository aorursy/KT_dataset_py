import os, sys

import time, math

import argparse, random

from math import exp

import numpy as np



import torch

from torch import nn, optim

import torch.nn.functional as F

import torch.utils.data as data

from torch.utils.data import DataLoader

from torch.backends import cudnn

from torch.autograd import Variable



import torchvision

import torchvision.transforms as tfs

from torchvision.transforms import ToPILImage

from torchvision.transforms import functional as FF

import torchvision.utils as vutils

from torchvision.utils import make_grid

from torchvision.models import vgg16



from PIL import Image

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
# number of training steps

steps = 10000

# Device name

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# resume Training

resume = True

# number of evaluation steps

eval_step = 5000

# learning rate

learning_rate = 0.0001

# pre-trained model directory

pretrained_model_dir = '../input/ffa-net-for-single-image-dehazing-pytorch/trained_models/'

# directory to save models to

model_dir = './trained_models/'

# train data

trainset = 'its_train'

# test data

testset = 'its_test'

# model to be used

network = 'ffa'

# residual_groups

gps = 3

# residual_blocks

blocks = 12

# batch size

bs = 1

# crop image

crop = True

# Takes effect when crop = True

crop_size = 240

# No lr cos schedule

no_lr_sche = True

# perceptual loss

perloss = True



model_name = trainset + '_' + network.split('.')[0] + '_' + str(gps) + '_' + str(blocks)

pretrained_model_dir = pretrained_model_dir + model_name + '.pk'

model_dir = model_dir + model_name + '.pk'

log_dir = 'logs/' + model_name



if not os.path.exists('trained_models'):

    os.mkdir('trained_models')

if not os.path.exists('numpy_files'):

    os.mkdir('numpy_files')

if not os.path.exists('logs'):

    os.mkdir('logs')

if not os.path.exists('samples'):

    os.mkdir('samples')

if not os.path.exists(f"samples/{model_name}"):

    os.mkdir(f'samples/{model_name}')

if not os.path.exists(log_dir):

    os.mkdir(log_dir)

    

crop_size='whole_img'

if crop:

    crop_size = crop_size

def tensorShow(tensors,titles=None):

    '''t:BCWH'''

    fig=plt.figure()

    for tensor, title, i in zip(tensors, titles, range(len(tensors))):

        img = make_grid(tensor)

        npimg = img.numpy()

        ax = fig.add_subplot(211+i)

        ax.imshow(np.transpose(npimg, (1, 2, 0)))

        ax.set_title(title)

    plt.show()

    

def lr_schedule_cosdecay(t, T, init_lr=learning_rate):

    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr

    return lr
def default_conv(in_channels, out_channels, kernel_size, bias=True):

    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

    

    

class PALayer(nn.Module):

    def __init__(self, channel):

        super(PALayer, self).__init__()

        self.pa = nn.Sequential(

                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),

                nn.ReLU(inplace=True),

                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),

                nn.Sigmoid()

        )

    def forward(self, x):

        y = self.pa(x)

        return x * y



    

class CALayer(nn.Module):

    def __init__(self, channel):

        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.ca = nn.Sequential(

                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),

                nn.ReLU(inplace=True),

                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),

                nn.Sigmoid()

        )



    def forward(self, x):

        y = self.avg_pool(x)

        y = self.ca(y)

        return x * y



    

class Block(nn.Module):

    def __init__(self, conv, dim, kernel_size,):

        super(Block, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)

        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = conv(dim, dim, kernel_size, bias=True)

        self.calayer = CALayer(dim)

        self.palayer = PALayer(dim)



    def forward(self, x):

        res = self.act1(self.conv1(x))

        res = res+x 

        res = self.conv2(res)

        res = self.calayer(res)

        res = self.palayer(res)

        res += x 

        return res



    

class Group(nn.Module):

    def __init__(self, conv, dim, kernel_size, blocks):

        super(Group, self).__init__()

        modules = [Block(conv, dim, kernel_size)  for _ in range(blocks)]

        modules.append(conv(dim, dim, kernel_size))

        self.gp = nn.Sequential(*modules)



    def forward(self, x):

        res = self.gp(x)

        res += x

        return res



    

class FFA(nn.Module):

    def __init__(self,gps,blocks,conv=default_conv):

        super(FFA, self).__init__()

        self.gps = gps

        self.dim = 64

        kernel_size = 3

        pre_process = [conv(3, self.dim, kernel_size)]

        assert self.gps==3

        self.g1 = Group(conv, self.dim, kernel_size,blocks=blocks)

        self.g2 = Group(conv, self.dim, kernel_size,blocks=blocks)

        self.g3 = Group(conv, self.dim, kernel_size,blocks=blocks)

        self.ca = nn.Sequential(*[

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),

            nn.ReLU(inplace=True),

            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),

            nn.Sigmoid()

            ])

        self.palayer = PALayer(self.dim)



        post_process = [

            conv(self.dim, self.dim, kernel_size),

            conv(self.dim, 3, kernel_size)]



        self.pre = nn.Sequential(*pre_process)

        self.post = nn.Sequential(*post_process)



    def forward(self, x1):

        x = self.pre(x1)

        res1 = self.g1(x)

        res2 = self.g2(res1)

        res3 = self.g3(res2)

        w = self.ca(torch.cat([res1,res2,res3],dim=1))

        w = w.view(-1,self.gps, self.dim)[:,:,:,None,None]

        out = w[:,0,::] * res1 + w[:,1,::] * res2+w[:,2,::] * res3

        out = self.palayer(out)

        x = self.post(out)

        return x + x1
# --- Perceptual loss network  --- #

class PerLoss(torch.nn.Module):

    def __init__(self, vgg_model):

        super(PerLoss, self).__init__()

        self.vgg_layers = vgg_model

        self.layer_name_mapping = {

            '3': "relu1_2",

            '8': "relu2_2",

            '15': "relu3_3"

        }



    def output_features(self, x):

        output = {}

        for name, module in self.vgg_layers._modules.items():

            x = module(x)

            if name in self.layer_name_mapping:

                output[self.layer_name_mapping[name]] = x

        return list(output.values())



    def forward(self, dehaze, gt):

        loss = []

        dehaze_features = self.output_features(dehaze)

        gt_features = self.output_features(gt)

        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):

            loss.append(F.mse_loss(dehaze_feature, gt_feature))



        return sum(loss)/len(loss)
def gaussian(window_size, sigma):

    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

    return gauss / gauss.sum()



def create_window(window_size, channel):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window



def _ssim(img1, img2, window, window_size, channel, size_average=True):

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)

    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)

    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq

    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2

    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))



    if size_average:

        return ssim_map.mean()

    else:

        return ssim_map.mean(1).mean(1).mean(1)



def ssim(img1, img2, window_size=11, size_average=True):

    img1=torch.clamp(img1,min=0,max=1)

    img2=torch.clamp(img2,min=0,max=1)

    (_, channel, _, _) = img1.size()

    window = create_window(window_size, channel)

    if img1.is_cuda:

        window = window.cuda(img1.get_device())

    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)



def psnr(pred, gt):

    pred=pred.clamp(0,1).cpu().numpy()

    gt=gt.clamp(0,1).cpu().numpy()

    imdff = pred - gt

    rmse = math.sqrt(np.mean(imdff ** 2))

    if rmse == 0:

        return 100

    return 20 * math.log10( 1.0 / rmse)
class RESIDE_Dataset(data.Dataset):

    def __init__(self, path, train, size=crop_size, format='.png'):

        super(RESIDE_Dataset, self).__init__()

        self.size = size

        self.train = train

        self.format = format

        self.haze_imgs_dir = os.listdir(os.path.join(path,'hazy'))

        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]

        self.clear_dir = os.path.join(path,'clear')

        

    def __getitem__(self, index):

        haze = Image.open(self.haze_imgs[index])

        if isinstance(self.size, int):

            while haze.size[0] < self.size or haze.size[1] < self.size :

                index = random.randint(0, 20000)

                haze = Image.open(self.haze_imgs[index])

        img = self.haze_imgs[index]

        id = img.split('/')[-1].split('_')[0]

        clear_name = id + self.format

        clear = Image.open(os.path.join(self.clear_dir, clear_name))

        clear = tfs.CenterCrop(haze.size[::-1])(clear)

        if not isinstance(self.size, str):

            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))

            haze = FF.crop(haze, i, j, h, w)

            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB") )

        return haze, clear

    

    def augData(self, data, target):

        if self.train:

            rand_hor = random.randint(0,1)

            rand_rot = random.randint(0,3)

            data = tfs.RandomHorizontalFlip(rand_hor)(data)

            target = tfs.RandomHorizontalFlip(rand_hor)(target)

            if rand_rot:

                data = FF.rotate(data, 90*rand_rot)

                target = FF.rotate(target, 90*rand_rot)

        data = tfs.ToTensor()(data)

        data = tfs.Normalize(mean=[0.64,0.6,0.58], std=[0.14,0.15,0.152])(data)

        target = tfs.ToTensor()(target)

        return data, target



    def __len__(self):

        return len(self.haze_imgs)





# path to your 'data' folder

its_train_path = '../input/indoor-training-set-its-residestandard'

its_test_path = '../input/synthetic-objective-testing-set-sots-reside/indoor'



ITS_train_loader = DataLoader(dataset=RESIDE_Dataset(its_train_path, train=True, size=crop_size), batch_size=bs, shuffle=True)

ITS_test_loader = DataLoader(dataset=RESIDE_Dataset(its_test_path, train=False, size='whole img'), batch_size=1, shuffle=False)
print('log_dir :', log_dir)

print('model_name:', model_name)



models_ = {'ffa': FFA(gps = gps, blocks = blocks)}

loaders_ = {'its_train': ITS_train_loader, 'its_test': ITS_test_loader}

# loaders_ = {'its_train': ITS_train_loader, 'its_test': ITS_test_loader, 'ots_train': OTS_train_loader, 'ots_test': OTS_test_loader}

start_time = time.time()

T = steps



def train(net, loader_train, loader_test, optim, criterion):

    losses = []

    start_step = 0

    max_ssim = max_psnr = 0

    ssims, psnrs = [], []

    if resume and os.path.exists(pretrained_model_dir):

        print(f'resume from {pretrained_model_dir}')

        ckp = torch.load(pretrained_model_dir)

        losses = ckp['losses']

        net.load_state_dict(ckp['model'])

        start_step = ckp['step']

        max_ssim = ckp['max_ssim']

        max_psnr = ckp['max_psnr']

        psnrs = ckp['psnrs']

        ssims = ckp['ssims']

        print(f'Resuming training from step: {start_step} ***')

    else :

        print('Training from scratch *** ')

    for step in range(start_step+1, steps+1):

        net.train()

        lr = learning_rate

        if not no_lr_sche:

            lr = lr_schedule_cosdecay(step,T)

            for param_group in optim.param_groups:

                param_group["lr"] = lr

        x, y = next(iter(loader_train))

        x = x.to(device); y = y.to(device)

        out = net(x)

        loss = criterion[0](out,y)

        if perloss:

            loss2 = criterion[1](out,y)

            loss = loss + 0.04*loss2



        loss.backward()



        optim.step()

        optim.zero_grad()

        losses.append(loss.item())

        print(f'\rtrain loss: {loss.item():.5f} | step: {step}/{steps} | lr: {lr :.7f} | time_used: {(time.time()-start_time)/60 :.1f}',end='',flush=True)



        if step % eval_step ==0 :

            with torch.no_grad():

                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)

            print(f'\nstep: {step} | ssim: {ssim_eval:.4f} | psnr: {psnr_eval:.4f}')



            ssims.append(ssim_eval)

            psnrs.append(psnr_eval)

            if ssim_eval > max_ssim and psnr_eval > max_psnr :

                max_ssim = max(max_ssim,ssim_eval)

                max_psnr = max(max_psnr,psnr_eval)

                torch.save({

                            'step': step,

                            'max_psnr': max_psnr,

                            'max_ssim': max_ssim,

                            'ssims': ssims,

                            'psnrs': psnrs,

                            'losses': losses,

                            'model': net.state_dict()

                }, model_dir)

                print(f'\n model saved at step : {step} | max_psnr: {max_psnr:.4f} | max_ssim: {max_ssim:.4f}')



    np.save(f'./numpy_files/{model_name}_{steps}_losses.npy',losses)

    np.save(f'./numpy_files/{model_name}_{steps}_ssims.npy',ssims)

    np.save(f'./numpy_files/{model_name}_{steps}_psnrs.npy',psnrs)



def test(net, loader_test, max_psnr, max_ssim, step):

    net.eval()

    torch.cuda.empty_cache()

    ssims, psnrs = [], []

    for i, (inputs, targets) in enumerate(loader_test):

        inputs = inputs.to(device); targets = targets.to(device)

        pred = net(inputs)

        # # print(pred)

        # tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')

        # vutils.save_image(targets.cpu(),'target.png')

        # vutils.save_image(pred.cpu(),'pred.png')

        ssim1 = ssim(pred, targets).item()

        psnr1 = psnr(pred, targets)

        ssims.append(ssim1)

        psnrs.append(psnr1)

        #if (psnr1>max_psnr or ssim1 > max_ssim) and s :

#             ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])

#             vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')

#             s=False

    return np.mean(ssims) ,np.mean(psnrs)

%%time



loader_train = loaders_[trainset]

loader_test = loaders_[testset]

net = models_[network]

net = net.to(device)

if device == 'cuda':

    net = torch.nn.DataParallel(net)

    cudnn.benchmark = True

criterion = []

criterion.append(nn.L1Loss().to(device))

if perloss:

    vgg_model = vgg16(pretrained=True).features[:16]

    vgg_model = vgg_model.to(device)

    for param in vgg_model.parameters():

        param.requires_grad = False

    criterion.append(PerLoss(vgg_model).to(device))

optimizer = optim.Adam(params = filter(lambda x: x.requires_grad, net.parameters()), lr=learning_rate, betas=(0.9,0.999), eps=1e-08)

optimizer.zero_grad()

train(net, loader_train, loader_test, optimizer, criterion)
# its or ots

task = 'its'

# test imgs folder

test_imgs = '../input/synthetic-objective-testing-set-sots-reside/indoor/hazy/'



dataset = task

img_dir = test_imgs



output_dir = f'pred_FFA_{dataset}/'

print("pred_dir:",output_dir)



if not os.path.exists(output_dir):

    os.mkdir(output_dir)



ckp = torch.load(model_dir, map_location=device)

net = FFA(gps=gps, blocks=blocks)

net = nn.DataParallel(net)

net.load_state_dict(ckp['model'])

net.eval()



for im in os.listdir(img_dir):

    haze = Image.open(img_dir+im)

    haze1 = tfs.Compose([

        tfs.ToTensor(),

        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])

    ])(haze)[None,::]

    haze_no = tfs.ToTensor()(haze)[None,::]

    with torch.no_grad():

        pred = net(haze1)

    ts = torch.squeeze(pred.clamp(0,1).cpu())

    # tensorShow([haze_no, pred.clamp(0,1).cpu()],['haze', 'pred'])

    

    haze_no = make_grid(haze_no, nrow=1, normalize=True)

    ts = make_grid(ts, nrow=1, normalize=True)

    image_grid = torch.cat((haze_no, ts), -1)

    vutils.save_image(image_grid, output_dir+im.split('.')[0]+'_FFA.png')