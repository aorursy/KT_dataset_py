import cv2
import os
MAIN_PATH='/kaggle/input/anime-faces/faces/faces/'
def get_imgs():
    files = os.listdir(MAIN_PATH)
    imgs = []
    for file in files:
        imgs.append(cv2.imread(MAIN_PATH + file))
    print("get_imgs")
    return imgs
real_img = get_imgs()
import numpy as np
import matplotlib.pyplot as plt
def imshow(img):
    b,g,r = cv2.split(img)
    rgb = cv2.merge((r,g,b))
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
imshow(np.hstack((real_img[22],real_img[184],real_img[5884])))
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torchvision import transforms

 
trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ]
)
G_LR = 0.0002
D_LR = 0.0002
BATCHSIZE = 50
 
def init_ws_bs(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.normal_(m.weight.data, std=0.2)
        init.normal_(m.bias.data, std=0.2)
 
 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(  # stride(input_w-1)+k-2*Padding
                in_channels=100,
                out_channels=64 * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(  # stride(input_w-1)+k-2*Padding
                in_channels=64 * 8,
                out_channels=64 * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(  # stride(input_w-1)+k-2*Padding
                in_channels=64 * 4,
                out_channels=64 * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(inplace=True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(  # stride(input_w-1)+k-2*Padding
                in_channels=64 * 2,
                out_channels=64 * 1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 5, 3, 1, bias=False),
            nn.Tanh(),
        )
 
    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x
 
 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # batchsize,3,96,96
                in_channels=3,
                out_channels=64,
                kernel_size=5,
                padding=1,
                stride=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(.2, inplace=True),
 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False, ),  # batchsize,16,32,32
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(.2, inplace=True),
 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(.2, inplace=True),
 
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(.2, inplace=True),
 
        )
        self.output = nn.Sequential(
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  #
        )
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output(x)
        return x
 
 
g = Generator().cuda()
d = Discriminator().cuda()
 
init_ws_bs(g), init_ws_bs(d)
 
g_optimizer = torch.optim.Adam(g.parameters(), betas=(.5, 0.999), lr=G_LR)
d_optimizer = torch.optim.Adam(d.parameters(), betas=(.5, 0.999), lr=D_LR)
 
g_loss_func = nn.BCELoss().cuda()
d_loss_func = nn.BCELoss().cuda()
 
label_real = torch.ones(BATCHSIZE).cuda()
label_fake = torch.zeros(BATCHSIZE).cuda()
checkpoint = torch.load("../input/anime-faces/200.pkl")
g.load_state_dict(checkpoint['g'])
d.load_state_dict(checkpoint['d'])
start_epoch = checkpoint['epoch']+1
print(start_epoch)
EPOCHES = 201
for epoch in range(start_epoch, EPOCHES):
    np.random.shuffle(real_img)
    count = 0
    batch_imgs = []
    for i in range(len(real_img)):
        count = count + 1
        batch_imgs.append(trans(real_img[i]).numpy())  # tensor类型#这里经过trans操作通道维度从第四个到第二个了
 
        if count == BATCHSIZE:
            count = 0
 
            batch_real = torch.Tensor(batch_imgs).cuda()
            batch_imgs.clear()
            d_optimizer.zero_grad()
            pre_real = d(batch_real).squeeze()
            d_real_loss = d_loss_func(pre_real, label_real)
             
            batch_fake = torch.randn(BATCHSIZE, 100, 1, 1).cuda()
            img_fake = g(batch_fake).detach()
            pre_fake = d(img_fake).squeeze()
            d_fake_loss = d_loss_func(pre_fake, label_fake)
            
            d_loss = d_fake_loss+d_real_loss
            
            if d_loss > 0.01:
                d_loss.backward()
           
            d_optimizer.step()
 
            g_optimizer.zero_grad()
            batch_fake = torch.randn(BATCHSIZE, 100, 1, 1).cuda()
            img_fake = g(batch_fake)
            pre_fake = d(img_fake).squeeze()
            g_loss = g_loss_func(pre_fake, label_real)
            
            if g_loss > 0.01:
                g_loss.backward()
            g_optimizer.step()
        if count == BATCHSIZE:
            print(i,(d_real_loss + d_fake_loss).detach().cpu().numpy(), g_loss.detach().cpu().numpy(), epoch)
    
state = { 'g': g.state_dict(),'d':d.state_dict(), 'epoch': epoch }   
torch.save(state, '1000.pkl')
torch.save(g,'g1000.pkl')
import cv2
imgs= g(torch.randn(10,100,1,1).cuda())
for i in range(len(imgs)):
    img=imgs[i].cpu().permute(1,2,0).detach().numpy()
    b,g,r = cv2.split(img)
    rgb = cv2.merge((r,g,b))
    plt.imshow(rgb)
    plt.show()
