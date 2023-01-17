import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from torch import nn

import numpy as np

from torch.autograd import Variable

import torch

from torchvision.datasets import ImageFolder

from torchvision.transforms import transforms

import zipfile

cuda = True if torch.cuda.is_available() else False
data_dir='/kaggle/input/pokemon'

batch_size=256

transform=transforms.Compose([

    transforms.Resize((64,64)),

    transforms.ToTensor(),

    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

])

pokemon=ImageFolder(data_dir,transform)

data_iter=DataLoader(pokemon,batch_size=batch_size,shuffle=True)
fig=plt.figure(figsize=(4,4))

imgs=data_iter.dataset.imgs

for i in range(20):

    img = plt.imread(imgs[i*150][0])

    plt.subplot(4,5,i+1)

    plt.imshow(img)

    plt.axis('off')

plt.show()
class G_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4,strides=2, padding=1):

        super(G_block,self).__init__()

        self.conv2d_trans=nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,

                                             stride=strides, padding=padding, bias=False)

        self.batch_norm=nn.BatchNorm2d(out_channels,0.8)

        self.activation=nn.ReLU()

    def forward(self,x):

        return self.activation(self.batch_norm(self.conv2d_trans(x)))
Tensor=torch.cuda.FloatTensor

x=Variable(Tensor(np.zeros((2,3,16,16))))

g_blk=G_block(3,20)

g_blk.cuda()

print(g_blk(x).shape)
x=Variable(Tensor(np.zeros((2,3,1,1))))

g_blk=G_block(3,20,strides=1,padding=0)

g_blk.cuda()

print(g_blk(x).shape)
class net_G(nn.Module):

    def __init__(self,in_channels):

        super(net_G,self).__init__()

        n_G=64

        self.model=nn.Sequential(

            G_block(in_channels,n_G*8,strides=1,padding=0),

            G_block(n_G*8,n_G*4),

            G_block(n_G*4,n_G*2),

            G_block(n_G*2,n_G),

            nn.ConvTranspose2d(n_G,3,kernel_size=4,stride=2,padding=1,bias=False),

            nn.Tanh()

        )

    def forward(self,x):

        x=self.model(x)

        return x

    

    

def weights_init_normal(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        torch.nn.init.normal_(m.weight.data, mean=0, std=0.02)

    elif classname.find("BatchNorm2d") != -1:

        torch.nn.init.normal_(m.weight.data, mean=1.0, std=0.02)

        torch.nn.init.constant_(m.bias.data, 0.0)
x=Variable(Tensor(np.zeros((1,100,1,1))))

generator=net_G(100)

generator.cuda()

generator.apply(weights_init_normal)

print(generator(x).shape)
alphas = [0, 0.2, 0.4, .6]

x = np.arange(-2, 1, 0.1)

Y = [nn.LeakyReLU(alpha)(Tensor(x)).cpu().numpy()for alpha in alphas]

plt.figure(figsize=(4,4))

for y in Y:

    plt.plot(x,y)

plt.legend()

plt.show()
class D_block(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=4,strides=2,

                 padding=1,alpha=0.2):

        super(D_block,self).__init__()

        self.conv2d=nn.Conv2d(in_channels,out_channels,kernel_size,strides,padding,bias=False)

        self.batch_norm=nn.BatchNorm2d(out_channels,0.8)

        self.activation=nn.LeakyReLU(alpha)

    def forward(self,X):

        return self.activation(self.batch_norm(self.conv2d(X)))
x = Variable(Tensor(np.zeros((2, 3, 16, 16))))

d_blk = D_block(3,20)

d_blk.cuda()

print(d_blk(x).shape)
class net_D(nn.Module):

    def __init__(self,in_channels):

        super(net_D,self).__init__()

        n_D=64

        self.model=nn.Sequential(

            D_block(in_channels,n_D),

            D_block(n_D,n_D*2),

            D_block(n_D*2,n_D*4),

            D_block(n_D*4,n_D*8)

        )

        self.conv=nn.Conv2d(n_D*8,1,kernel_size=4,bias=False)

        self.activation=nn.Sigmoid()

    def forward(self,x):

        x=self.model(x)

        x=self.conv(x)

        x=self.activation(x)

        return x
x = Variable(Tensor(np.zeros((1, 3, 64, 64))))

discriminator=net_D(3)

discriminator.cuda()

discriminator.apply(weights_init_normal)

print(discriminator(x).shape)
def update_D(X,Z,net_D,net_G,loss,trainer_D):

    batch_size=X.shape[0]

    Tensor=torch.cuda.FloatTensor

    ones=Variable(Tensor(np.ones(batch_size,)),requires_grad=False).view(batch_size,1)

    zeros = Variable(Tensor(np.zeros(batch_size,)),requires_grad=False).view(batch_size,1)

    real_Y=net_D(X).view(batch_size,-1)

    fake_X=net_G(Z)

    fake_Y=net_D(fake_X).view(batch_size,-1)

    loss_D=(loss(real_Y,ones)+loss(fake_Y,zeros))/2

    loss_D.backward()

    trainer_D.step()

    return float(loss_D.sum())



def update_G(Z,net_D,net_G,loss,trainer_G):

    batch_size=Z.shape[0]

    Tensor=torch.cuda.FloatTensor

    ones=Variable(Tensor(np.ones((batch_size,))),requires_grad=False).view(batch_size,1)

    fake_X=net_G(Z)

    fake_Y=net_D(fake_X).view(batch_size,-1)

    loss_G=loss(fake_Y,ones)

    loss_G.backward()

    trainer_G.step()

    return float(loss_G.sum())



def train(net_D,net_G,data_iter,num_epochs,lr,latent_dim):

    loss=nn.BCELoss()

    Tensor=torch.cuda.FloatTensor

    trainer_D=torch.optim.Adam(net_D.parameters(),lr=lr,betas=(0.5,0.999))

    trainer_G=torch.optim.Adam(net_G.parameters(),lr=lr,betas=(0.5,0.999))

    plt.figure(figsize=(7,4))

    d_loss_point=[]

    g_loss_point=[]

    d_loss=0

    g_loss=0

    for epoch in range(1,num_epochs+1):

        d_loss_sum=0

        g_loss_sum=0

        batch=0

        for X in data_iter:

            X=X[:][0]

            batch+=1

            X=Variable(X.type(Tensor))

            batch_size=X.shape[0]

            Z=Variable(Tensor(np.random.normal(0,1,(batch_size,latent_dim,1,1))))

            

            trainer_D.zero_grad()

            d_loss = update_D(X, Z, net_D, net_G, loss, trainer_D)

            d_loss_sum+=d_loss

            trainer_G.zero_grad()

            g_loss = update_G(Z, net_D, net_G, loss, trainer_G)

            g_loss_sum+=g_loss

            

        d_loss_point.append(d_loss_sum/batch)

        g_loss_point.append(g_loss_sum/batch)        

        

    plt.ylabel('Loss', fontdict={ 'size': 14})

    plt.xlabel('epoch', fontdict={ 'size': 14})

    plt.xticks(range(0,num_epochs+1,3))

    plt.plot(range(1,num_epochs+1),d_loss_point,color='orange',label='discriminator')

    plt.plot(range(1,num_epochs+1),g_loss_point,color='blue',label='generator')

    plt.legend()

    plt.show()



    Z = Variable(Tensor(np.random.normal(0, 1, size=(21, latent_dim, 1, 1))),requires_grad=False)

    fake_x = generator(Z)

    fake_x=fake_x.cpu().detach().numpy()

    plt.figure(figsize=(14,6))

    for i in range(21):

        im=np.transpose(fake_x[i])

        plt.subplot(3,7,i+1)

        plt.imshow(im)

    plt.show()
if __name__ == '__main__':

    lr,latent_dim,num_epochs=0.005,100,20

    train(discriminator,generator,data_iter,num_epochs,lr,latent_dim)