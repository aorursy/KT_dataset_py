#Get some packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from skimage.io import imshow
import time
import random
#Read CSV
csv = pd.read_csv('../input/train.csv')
#Separate into matricies
X_train = csv.iloc[:,1:786].as_matrix()
Y_train = csv.iloc[:,0].as_matrix()
X_train_imgs = np.zeros([42000,1,28,28])
for i in range(X_train.shape[0]):
    img = X_train[i,:].reshape([1,28,28])/255.
    X_train_imgs[i] = img
Y_train_oh = np.zeros([42000,10])
for i in range(Y_train.shape[0]):
    oh = np.zeros([10])
    oh[int(Y_train[i])] = 1.
    Y_train_oh[i] = oh
ix = 599 #0-42000
imshow(np.squeeze(X_train_imgs[ix]))
plt.show()
print ('This is:',Y_train[ix])
class _G(nn.Module):
    def __init__(self, z_size, c_size):
        super(_G, self).__init__()
        
        self.conv2dtranspose_z = nn.ConvTranspose2d(in_channels=z_size, out_channels=256, kernel_size=4, stride=1)
        self.bn2d_z = nn.BatchNorm2d(256, momentum=0.9)
        self.conv2dtranspose_c = nn.ConvTranspose2d(in_channels=c_size, out_channels=256, kernel_size=4, stride=1)
        self.bn2d_c = nn.BatchNorm2d(256, momentum=0.9)
        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=2, stride=2, padding=2),
            nn.Tanh()
        )
    
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self, z, c):
        z = F.leaky_relu(self.bn2d_z(self.conv2dtranspose_z(z.view(-1,100,1,1))))
        c = F.leaky_relu(self.bn2d_c(self.conv2dtranspose_c(c.view(-1,10,1,1))))
        zc = torch.cat([z,c],dim=1)
        output = self.backbone(zc)
        return output
class _D(nn.Module):
    def __init__(self,c_size):
        super(_D, self).__init__()
        
        self.conv2d_x = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2d_c = nn.Conv2d(in_channels=10, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=2),
            nn.Sigmoid()
        )
    
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
    
    def forward(self, x, c):
        x = self.conv2d_x(x)
        c = c.view(-1,10,1,1)
        c = c.expand(-1,10,28,28)
        c = self.conv2d_c(c)
        xc = torch.cat([x,c],dim=1)
        output = self.backbone(xc)
        output = output.view(-1,1)
        return output
G = _G(100, 10) #Noise vector will have size 100, and we will have a condition vector of 10(1 for each type of item)
D = _D(10) #The Discriminator will also use the condition, so we say it has size 10
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
G.weight_init(mean=0, std=0.2) #GAN works better with these weight initializations
D.weight_init(mean=0, std=0.2)
G.cuda()
D.cuda()
#That's how you move it to the GPU
criterion = nn.BCELoss()#Binary Cross-Entropy Loss
optim_G = optim.Adam(G.parameters(), lr=0.0002)
optim_D = optim.Adam(D.parameters(), lr=0.0002)
def optimize_G(G, D, z, c, optimizer, criterion):
    
    """
    When we train the generator we want it to trick the discriminator. This means that we want the output of D to be close to 1,
    meaning it thinks its real. Keep that in mind. When we train G, we make the fake labels equal 1 so the optimizer tried to
    make the generator make an image that tricks D.
    """
    
    #Even though the images are fake, we want the discriminator to think they are real
    trick_labels = Variable(torch.ones([z.shape[0],1])-torch.rand([z.shape[0],1])/3).cuda()
    #Zero gradient buffers
    G.zero_grad()
    #Generate Images
    fake_x = G.forward(z, c)
    D_preds = D(fake_x, c)
    loss = criterion(D_preds, trick_labels)
    loss.backward()
    optimizer.step()
    
    return fake_x, loss
def optimize_D(net, fake_x, fake_c, real_x, real_c, optimizer, criterion):
    #We cannot feed a numpy variable. We have to use a torch.autograd.Variable
    fake_labels = Variable(torch.zeros([fake_x.shape[0],1])+torch.rand([z.shape[0],1])/3).cuda()
    real_labels = Variable(torch.ones([real_x.shape[0],1])-torch.rand([z.shape[0],1])/3).cuda()
    
    #We need to empty the gradient buffers
    net.zero_grad()
    
    #Let's get the discriminator predictions for the fake images
    fake_preds = net.forward(fake_x.detach(), fake_c)
    #Do the optimization
    fake_loss = criterion(fake_preds, fake_labels)
    #Let's get the discriminator predictions for the real images
    real_preds = net.forward(real_x, real_c)
    #Do the optimization
    real_loss = criterion(real_preds, real_labels)
    
    loss = fake_loss + real_loss
    loss.backward()
    optimizer.step()
    
    return fake_loss + real_loss
D_history = []
G_history = []
EPOCHS = 10
BATCH_SIZE = 128

for epoch in range(EPOCHS):
    train_loss = 0
    speed = 0
    for batch_number in range(int(Y_train.shape[0]/BATCH_SIZE)):
        G.train()
        time_start = time.time()
        real_x = Variable(torch.FloatTensor(X_train_imgs[batch_number*BATCH_SIZE:(1+batch_number)*BATCH_SIZE])).cuda()
        real_x = (real_x-real_x.mean())/real_x.std()
        real_c = Variable(torch.FloatTensor(Y_train_oh[batch_number*BATCH_SIZE:(1+batch_number)*BATCH_SIZE])).cuda()
        
        z = Variable(torch.FloatTensor(np.random.randn(BATCH_SIZE, 100))).cuda()
        fake_x, loss = optimize_G(G, D, z, real_c, optim_G, criterion)
        G_history.append(loss.data.cpu().numpy()[0])
        
        loss = optimize_D(D, fake_x, real_c, real_x, real_c, optim_D, criterion)
        D_history.append(loss.data.cpu().numpy()[0])
        
        
        if batch_number % 25 == 0:
            bigfig = []
            for i in range(0,10):
                z = np.random.randn(1,100)
                z = torch.FloatTensor(z)
                z = Variable(z).cuda()
                G.eval()
                fig = []
                for i in range(0, 10):
                    c = np.zeros([1,10])
                    c[0,i] = 1.
                    c = torch.FloatTensor(c)
                    c = Variable(c).cuda()
                    gens = G.forward(z, c)
                    gens = gens.data.cpu().numpy()
                    gens = gens.reshape([28,28])
                    fig.append(gens/2+0.5)
                fig = np.hstack(fig)
                bigfig.append(fig)
            bigfig = np.vstack(bigfig)
            print (bigfig.shape)
            imshow(bigfig)
            plt.show()
            
            print ('G loss: ',G_history[-1])
            print ('D loss: ',D_history[-1])
            print ('D loss variance: ',np.stack(D_history,axis=0).std())
    print ('Finished Epoch',epoch+1)

