import torch, os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
%matplotlib inline
lr = 0.0002
max_epoch = 8
batch_size = 32
z_dim = 100
image_size = 64
g_conv_dim = 64
d_conv_dim = 64
log_step = 100
sample_step = 500
sample_num = 32
IMAGE_PATH = '../input/celeba-dataset/img_align_celeba/'
SAMPLE_PATH = '../'

if not os.path.exists(SAMPLE_PATH):
    os.makedirs(SAMPLE_PATH)
transform = transforms.Compose([
    transforms.Scale(image_size),
    # transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(IMAGE_PATH, transform)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
def conv(ch_in, ch_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)
        self.fc = conv(conv_dim*8, 1, int(image_size/16), 1, 0, False)
        
    def forward(self, x):                                 # if image_size is 64, output shape is below
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 32, 32)     
        out = F.leaky_relu(self.conv2(out), 0.05) # (?, 128, 16, 16)
        out = F.leaky_relu(self.conv3(out), 0.05) # (?, 256, 8, 8)
        out = F.leaky_relu(self.conv4(out), 0.05) # (?, 512, 4, 4)
        out = F.sigmoid(self.fc(out)).squeeze()
        return out
    
D = Discriminator(image_size)
D.cuda()
def deconv(ch_in, ch_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_dim=256, image_size=128, conv_dim=64):
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim*8, int(image_size/16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)
        
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        out = F.leaky_relu(self.deconv2(out), 0.05)
        out = F.leaky_relu(self.deconv3(out), 0.05)
        out = F.tanh(self.deconv4(out))
        return out
    
G = Generator(z_dim, image_size, g_conv_dim)
G.cuda()
criterion = nn.BCELoss().cuda()
d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# denormalization : [-1,1] -> [0,1]
# normalization : [0,1] -> [-1,1]
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)
try:
    G.load_state_dict(torch.load('generator.pkl'))
    D.load_state_dict(torch.load('discriminator.pkl'))
    print("\n-------------model restored-------------\n")
except:
    print("\n-------------model not restored-------------\n")
    pass
total_batch = len(data_loader.dataset)//batch_size
fixed_z = Variable(torch.randn(sample_num, z_dim)).cuda()
for epoch in range(max_epoch):
    for i, (images, labels) in enumerate(data_loader):
        # Build mini-batch dataset
        image = Variable(images).cuda()
        # Create the labels which are later used as input for the BCE loss
        real_labels = Variable(torch.ones(batch_size)).cuda()
        fake_labels = Variable(torch.zeros(batch_size)).cuda()
        
        #============ train the discriminator ============
        # Compute BCE_loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels = 1
        outputs = D(image)
        d_loss_real = criterion(outputs, real_labels) # BCE
        real_score = outputs
        
        # compute BCE_loss using fake images
        z = Variable(torch.randn(batch_size, z_dim)).cuda()
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels) #BCE
        fake_score = outputs
        
        # Backprob + Optimize
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        #============ train the generator ============
        # Compute loss with fake images
        z = Variable(torch.randn(batch_size, z_dim)).cuda()
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z))) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels) # BCE
        
        # Backprob + Optimize
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if(i+1)%log_step == 0:
            print("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f"%(
            epoch, max_epoch, i+1, total_batch, d_loss.data[0], g_loss.data[0], real_score.data.mean(), fake_score.data.mean()))
        
        if(i+1)%sample_step == 0:
            fake_images = G(fixed_z)
            torchvision.utils.save_image(denorm(fake_images.data), os.path.join(SAMPLE_PATH, 'fake_samples-%d-%d.png')%(
            epoch+1, i+1), nrow=8)
            
torch.save(G.state_dict(), 'generator.pkl')
torch.save(D.state_dict(), 'discriminator.pkl')
fixed_z = Variable(torch.randn(sample_num, z_dim)).cuda()
fake_images = G(fixed_z)
plt.imshow(denorm(fake_images[0].cpu().permute(1, 2, 0).data).numpy())
plt.show()

plt.imshow(make_grid(denorm(fake_images).data.cpu()).permute(1, 2, 0).numpy())
plt.show()
import imageio

images = []
for epoch in range(max_epoch):
    for i in range(total_batch):
        if(i+1)%sample_step == 0:
            img_name = '../fake_samples-' + str(epoch + 1) + '-' + str(i + 1) + '.png'
            images.append(imageio.imread(img_name))
            print("epoch : {}, step : {}".format(epoch+1, i+1))
imageio.mimsave('dcgan_celebA_generation_animation.gif', images, fps=5)