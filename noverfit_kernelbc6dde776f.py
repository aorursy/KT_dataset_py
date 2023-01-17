#datset

import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)
label = {"face": 1}

class datast(Dataset):
    def __init__(self,datadir,transform=None):
        
        self.label = {'face':1}
        self.transform = transform
        self.data_info = self.get_data_info(datadir)
    
    def __getitem__(self,index):
        
        img,label = self.data_info[index]
        img = Image.open(img).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img,label
    
    def __len__(self):
        return len(self.data_info)
    
    
    @staticmethod
    def get_data_info(data_dir):
        data_info = []
        for root,dirs,file_name in os.walk(data_dir):            
                
            img_dir = os.listdir(root)
            img = list(filter(lambda x:x.endswith('.jpg'),img_dir))     
                 
            for i in range(len(img)):
                img_name = img[i]
                label = 'face'
                img_path = os.path.join(os.path.join(root,img_name))
                data_info.append((img_path,label))
                    
        return data_info

import torch.autograd
import random
import os 
import sys
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms,datasets
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader,Dataset

train_dataset = '../input/animation-avatar-data-set/faces'
# 创建文件夹
if not os.path.exists('./img'):
    os.mkdir('./img')

#输出图像
def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 3, 64, 64)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out
batch_size =20
num_epoch = 500
z_dimension =100
# 图像预处理

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

img_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
mydata = datast(datadir=train_dataset,transform =img_transform )
sampler = torch.utils.data.SequentialSampler(train_dataset)

# data loader 数据载入
dataloader = DataLoader(
    dataset=mydata, sampler = sampler,batch_size=batch_size, shuffle=False
)
# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片64*64展开成4096，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class discriminator(nn.Module):
    def __init__(self):       
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),  # batch, 32, 64, 64
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 32, 32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 32,32
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 16, 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2),  # batch, 64, 32,32
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 8, 8
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# ###### 定义生成器 Generator #####
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到4096*3维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成3*64*64维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc = nn.Linear(4096, 4096*3)  # batch, 4096
        self.br = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3, 50, 3, stride=1, padding=1),  # batch, 50, 64, 64
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 64, 64
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 3, 3, stride=1,padding=1),  # batch, 3, 64, 64
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 3, 64, 64)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x

# 创建对象
D = discriminator()
G = generator()
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()
# 首先需要定义loss的度量方式  （二分类的交叉熵）
# 其次定义 优化函数,优化函数的学习率为0.0003
criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)
# ##########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch):  # 进行多个epoch的训练
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # view()函数作用是将一个多行的Tensor,拼接成一行
        # 第一个参数是要拼接的tensor,第二个参数是-1
        # =============================训练判别器==================
        # img = img.view(num_img, -1)  # 将图片展开为28*28=784
        real_img = Variable(img).cuda()  # 将tensor变成Variable放入计算图中
        
        real_label = Variable(torch.ones(num_img)).cuda()  # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(num_img)).cuda()  # 定义假的图片的label为0
 
        # ########判别器训练train#####################
        # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
        # 计算真实图片的损失
        real_out = D(real_img).cuda()  # 将真实图片放入判别器中
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
        # 计算假的图片的损失
        z = Variable(torch.randn(num_img, 64*64)).cuda()  # 随机生成一些噪声
        fake_img = G(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
        fake_out = D(fake_img)  # 判别器判断假的图片，
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数
        
        # ==================训练生成器============================
        # ###############################生成网络的训练###############################
        # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
        # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        # 反向传播更新的参数是生成网络里面的参数，
        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
        # 这样就达到了对抗的目的
        # 计算假的图片的损失
        z = Variable(torch.randn(num_img, 64*64)).cuda()  # 得到随机噪声
        fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
        output = D(fake_img)  # 经过判别器得到的结果
        g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss
        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
        
        # 打印中间的损失
        if (i + 1) % 1 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
            ))
        if epoch == 0:
            real_images = to_img(real_img.gpu().data)
            save_image(real_images, './img/real_images.png')
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))
    