from torch.utils.data import Dataset

import torch

import torch.nn as nn

from torchvision import transforms

import torchvision.models as models

from torch.utils.data import DataLoader

from torch.optim import Adam

import PIL

import matplotlib.pyplot as plt

import numpy as np

import pickle as pkl

import random
#need to implement __len__() nd __getitem__() for DataLoader

class MyDataset(Dataset):



    def __init__(self):

        

        self.transform = transforms.Compose([

            transforms.Resize(256),

            transforms.CenterCrop(256),

            transforms.ToTensor(),

        ])



        self.dataset = pkl.load(open("/kaggle/input/dataset.data", 'rb'))

        self.length = len(self.dataset)

        self.indexes = [idx for idx in range(self.length)]



        random.shuffle(self.indexes)



    def __len__(self):

        return self.length



    def __getitem__(self, idx):

        real_idx = self.indexes[idx]

        data = self.dataset[real_idx]

        

        random.shuffle(data)



        data_array = []

        for d in data:

            x = self.transform(PIL.Image.fromarray(d['frame'], 'RGB'))

            y = self.transform(plot_landmarks(d['frame'], d['landmarks']))

            data_array.append(torch.stack((x, y)))

        data_array = torch.stack(data_array)



        return real_idx, data_array



#taken from here https://github.com/grey-eye/talking-heads/blob/1f8edb68ffa7b7a8e5d02a19ca39bdec5fc2d30c/dataset/dataset.py

#plots the vector landmarks from FaceAlignment and plots on figure with dimmension equal with frame and RGB encodes them

def plot_landmarks(frame, landmarks):

    dpi = 100

    fig = plt.figure(figsize=(frame.shape[0] / dpi, frame.shape[1] / dpi), dpi=dpi)

    ax = fig.add_subplot(111)

    ax.axis('off')

    plt.imshow(np.ones(frame.shape))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)



    # Head

    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)

    # Eyebrows

    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)

    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)

    # Nose

    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)

    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)

    # Eyes

    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)

    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)

    # Mouth

    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)



    fig.canvas.draw()

    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)

    plt.close(fig)

    return data

#Taken from here https://github.com/dxyang/StyleTransfer/blob/master/network.py

#implementation of Perceptual Losses for Real-Time Style Transfer and Super-Resolution

#Justin Johnson, Alexandre Alahi, Li Fei-Fei

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):

        super(ConvLayer, self).__init__()

        padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(padding)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) #, padding)



    def forward(self, x):

        out = self.reflection_pad(x)

        out = self.conv2d(out)

        return out



class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):

        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample

        if upsample:

            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)



    def forward(self, x):

        if self.upsample:

            x = self.upsample(x)

        out = self.reflection_pad(x)

        out = self.conv2d(out)

        return out



class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super(ResidualBlock, self).__init__()

        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)

        self.in1 = nn.InstanceNorm2d(channels, affine=True)

        self.relu = nn.ReLU()

        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)

        self.in2 = nn.InstanceNorm2d(channels, affine=True)



    def forward(self, x):

        residual = x

        out = self.relu(self.in1(self.conv1(x)))

        out = self.in2(self.conv2(out))

        out = out + residual

        out = self.relu(out)

        return out 
from torch.nn import init

#Taken from here https://github.com/sxhxliang/BigGAN-pytorch/blob/master/model_resnet.py

#implementation of Large scale gan training for high fidelity natural image synthesis

#K. S. Andrew Brock, Jeff Donahue

def init_conv(conv, glu=True):

    init.xavier_uniform_(conv.weight)

    if conv.bias is not None:

        conv.bias.data.zero_()

        

class SelfAttention(nn.Module):

    """ Self attention Layer"""

    def __init__(self,in_dim,activation=torch.nn.functional.relu):

        super(SelfAttention,self).__init__()

        self.chanel_in = in_dim

        self.activation = activation

        

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)

        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)

        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.gamma = nn.Parameter(torch.zeros(1))



        self.softmax  = nn.Softmax(dim=-1) #



        init_conv(self.query_conv)

        init_conv(self.key_conv)

        init_conv(self.value_conv)

        

    def forward(self,x):

        """

            inputs :

                x : input feature maps( B X C X W X H)

            returns :

                out : self attention value + input feature 

                attention: B X N X N (N is Width*Height)

        """

        m_batchsize,C,width ,height = x.size()

        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)

        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)

        energy =  torch.bmm(proj_query,proj_key) # transpose check

        attention = self.softmax(energy) # BX (N) X (N) 

        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N



        out = torch.bmm(proj_value,attention.permute(0,2,1) )

        out = out.view(m_batchsize,C,width,height)

        

        out = self.gamma*out + x

        return out
#same as the ResidualBlock w/o the Norm Layer

class ResidualDownsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):

        super(ResidualDownsamplingBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.relu = nn.ReLU()

        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=kernel_size, stride=stride)

        

        self.res = ConvLayer(in_channels, out_channels, 1, 1)



    def forward(self, x):

        residual = x

        out = self.relu(self.conv1(x))

        out = self.conv2(out)

        

        residual = self.res(residual)

        out = out + residual

        out = self.relu(out)

        return out 

    

class ResidualUpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample):

        super(ResidualUpsamplingBlock, self).__init__()

        self.up_conv1 = UpsampleConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, upsample = upsample)

        self.relu = nn.ReLU()

        self.up_conv2 = UpsampleConvLayer(out_channels, out_channels, kernel_size=kernel_size, stride=stride, upsample = upsample)



        self.res = UpsampleConvLayer(in_channels, out_channels, 1, 1)

        

    def forward(self, x):

        residual = x

        out = self.relu(self.up_conv1(x))

        out = self.up_conv2(out)

        

        residual = self.res(residual)

        out = out + residual

        out = self.relu(out)

        return out 
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv2d') != -1 or classname.find('InstanceNorm2d') != -1:

        nn.init.xavier_uniform_(m.weight.data)
#Based on https://github.com/dxyang/StyleTransfer/blob/master/network.py

#implementation of Perceptual Losses for Real-Time Style Transfer and Super-Resolution

#Justin Johnson, Alexandre Alahi, Li Fei-Fei

#replaced downsampling and upsampling layers with residual blocks

class GeneratorNetwork(nn.Module):

    def __init__(self):

        super(GeneratorNetwork, self).__init__()

        

        # nonlineraity

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()



        # encoding layers

        self.res_d1 = ResidualDownsamplingBlock(6, 32, kernel_size=9, stride=1)

        self.in1_e = nn.InstanceNorm2d(32, affine=True)



        self.res_d2 = ResidualDownsamplingBlock(32, 64, kernel_size=3, stride=2)

        self.in2_e = nn.InstanceNorm2d(64, affine=True)



        self.res_d3 = ResidualDownsamplingBlock(64, 128, kernel_size=3, stride=2)

        self.in3_e = nn.InstanceNorm2d(128, affine=True)



        # residual layers

        self.res1 = ResidualBlock(128)

        self.res2 = ResidualBlock(128)

        self.res3 = ResidualBlock(128)

        self.res4 = ResidualBlock(128)

        self.res5 = ResidualBlock(128)



        # decoding layers

        self.deres_u3 = ResidualUpsamplingBlock(128, 64, kernel_size=3, stride=1, upsample=2 )

        self.in3_d = nn.InstanceNorm2d(64, affine=True)



        self.deres_u2 = ResidualUpsamplingBlock(64, 32, kernel_size=3, stride=1, upsample=2 )

        self.in2_d = nn.InstanceNorm2d(32, affine=True)



        self.deres_u1 = ResidualUpsamplingBlock(32, 3, kernel_size=9, stride=1, upsample=2 )

        self.in1_d = nn.InstanceNorm2d(3, affine=True)

        

        #self.apply(weights_init)



    def forward(self, x, embedder_vect):

        # still work to do on embedder_vect

        # encode

        y = self.relu(self.in1_e(self.res_d1(x)))

        y = self.relu(self.in2_e(self.res_d2(y)))

        y = self.relu(self.in3_e(self.res_d3(y)))



        # residual layers

        y = self.res1(y)

        y = self.res2(y)

        y = self.res3(y)

        y = self.res4(y)

        y = self.res5(y)



        # decode

        y = self.relu(self.in3_d(self.deres_u3(y)))

        y = self.relu(self.in2_d(self.deres_u2(y)))

        #y = self.tanh(self.in1_d(self.deconv1(y)))

        y = self.deres_u1(y)



        return y
#same as the GeneratorNetwork downsampling, without the Norm layers

#selfAttention layer at 32x32

#finally performs a global sum pooling over spatial dimensions followed by ReLU

class EmbedderNetwork(nn.Module):

    def __init__(self):

        super(EmbedderNetwork, self).__init__()

        

        # nonlineraity

        self.relu = nn.ReLU()



        # encoding layers

        self.res_d1 = ResidualDownsamplingBlock(6, 32, kernel_size=9, stride=1)

        self.res_d2 = ResidualDownsamplingBlock(32, 256, kernel_size=3, stride=2)

        self.att = SelfAttention(256)

        self.res_d3 = ResidualDownsamplingBlock(256, 512, kernel_size=3, stride=2)

        

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        

        self.apply(weights_init)



    def forward(self, x):

        # encode

        y = self.relu(self.res_d1(x))

        y = self.relu(self.res_d2(y))

        y = self.att(y)

        y = self.relu(self.res_d3(y))

        y = self.relu(self.pooling(y).view(-1, 512))



        return y

#same as the EmbedderNetwork with an additional residual block at the end

#selfAttention layer at 32x32

#finally performs a global sum pooling over spatial dimensions followed by ReLU

class DiscriminatorNetwork(nn.Module):

    def __init__(self, number_videos):

        super(DiscriminatorNetwork, self).__init__()

        

        # nonlineraity

        self.relu = nn.ReLU()



        # encoding layers

        self.res_d1 = ResidualDownsamplingBlock(6, 32, kernel_size=9, stride=1)

        self.res_d2 = ResidualDownsamplingBlock(32, 256, kernel_size=3, stride=2)

        self.att = SelfAttention(256)

        self.res_d3 = ResidualDownsamplingBlock(256, 512, kernel_size=3, stride=2)

        self.res_block = ResidualBlock(512)

        

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

                

        self.W = nn.Parameter(torch.rand(512, number_videos))

        self.w_0 = nn.Parameter(torch.rand(512, 1))

        self.b = nn.Parameter(torch.rand(1))

                                

    def forward(self, x, idx):

        # encode

        y = self.relu(self.res_d1(x))

        y = self.relu(self.res_d2(y))

        y = self.att(y)

        y = self.relu(self.res_d3(y))

        y = self.relu(self.res_block(y))

        y = self.relu(self.pooling(y).view(-1, 512))

        

        _W_i = (self.W[:, i].unsqueeze(-1)).transpose(0, 1)

        y = torch.bmm(y.transpose(1, 2), _W_i + self.w_0) + self.b

                                

        return y
raw_dataset = MyDataset()

train_set, test_set = torch.utils.data.random_split(raw_dataset,[500, 32])

train_set = DataLoader(train_set, batch_size=3, shuffle=True)

test_set = DataLoader(train_set, batch_size=1, shuffle=True)



Embedder = EmbedderNetwork()

Generator = GeneratorNetwork()

Discriminator = DiscriminatorNetwork(len(train_set))



optimizer1 = Adam(params=list(Embedder.parameters()) + list(Generator.parameters()),lr=5e-5)

optimizer2 = Adam(params=Discriminator.parameters(),lr=2e-4)



Vgg19 = models.vgg19(pretrained=True)



def criterion_vgg19(model, original, generated):

    original_vgg19 = model(original)

    generated_vgg19 =  model(generated)

    for i in range(0, len(original_vgg19)):

            vgg19_loss += torch.nn.functional.l1_loss(generated_vgg19[i], original_vgg19[i])

    return vgg19_loss



def criterion_Discriminator(original, generated):

    return max(0.1 + generated)+ max(0.1 - original)
for epoch in range(3):

    Embedder.train()

    Generator.train()

    Discriminator.train()

    for batch_num,(idx, video) in enumerate(train_set):



        # video [Batch, 8+1, 2, Chanels, Width, Height]

        

        t_frame = video[:, -1, ...]  # [Batch, 2, Chanels, Width, Height]

        video = video[:, :-1, ...]  # [Batch, 8, Chanels, Width, Height]

        dims = video.shape



        

        embedder_in = video.reshape(dims[0] * dims[1], dims[2], dims[3], dims[4], dims[5])  # [Batch * 8, 2, Chanels, Width, Height]

        embedder_frames, embedder_landmarks = embedder_in[:, 0, ...], embedder_in[:, 1, ...]

        del embedder_in

        #adding them together and forming an 6 chanel images varray

        embedder_in = torch.cat((embedder_frames, embedder_landmarks), dim=1)

        

        embedder_vectors = Embedder(embedder_in).reshape(dims[0], dims[1], -1)  # Batch, 8

        embedder_average = embedder_vectors.mean(dim=1)



        del embedder_in

        del embedder_frames

        del embedder_landmarks

        del embedder_vectors

        

        generator_frame, generator_landmark = t_frame[:, 0, ...],t_frame[:, 1, ...]

        generator_result = Generator(generator_landmark, embedder_average)



        #adding them together and forming an 6 chanel images varray

        original_in = torch.cat((generator_frame, generator_landmark), dim=1)

        generated_in = torch.cat((generator_result, generator_landmark), dim=1)

        

        discriminator_original= Discriminator(original_in, idx)

        discriminator_generated = Discriminator(generated_in, idx)



        optimizer1.zero_grad()

        optimizer2.zero_grad()



        loss_vgg19 = criterion_vgg19(Vgg19, discriminator_original, discriminator_generated)

        loss_Discriminator = criterion_Discriminator(discriminator_original, discriminator_generated)

        loss = loss_vgg19 + loss_Discriminator

        loss.backward()



        optimizer1.step()

        optimizer2.step()



        # Optimize Discriminator again

        generator_result = Generator(generator_landmark, embedder_average)

        generated_in = torch.cat((generator_result, generator_landmark), dim=1)



        discriminator_original= Discriminator(original_in, idx)

        discriminator_generated = Discriminator(generated_in, idx)



        optimizer2.zero_grad()

        loss_Discriminator = criterion_Discriminator(discriminator_original, discriminator_generated)

        loss_Discriminator.backward()

        optimizer2.step()
